"""SurfaceDetection - IA geometrica de superficies universales.

Detecta lo que toda escena tiene: suelo, paredes, mesas, techo. No usa otra red
neuronal -eso costaria decenas de milisegundos y arrastraria los fps- sino la
estructura geometrica que esas superficies comparten en cualquier lugar del
mundo: son PLANOS, y su orientacion los delata. El suelo es un plano horizontal
bajo la camara; una mesa, un plano horizontal por encima del suelo; una pared,
un plano vertical. Eso es lo universal, y es detectable con RANSAC en ~3 ms.

La deteccion hace doble servicio:

1. **Semantica**: el visor muestra que superficies hay y cuanto ocupan.
2. **Estabilidad**: la profundidad de los pixeles que pertenecen a un plano se
   proyecta SOBRE el plano, y los parametros del plano se suavizan entre frames
   (EMA). Un suelo detectado deja de temblar por completo: es la mayor mejora
   de quietud posible, porque las superficies grandes son justo donde el ruido
   de la red mas se nota.

Convenio de camara OpenCV: x derecha, y ABAJO, z hacia delante. "Arriba" es
(0,-1,0) asumiendo el movil en vertical; cuando haya pose de la IMU (fase 2),
este vector vendra del acelerometro y la clasificacion sera exacta tambien con
el movil inclinado.
"""

from __future__ import annotations

import numpy as np

#: "arriba" en el sistema de camara, con el movil en vertical
UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)

#: nombres en el orden en que conviene ensenarlos
KIND_ORDER = ("suelo", "mesa", "pared", "techo", "superficie")


class SurfaceDetection:
    def __init__(self, max_planes: int = 4, hypotheses: int = 64):
        self.max_planes = max_planes
        self.hypotheses = hypotheses
        self._rng = np.random.default_rng(7)
        #: planos persistentes: dicts con n, d, kind, area, missing
        self._planes: list[dict] = []

    def reset(self) -> None:
        self._planes = []
        self._prev_final: np.ndarray | None = None
        #: mascara HxW de pixeles que pertenecen a algun plano confirmado
        self.plane_mask: np.ndarray | None = None

    def settle(self, depth: np.ndarray, far: float) -> np.ndarray:
        """EMA final sobre la profundidad ya refinada.

        Existe porque el refinado guiado por imagen va DESPUES del filtro
        temporal del estimador: la imagen tiembla con el pulso, la guia cambia y
        el resultado final vuelve a temblar aunque la entrada estuviera quieta.
        Este es el ultimo paso de la cadena, asi que lo que sale de aqui es
        exactamente lo que se malla: donde el cambio es pequeno (ruido) manda el
        historico, donde es grande (movimiento real) manda el frame nuevo.
        """
        prev = getattr(self, "_prev_final", None)
        if prev is None or prev.shape != depth.shape:
            self._prev_final = depth.copy()
            return depth
        diff = np.abs(depth - prev)
        rel = diff / np.maximum(depth, 0.2)
        base = 0.55 + 0.40 * np.clip(depth / max(far, 1.0) - 0.35, 0.0, 1.0)
        w = np.minimum(base, 0.95) * np.exp(-(rel / 0.10) ** 2)
        out = (w * prev + (1.0 - w) * depth).astype(np.float32)
        self._prev_final = out
        return out

    # -- ciclo principal ----------------------------------------------------
    def process(self, depth: np.ndarray, rays: np.ndarray, snap: float,
                far_limit: float) -> tuple[np.ndarray, list[dict]]:
        """Detecta planos, estabiliza la profundidad y devuelve el resumen.

        `rays` es la rejilla HxWx3 de rayos con z=1 (de `PointCloud.rays_for`).
        `snap` en [0,1]: cuanto se pega la profundidad al plano detectado.
        """
        pts, weight = self._sample(depth, rays, far_limit)
        if pts.shape[0] < 400:
            self._age_out()
            return depth, self._summary()

        detected = self._extract_planes(pts)
        self._track(detected)
        self._classify()

        if snap > 0.01 and self._planes:
            depth = self._snap(depth, rays, snap)
        self.plane_mask = self._membership(depth, rays)
        _ = weight
        return depth, self._summary()

    # -- muestreo -----------------------------------------------------------
    @staticmethod
    def _sample(depth: np.ndarray, rays: np.ndarray, far_limit: float,
                target: int = 3600) -> tuple[np.ndarray, float]:
        stride = max(1, int(round((depth.size / target) ** 0.5)))
        z = depth[::stride, ::stride]
        r = rays[::stride, ::stride]
        pts = (r * z[..., None]).reshape(-1, 3).astype(np.float64)
        ok = (pts[:, 2] > 0.3) & (pts[:, 2] < far_limit)
        return pts[ok], float(stride)

    # -- RANSAC vectorizado -------------------------------------------------
    def _extract_planes(self, pts: np.ndarray) -> list[dict]:
        planes: list[dict] = []
        remaining = pts
        total = pts.shape[0]

        for _ in range(self.max_planes):
            n_pts = remaining.shape[0]
            if n_pts < max(300, int(0.06 * total)):
                break

            # Hipotesis en lote: ternas aleatorias -> normales por producto
            # vectorial. Todo el RANSAC son dos productos de matrices.
            idx = self._rng.integers(0, n_pts, size=(self.hypotheses, 3))
            p0, p1, p2 = (remaining[idx[:, k]] for k in range(3))
            normals = np.cross(p1 - p0, p2 - p0)
            lengths = np.linalg.norm(normals, axis=1)
            good = lengths > 1e-6
            if not good.any():
                break
            normals = normals[good] / lengths[good, None]
            offsets = np.einsum("ij,ij->i", normals, p0[good])

            # Tolerancia proporcional a la distancia: el ruido de la red crece
            # con la profundidad.
            tol = 0.020 + 0.020 * remaining[:, 2]
            dist = np.abs(remaining @ normals.T - offsets[None, :])
            inliers = dist < tol[:, None]
            counts = inliers.sum(axis=0)
            best = int(np.argmax(counts))
            if counts[best] < max(200, int(0.05 * total)):
                break

            support = remaining[inliers[:, best]]
            plane = self._fit(support)
            if plane is None:
                break
            plane["area"] = float(counts[best]) / float(total)
            planes.append(plane)
            remaining = remaining[~inliers[:, best]]
        return planes

    @staticmethod
    def _fit(support: np.ndarray) -> dict | None:
        """Ajuste fino por minimos cuadrados (SVD) sobre los inliers."""
        center = support.mean(axis=0)
        centered = support - center
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        normal = vt[2]
        # La normal mira hacia la camara (el origen): asi el signo es estable y
        # la clasificacion por orientacion no depende del azar del SVD.
        if float(normal @ center) > 0.0:
            normal = -normal
        return {"n": normal, "d": float(normal @ center), "mean": center}

    # -- seguimiento temporal -----------------------------------------------
    def _track(self, detected: list[dict], alpha: float = 0.15) -> None:
        """Casa los planos nuevos con los conocidos y suaviza sus parametros.

        Este EMA es la mitad de la estabilidad: el plano del suelo pasa a ser
        una media movil de todas sus detecciones, y la malla pegada a el se
        queda clavada aunque cada frame individual venga con ruido.
        """
        for old in self._planes:
            old["missing"] = old.get("missing", 0) + 1

        for new in detected:
            match = None
            for old in self._planes:
                angle_ok = float(new["n"] @ old["n"]) > 0.94  # < ~20 grados
                offset_ok = abs(new["d"] - old["d"]) < 0.35
                if angle_ok and offset_ok:
                    match = old
                    break
            if match is None:
                new["missing"] = 0
                self._planes.append(new)
            else:
                blended = (1 - alpha) * match["n"] + alpha * new["n"]
                match["n"] = blended / max(np.linalg.norm(blended), 1e-9)
                match["d"] = (1 - alpha) * match["d"] + alpha * new["d"]
                match["mean"] = (1 - alpha) * match["mean"] + alpha * new["mean"]
                match["area"] = new["area"]
                match["missing"] = 0

        self._age_out()

    def _age_out(self, patience: int = 6) -> None:
        # Un plano que lleva unos frames sin verse se olvida; mantenerlo un poco
        # evita que parpadee al rozar el borde del encuadre.
        self._planes = [p for p in self._planes if p.get("missing", 0) <= patience]

    # -- clasificacion ------------------------------------------------------
    def _classify(self) -> None:
        horizontals = []
        for plane in self._planes:
            tilt = float(plane["n"] @ UP)
            if tilt > 0.75:
                horizontals.append(plane)
            elif tilt < -0.75:
                plane["kind"] = "techo"
            elif abs(tilt) < 0.35:
                plane["kind"] = "pared"
            else:
                plane["kind"] = "superficie"

        if horizontals:
            # El suelo es la superficie horizontal MAS BAJA (y crece hacia
            # abajo en el convenio de camara). Lo demas horizontal claramente
            # por encima es una mesa u otra superficie de apoyo.
            floor = max(horizontals, key=lambda p: float(p["mean"][1]))
            floor_y = float(floor["mean"][1])
            for plane in horizontals:
                if plane is floor:
                    plane["kind"] = "suelo"
                elif floor_y - float(plane["mean"][1]) > 0.25:
                    plane["kind"] = "mesa"
                else:
                    plane["kind"] = "suelo"  # el mismo suelo partido en dos

    # -- estabilizacion -----------------------------------------------------
    def _snap(self, depth: np.ndarray, rays: np.ndarray, snap: float) -> np.ndarray:
        """Proyecta sobre cada plano los pixeles que le pertenecen.

        Para el pixel de rayo r, la interseccion con el plano n.p = d esta en
        z = d / (n.r). Donde la profundidad medida cae cerca de esa, se mezcla
        hacia el plano: la superficie queda matematicamente plana y quieta.
        """
        out = depth
        for plane in self._planes:
            if plane.get("missing", 0) > 0:
                continue  # solo planos confirmados en este frame
            ndotr = np.tensordot(rays, plane["n"].astype(np.float32), axes=([2], [0]))
            safe = np.abs(ndotr) > 0.08
            z_plane = np.where(safe, plane["d"] / np.where(safe, ndotr, 1.0), 0.0)
            tol = 0.030 + 0.025 * out
            # Mezcla CONTINUA, no umbral duro. Con umbral, un pixel que esta al
            # borde de la banda entra y sale del plano entre frames y ese
            # parpadeo METIA temblor en vez de quitarlo (medido: 11,1 -> 11,4 cm
            # por frame). Con el peso gaussiano la transicion es suave y el
            # borde no puede oscilar.
            residual = (out - z_plane) / np.maximum(tol, 1e-6)
            weight = snap * np.exp(-residual * residual)
            weight = np.where(safe & (z_plane > 0.25), weight, 0.0)
            out = (1.0 - weight) * out + weight * z_plane
        return out.astype(np.float32)

    def _membership(self, depth: np.ndarray, rays: np.ndarray) -> np.ndarray | None:
        """Que pixeles estan SOBRE un plano confirmado.

        Es la salida que permite al mallador vaciar de vertices las superficies
        planas: alli la geometria ya la cuenta el plano entero, no hace falta
        triangulacion fina. Solo cuentan los planos vistos en este frame.
        """
        mask: np.ndarray | None = None
        for plane in self._planes:
            if plane.get("missing", 0) > 0:
                continue
            ndotr = np.tensordot(rays, plane["n"].astype(np.float32), axes=([2], [0]))
            safe = np.abs(ndotr) > 0.08
            z_plane = np.where(safe, plane["d"] / np.where(safe, ndotr, 1.0), 0.0)
            tol = 0.035 + 0.028 * depth
            on = safe & (z_plane > 0.25) & (np.abs(depth - z_plane) < tol)
            mask = on if mask is None else (mask | on)
        return mask.astype(np.uint8) if mask is not None else None

    # -- salida -------------------------------------------------------------
    def _summary(self) -> list[dict]:
        active = [p for p in self._planes if p.get("missing", 0) == 0 and "kind" in p]
        active.sort(key=lambda p: (KIND_ORDER.index(p["kind"])
                                   if p["kind"] in KIND_ORDER else 9, -p["area"]))
        return [{"tipo": p["kind"], "area": round(100.0 * p["area"], 1)} for p in active]
