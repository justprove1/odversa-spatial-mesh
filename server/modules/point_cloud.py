"""PointCloud - retroproyeccion del mapa de profundidad a 3D.

Convierte profundidad + intrinsecos en puntos del sistema de camara y, si se
pide, al sistema de mundo usando la pose. Tambien filtra lo que no es fiable:
fuera de rango, baja confianza y bordes de profundidad.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.camera import Intrinsics, pixel_rays
from ..core.types import PointCloudChunk


class PointCloud:
    def __init__(self):
        self._rays: np.ndarray | None = None
        self._rays_key: tuple | None = None

    def rays_for(self, intr: Intrinsics) -> np.ndarray:
        """Cachea la rejilla de rayos: solo cambia si cambian los intrinsecos."""
        key = (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
        if self._rays_key != key:
            self._rays = pixel_rays(intr)
            self._rays_key = key
        return self._rays  # type: ignore[return-value]

    def unproject(self, depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
        """HxW metros -> HxWx3 en coordenadas de camara (OpenCV)."""
        return self.rays_for(intr) * depth[..., None]

    @staticmethod
    def smooth_depth(depth: np.ndarray, strength: float) -> np.ndarray:
        """Suavizado bilateral: aplana superficies sin fundir los bordes.

        `strength` en [0,1] mapea el control "suavizado" de la interfaz.
        """
        s = float(np.clip(strength, 0.0, 1.0))
        if s <= 0.01:
            return depth
        sigma_color = 0.02 + 0.20 * s  # en metros
        sigma_space = 1.0 + 6.0 * s
        return cv2.bilateralFilter(depth, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    @staticmethod
    def guided_filter(depth: np.ndarray, guide: np.ndarray, radius: int = 4,
                      eps: float = 4e-3) -> np.ndarray:
        """Filtro guiado (He et al.) usando la imagen como referencia.

        El estimador de profundidad trabaja a menos resolucion que la camara y
        entrega bordes redondeados: el contorno de un mando o de un teclado sale
        como un bulto blando. La imagen en cambio SI tiene ese contorno nitido.

        El filtro guiado ajusta, en cada ventana, una relacion lineal
        `depth ~ a*guide + b` y reconstruye la profundidad con ella. El efecto
        practico es que los saltos de profundidad se enganchan a los bordes
        reales del objeto, y dentro de cada superficie se alisa el ruido.
        Es O(n) gracias a los filtros de caja, asi que cabe en el bucle.
        """
        g = guide.astype(np.float32)
        if g.shape != depth.shape:
            g = cv2.resize(g, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        g = g / 255.0 if g.max() > 1.5 else g

        p = depth.astype(np.float32)
        k = (radius * 2 + 1, radius * 2 + 1)
        box = lambda m: cv2.blur(m, k)  # noqa: E731 - media local, legible asi

        mean_g, mean_p = box(g), box(p)
        var_g = box(g * g) - mean_g * mean_g
        cov_gp = box(g * p) - mean_g * mean_p

        a = cov_gp / (var_g + eps)
        b = mean_p - a * mean_g
        return (box(a) * g + box(b)).astype(np.float32)

    @staticmethod
    def enhance_detail(depth: np.ndarray, amount: float, radius: int = 3) -> np.ndarray:
        """Realza el relieve fino que la red aplana.

        Los modelos monoculares aciertan con la estructura general y comprimen
        las diferencias pequenas: las teclas de un teclado o los botones de un
        mando quedan a milimetros de la superficie y desaparecen. Esto es una
        mascara de enfoque sobre la profundidad: amplifica lo que se separa de
        la media local sin tocar la forma global.

        Se limita la correccion a una fraccion de la distancia para que no
        invente relieves imposibles ni amplifique el ruido de las zonas lejanas.
        """
        a = float(np.clip(amount, 0.0, 1.0))
        if a <= 0.01:
            return depth
        low = cv2.GaussianBlur(depth, (0, 0), radius)
        detail = depth - low
        limit = 0.05 * np.maximum(depth, 0.2)  # como mucho un 5% de la distancia
        detail = np.clip(detail, -limit, limit)
        return (depth + (1.6 * a) * detail).astype(np.float32)

    def refine_depth(self, depth: np.ndarray, bgr: np.ndarray, cfg) -> np.ndarray:
        """Cadena de refinado de la profundidad, en el orden que importa.

        1. Filtro guiado -> engancha los bordes a los del objeto real.
        2. Bilateral suave -> alisa el ruido que quede dentro de las superficies.
        3. Realce -> devuelve el relieve fino que los dos pasos anteriores y la
           propia red aplanan.

        Vive aqui, y no en el pipeline, para que las herramientas de
        previsualizacion apliquen exactamente lo mismo que el sistema real: si
        divergen, las pruebas dejan de valer.
        """
        out = depth
        if getattr(cfg, "guided_filter", False):
            guide = cv2.cvtColor(
                cv2.resize(bgr, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
            out = self.guided_filter(out, guide, cfg.guided_radius)
        out = self.smooth_depth(out, cfg.spatial_smoothing)
        return self.enhance_detail(out, cfg.detail_boost)

    @staticmethod
    def valid_mask(depth: np.ndarray, confidence: np.ndarray, near: float, far: float,
                   min_conf: float = 0.08) -> np.ndarray:
        # El corte lejano va MUY por encima de `far_m`, no justo en el.
        #
        # `far_m` es el anclaje del percentil 10, asi que recortar ahi descartaba
        # por construccion el 10% mas lejano de cada escena -tipicamente la pared
        # del fondo- y abria un agujero permanente en la malla. Solo se descarta
        # lo que de verdad esta saturado contra el limite duro.
        limit = far * 2.5
        m = np.isfinite(depth) & (depth > near * 0.45) & (depth < limit * 0.98)
        return m & (confidence > min_conf)

    def build(self, depth: np.ndarray, confidence: np.ndarray, intr: Intrinsics,
              T_wc: np.ndarray | None, near: float, far: float,
              stride: int = 1, max_points: int = 60000) -> PointCloudChunk:
        """Nube de puntos filtrada, en mundo si se da `T_wc`."""
        if stride > 1:
            depth = depth[::stride, ::stride]
            confidence = confidence[::stride, ::stride]
            intr = intr.scaled(depth.shape[1], depth.shape[0])

        pts_cam = self.unproject(depth, intr)
        mask = self.valid_mask(depth, confidence, near, far)
        pts = pts_cam[mask]
        conf = confidence[mask]

        if pts.shape[0] > max_points:
            sel = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
            pts, conf = pts[sel], conf[sel]

        if T_wc is not None:
            R, t = T_wc[:3, :3].astype(np.float32), T_wc[:3, 3].astype(np.float32)
            pts = pts @ R.T + t

        return PointCloudChunk(points=pts.astype(np.float32), normals=None,
                               confidence=conf.astype(np.float32))
