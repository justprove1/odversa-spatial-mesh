"""Triangulacion adaptativa del mapa de profundidad.

La rejilla uniforme gasta exactamente los mismos triangulos en una pared lisa
que en una planta: es un desperdicio que se paga en ancho de banda, en GPU y en
fps. Aqui se hace lo contrario: geometria gruesa donde la superficie es plana y
fina solo donde hace falta.

Hay dos malladores, y el que se usa por defecto es el segundo:

* `AdaptiveMesher` - cuadtree restringido. Rapido de construir (~3 ms) pero sus
  triangulos van alineados a una cuadricula y se le nota el origen.
* `DelaunayMesher` - **el bueno**. Coloca vertices por refinamiento voraz
  -donde la malla actual mas se aleja de la superficie real- mas las esquinas de
  los contornos de la escena, y los une con Delaunay. Da triangulos irregulares
  que se alinean con la arquitectura, como una malla de escaneo de verdad.
  Medido: 1.200 triangulos frente a los 8.241 del cuadtree y los 46.206 de la
  rejilla uniforme, sobre la misma escena.

Delaunay es lo que hace que "cuadre matematicamente": produce por construccion
una triangulacion valida del conjunto de puntos, sin huecos, sin solapes y sin
vertices colgando en mitad de una arista.

El cuadtree, paso a paso
------------------------
1. **Error por bloque.** Un bloque se puede dejar grande si la superficie que
   contiene se parece a la interpolacion bilineal de sus cuatro esquinas. Se
   mide la desviacion maxima relativa a la distancia: a 5 m un error de 2 cm no
   se ve, a 0.5 m si.
2. **Asignacion de nivel**, de grueso a fino: se acepta el bloque mas grande que
   pasa la prueba y se marca su area como cubierta.
3. **Regla 2:1.** Dos bloques vecinos no pueden diferir en mas de un nivel. Se
   impone con una erosion sobre el mapa de niveles, que es justo "el nivel de
   cada celda no puede superar en mas de uno al de su vecino mas fino".
4. **Abanico desde el centro con puntos medios.** Cada bloque se triangula
   uniendo su centro con las esquinas, insertando el punto medio de una arista
   solo cuando el vecino de ese lado es mas fino. Esa insercion es lo que evita
   las grietas: sin ella, el vertice extra del vecino fino cae en mitad de la
   arista del bloque grande y abre un agujero.

Todos los vertices (esquinas, centros y puntos medios) son puntos de la rejilla
original, porque los bloques tienen lado par. No hay que crear geometria nueva:
solo se eligen indices distintos.
"""

from __future__ import annotations

import cv2
import numpy as np

# Lado minimo de bloque en celdas. Ha de ser par para que centro y puntos medios
# caigan sobre puntos existentes de la rejilla.
MIN_BLOCK = 2


class AdaptiveMesher:
    """Convierte un mapa de profundidad en una malla adaptativa."""

    def __init__(self):
        self._cache: dict[tuple, np.ndarray] = {}

    # -- pesos bilineales ---------------------------------------------------
    def _weights(self, side: int) -> tuple[np.ndarray, ...]:
        """Pesos de la interpolacion bilineal para un bloque de `side` celdas."""
        key = ("w", side)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        u = np.linspace(0.0, 1.0, side + 1, dtype=np.float32)
        wy = u[:, None]
        wx = u[None, :]
        weights = ((1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx)
        self._cache[key] = weights
        return weights

    # -- nivel por celda ----------------------------------------------------
    def _level_map(self, depth: np.ndarray, usable: np.ndarray, max_level: int,
                   tolerance: float) -> np.ndarray:
        """Nivel (exponente del lado del bloque) deseado por cada celda.

        Se calcula de fino a grueso: una celda quiere el bloque mas grande cuya
        superficie siga siendo aproximable por sus cuatro esquinas.
        """
        cells_y, cells_x = usable.shape
        min_exp = int(np.log2(MIN_BLOCK))
        level = np.full((cells_y, cells_x), min_exp, np.int8)

        for exp in range(min_exp + 1, max_level + 1):
            side = 1 << exp
            nby, nbx = cells_y // side, cells_x // side
            if nby < 1 or nbx < 1:
                break

            # Ventanas (side+1)x(side+1) con solape de un punto: son los bloques.
            blocks = np.lib.stride_tricks.sliding_window_view(
                depth, (side + 1, side + 1))[::side, ::side][:nby, :nbx]

            w00, w01, w10, w11 = self._weights(side)
            c00 = blocks[:, :, 0, 0][:, :, None, None]
            c01 = blocks[:, :, 0, -1][:, :, None, None]
            c10 = blocks[:, :, -1, 0][:, :, None, None]
            c11 = blocks[:, :, -1, -1][:, :, None, None]
            predicted = c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11

            error = np.abs(blocks - predicted) / np.maximum(blocks, 0.2)
            cell_block = usable[:nby * side, :nbx * side].reshape(nby, side, nbx, side)
            fits = (error.max(axis=(2, 3)) < tolerance) & cell_block.all(axis=(1, 3))

            if fits.any():
                big = np.repeat(np.repeat(fits, side, axis=0), side, axis=1)
                region = level[:nby * side, :nbx * side]
                region[big] = exp
        return level

    @staticmethod
    def _quantize(level: np.ndarray, max_level: int, min_exp: int) -> np.ndarray:
        """Obliga al mapa a ser un cuadtree de verdad.

        Un bloque solo puede estar en el nivel `exp` si TODAS sus celdas quieren
        ese nivel o mas; si no, se parte en cuatro y el asunto baja al nivel
        siguiente. Sin este paso el mapa tiene bloques a medio nivel que luego no
        se emiten en ninguna pasada, y eso es exactamente lo que abria agujeros
        cuadrados en la malla.
        """
        out = level.copy()
        cells_y, cells_x = out.shape
        for exp in range(max_level, min_exp, -1):
            side = 1 << exp
            nby, nbx = cells_y // side, cells_x // side
            if nby < 1 or nbx < 1:
                continue
            view = out[:nby * side, :nbx * side].reshape(nby, side, nbx, side)
            uniform = view.min(axis=(1, 3)) >= exp
            view[...] = np.where(uniform[:, None, :, None], np.int8(exp),
                                 np.minimum(view, np.int8(exp - 1)))
            # El resto que no cabe en bloques enteros no puede reclamar `exp`.
            out[nby * side:, :] = np.minimum(out[nby * side:, :], np.int8(exp - 1))
            out[:, nbx * side:] = np.minimum(out[:, nbx * side:], np.int8(exp - 1))
        return out

    @staticmethod
    def _balance(level: np.ndarray, iterations: int = 4) -> np.ndarray:
        """Impone la regla 2:1 entre bloques vecinos.

        `cv2.erode` con un kernel de cruz da el minimo local; el nivel de una
        celda no puede pasar de ese minimo mas uno. Repetido unas pocas veces,
        propaga la restriccion por todo el mapa.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        out = level.astype(np.uint8)
        for _ in range(iterations):
            neighbour_min = cv2.erode(out, kernel, borderType=cv2.BORDER_REPLICATE)
            limited = np.minimum(out, neighbour_min + 1)
            if np.array_equal(limited, out):
                break
            out = limited
        return out.astype(np.int8)

    # -- triangulacion ------------------------------------------------------
    def build(self, depth: np.ndarray, usable: np.ndarray, max_level: int = 4,
              tolerance: float = 0.02) -> np.ndarray:
        """Devuelve los triangulos (Tx3) como indices de la rejilla de puntos.

        `usable` es la mascara POR CELDA (h-1 x w-1) que ya trae aplicados los
        cortes de borde: una celda inutilizable nunca entra en ningun bloque.
        """
        h, w = depth.shape
        if h < 3 or w < 3:
            return np.zeros((0, 3), np.int32)

        min_exp = int(np.log2(MIN_BLOCK))
        level = self._level_map(depth, usable, max_level, tolerance)
        # Cuantizar y equilibrar se persiguen mutuamente -equilibrar rompe la
        # alineacion, cuantizar puede romper el equilibrio-, asi que se alternan
        # hasta que se quedan quietos. Ambos solo bajan niveles, o sea que
        # converge siempre.
        for _ in range(3):
            level = self._quantize(level, max_level, min_exp)
            balanced = self._balance(level)
            if np.array_equal(balanced, level):
                break
            level = balanced
        level = self._quantize(level, max_level, min_exp)

        cells_y, cells_x = level.shape
        idx = np.arange(h * w, dtype=np.int32).reshape(h, w)

        faces: list[np.ndarray] = []
        for exp in range(min_exp, max_level + 1):
            side = 1 << exp
            half = side // 2
            nby, nbx = cells_y // side, cells_x // side
            if nby < 1 or nbx < 1:
                continue

            # Un bloque se emite si TODAS sus celdas estan en este nivel y son
            # utilizables. Lo segundo es lo que deja los huecos donde el dato es
            # malo de verdad, en vez de rellenarlos con geometria inventada.
            block_level = level[:nby * side, :nbx * side].reshape(nby, side, nbx, side)
            block_use = usable[:nby * side, :nbx * side].reshape(nby, side, nbx, side)
            uniform = (block_level == exp).all(axis=(1, 3)) & block_use.all(axis=(1, 3))
            if not uniform.any():
                continue

            by, bx = np.nonzero(uniform)
            y0, x0 = by * side, bx * side
            y1, x1 = y0 + side, x0 + side
            ym, xm = y0 + half, x0 + half

            # Esquinas, centro y puntos medios: todos son puntos de la rejilla.
            v_center = idx[ym, xm]
            corner = {
                "tl": idx[y0, x0], "tr": idx[y0, x1],
                "br": idx[y1, x1], "bl": idx[y1, x0],
            }
            mid = {
                "top": idx[y0, xm], "right": idx[ym, x1],
                "bottom": idx[y1, xm], "left": idx[ym, x0],
            }

            # Vecino mas fino? Se mira el nivel de la celda de fuera del borde.
            finer = {
                "top": self._neighbour_finer(level, y0 - 1, x0 + half, exp),
                "bottom": self._neighbour_finer(level, y1, x0 + half, exp),
                "left": self._neighbour_finer(level, y0 + half, x0 - 1, exp),
                "right": self._neighbour_finer(level, y0 + half, x1, exp),
            }

            edges = (("top", "tl", "tr"), ("right", "tr", "br"),
                     ("bottom", "br", "bl"), ("left", "bl", "tl"))
            for name, a, b in edges:
                va, vb, vm = corner[a], corner[b], mid[name]
                split = finer[name]
                # Vecino igual o mas grueso: un solo triangulo por arista.
                whole = ~split
                if whole.any():
                    faces.append(np.stack([v_center[whole], va[whole], vb[whole]], axis=1))
                # Vecino mas fino: se parte la arista por su punto medio.
                if split.any():
                    faces.append(np.stack([v_center[split], va[split], vm[split]], axis=1))
                    faces.append(np.stack([v_center[split], vm[split], vb[split]], axis=1))

        if not faces:
            return np.zeros((0, 3), np.int32)
        return np.concatenate(faces, axis=0).astype(np.int32)

    @staticmethod
    def _neighbour_finer(level: np.ndarray, y: np.ndarray, x: np.ndarray,
                         exp: int) -> np.ndarray:
        """True donde la celda vecina pertenece a un bloque mas pequeno."""
        cells_y, cells_x = level.shape
        inside = (y >= 0) & (y < cells_y) & (x >= 0) & (x < cells_x)
        yy = np.clip(y, 0, cells_y - 1)
        xx = np.clip(x, 0, cells_x - 1)
        return inside & (level[yy, xx] < exp)


class DelaunayMesher(AdaptiveMesher):
    """Triangulacion irregular de Delaunay sobre puntos repartidos por detalle.

    El cuadtree produce triangulos alineados a una cuadricula: se le nota el
    origen. Aqui la idea es otra y da el aspecto de una malla de escaneo de
    verdad: se eligen PUNTOS con densidad variable -pocos en una pared lisa,
    muchos alrededor de una planta o un canto- y se triangulan con Delaunay.

    Delaunay es lo que hace que "cuadre matematicamente": produce por
    construccion una triangulacion valida del conjunto de puntos, sin huecos ni
    solapes y sin vertices colgando en mitad de una arista. Ya no hacen falta la
    regla 2:1 ni los puntos medios: el problema de las grietas desaparece porque
    no existe la nocion de bloque vecino.

    La densidad de puntos se hereda del mismo mapa de error del cuadtree, que ya
    sabe donde la superficie es plana y donde no.
    """

    def __init__(self):
        super().__init__()
        #: uno de cada cuantos pixeles se evalua como candidato
        # Uno de cada 3: con 2 el mallado se iba a 36 ms para el mismo detalle.
        # El campo de error es suave, asi que mirar menos pixeles no cambia
        # donde acaban los vertices, solo cuanto cuesta encontrarlos.
        self.candidate_stride = 3
        #: puntos del frame anterior (ys, xs, forma de la rejilla)
        self._previous: tuple[np.ndarray, np.ndarray, tuple[int, int]] | None = None

    def reset(self) -> None:
        """Olvida los puntos acumulados (cambio de resolucion o reinicio)."""
        self._previous = None

    #: separacion (en celdas) de la reticula de apoyo dentro de un plano
    PLANE_LATTICE = 22

    def build(self, depth: np.ndarray, usable: np.ndarray, max_level: int = 4,
              tolerance: float = 0.02, spread_tolerance: float = 0.45,
              rounds: int = 5, max_points: int = 2400,
              plane_mask: np.ndarray | None = None) -> np.ndarray:
        h, w = depth.shape
        if h < 5 or w < 5:
            return np.zeros((0, 3), np.int32)

        # Interior de los planos detectados: alli apenas hacen falta vertices,
        # la superficie ya es plana de verdad (la profundidad esta pegada al
        # plano). El borde del plano en cambio se conserva integro: es donde la
        # malla tiene que ser nitida.
        interior = None
        if plane_mask is not None and plane_mask.shape == depth.shape:
            interior = cv2.erode(plane_mask, np.ones((3, 3), np.uint8),
                                 iterations=2).astype(bool)

        ys, xs = self._greedy_points(depth, usable, tolerance, rounds, max_points,
                                     interior)
        if ys.size < 4:
            return np.zeros((0, 3), np.int32)

        from scipy.spatial import Delaunay, QhullError

        # Delaunay trabaja en el plano de la imagen; la profundidad se usa
        # despues para decidir que triangulos sobreviven.
        try:
            tri = Delaunay(np.stack([xs, ys], axis=1).astype(np.float64))
        except (QhullError, ValueError):
            return np.zeros((0, 3), np.int32)

        simplices = tri.simplices
        if simplices.size == 0:
            return np.zeros((0, 3), np.int32)

        ty = ys[simplices]
        tx = xs[simplices]
        keep = self._filter(depth, usable, ty, tx, spread_tolerance)
        if not keep.any():
            return np.zeros((0, 3), np.int32)

        ty, tx = ty[keep], tx[keep]
        return (ty * w + tx).astype(np.int32)

    # -- puntos por refinamiento voraz ---------------------------------------
    @staticmethod
    def _feature_points(usable: np.ndarray, epsilon: float = 1.6
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Vertices de los contornos de la mascara, simplificados.

        Son las esquinas de la escena: el canto de una jardinera, el marco de
        una puerta, la silueta de un objeto contra el fondo. Poniendo vertices
        justo ahi, las aristas de los triangulos se alinean con la arquitectura
        en vez de con una cuadricula, que es lo que distingue una malla de
        escaneo de una rejilla subdividida.
        """
        mask = usable.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ys: list[np.ndarray] = []
        xs: list[np.ndarray] = []
        for contour in contours:
            if len(contour) < 4:
                continue
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            xs.append(approx[:, 0])
            ys.append(approx[:, 1])
        if not ys:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        return (np.concatenate(ys).astype(np.int32),
                np.concatenate(xs).astype(np.int32))

    def _greedy_points(self, depth: np.ndarray, usable: np.ndarray, tolerance: float,
                       rounds: int, max_points: int,
                       interior: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Inserta vertices donde la malla actual se aleja mas de la superficie.

        Es el criterio correcto: no "cada tantos pixeles" sino "donde el plano
        que ya hay miente mas". Un suelo liso se queda con cuatro triangulos por
        grandes que sean, y una planta recibe todos los que necesite.

        Se hace por rondas en lote -en cada una se mete el peor pixel de cada
        triangulo que pase de tolerancia- en vez de uno a uno, que exigiria
        rehacer la triangulacion miles de veces.
        """
        from scipy.spatial import Delaunay, QhullError

        h, w = depth.shape
        cells_y, cells_x = usable.shape

        # Puntos del frame anterior: son la mayor parte de la semilla.
        #
        # Es lo que quita el hervor. Recalculando los vertices desde cero en cada
        # frame, dos imagenes casi identicas producen conjuntos de puntos
        # distintos y la malla salta sin parar, lo que hace ilegible la forma.
        # Conservando los puntos que siguen siendo validos, la geometria se queda
        # quieta y solo cambia donde la escena cambia de verdad. Y de paso sale
        # mas barato: partiendo de un conjunto ya bueno bastan una o dos rondas.
        L = self.PLANE_LATTICE

        def on_lattice(py, px):
            return (py % L == 0) & (px % L == 0)

        def keep_outside_planes(py, px):
            """Fuera del interior de un plano, o sobre la reticula de apoyo."""
            if interior is None:
                return np.ones(py.shape, bool)
            inside = interior[np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)]
            return ~inside | on_lattice(py, px)

        reused = np.zeros(0, np.int32), np.zeros(0, np.int32)
        if self._previous is not None and self._previous[2] == (h, w):
            prev_y, prev_x = self._previous[0], self._previous[1]
            alive = usable[np.clip(prev_y, 0, cells_y - 1),
                           np.clip(prev_x, 0, cells_x - 1)]
            # La poda es determinista (mascara de planos + reticula fija), asi
            # que no reintroduce parpadeo: dos frames parecidos podan lo mismo.
            alive &= keep_outside_planes(prev_y, prev_x)
            reused = prev_y[alive], prev_x[alive]
            # Con historial basta UNA ronda: los puntos ya estan casi donde
            # toca y solo hay que corregir lo que cambio. Medido, pasar de
            # dos rondas a una baja el mallado de 7,6 a 6,4 ms sin tocar la
            # estabilidad (90%).
            rounds = min(rounds, 1)

        # Semilla deliberadamente escasa: las esquinas del encuadre y las
        # esquinas de la escena, nada mas.
        #
        # Se probo sembrar tambien una reticula gruesa para llegar antes a la
        # densidad final, y el resultado fue peor: los puntos de reticula que
        # nadie necesita NO se eliminan luego, asi que el suelo volvia a salir
        # cuadriculado. Partiendo de poco, cada punto que aparece esta ahi
        # porque el error lo pidio, y una superficie plana se queda con cuatro
        # triangulos enormes -que es justo el aspecto que se busca.
        fy, fx = self._feature_points(usable)
        if interior is not None:
            # El contorno del plano (la junta suelo-pared, el canto de una mesa)
            # entra como vertices propios: las aristas de la malla se alinean
            # con la arquitectura en vez de cruzarla, que es lo que da el
            # aspecto nitido de una malla de escaneo.
            by, bx = self._feature_points(interior.astype(np.uint8), epsilon=2.2)
            fy = np.concatenate([fy, by])
            fx = np.concatenate([fx, bx])
        ys = np.concatenate([[0, 0, h - 1, h - 1], fy, reused[0]])
        xs = np.concatenate([[0, w - 1, 0, w - 1], fx, reused[1]])

        # Pixeles candidatos: los que caen sobre celdas utilizables.
        #
        # Se submuestrean con `stride`. Localizar el triangulo que contiene cada
        # candidato es la parte cara del metodo, y mirar uno de cada dos pixeles
        # no cambia donde acaban los vertices -el error es una magnitud suave-
        # pero reduce el coste a la mitad. Sobre una escena muy ruidosa, sin esto
        # el mallado se ponia en 57 ms y se comia el margen del hilo.
        stride = self.candidate_stride if usable.size > 20000 else 1
        sub = usable[::stride, ::stride]
        py, px = np.nonzero(sub)
        py = (py * stride).astype(np.int32)
        px = (px * stride).astype(np.int32)
        keep_cand = keep_outside_planes(py, px)
        py, px = py[keep_cand], px[keep_cand]
        if py.size == 0:
            return ys.astype(np.int32), xs.astype(np.int32)
        cand_xy = np.stack([px, py], axis=1).astype(np.float64)
        cand_z = depth[py, px]

        for _ in range(rounds):
            unique = np.unique(np.stack([ys, xs], axis=1), axis=0)
            if unique.shape[0] < 3 or unique.shape[0] >= max_points:
                break
            ys, xs = unique[:, 0], unique[:, 1]
            try:
                tri = Delaunay(np.stack([xs, ys], axis=1).astype(np.float64))
            except (QhullError, ValueError):
                break

            simplex = tri.find_simplex(cand_xy)
            inside = simplex >= 0
            if not inside.any():
                break

            # Profundidad que predice la malla, por coordenadas baricentricas.
            s = simplex[inside]
            transform = tri.transform[s]
            delta = cand_xy[inside] - transform[:, 2]
            bary = np.einsum("ijk,ik->ij", transform[:, :2], delta)
            weights = np.column_stack([bary, 1.0 - bary.sum(axis=1)])
            corner_z = depth[ys[tri.simplices[s]], xs[tri.simplices[s]]]
            predicted = (weights * corner_z).sum(axis=1)

            actual = cand_z[inside]
            error = np.abs(predicted - actual) / np.maximum(actual, 0.2)
            bad = error > tolerance
            if not bad.any():
                break

            # El pixel de peor error de cada triangulo: se ordena por triangulo
            # y, dentro de cada uno, por error descendente; el primero de cada
            # grupo es el que entra.
            s_bad = s[bad]
            e_bad = error[bad]
            idx_bad = np.nonzero(inside)[0][bad]
            order = np.lexsort((-e_bad, s_bad))
            s_sorted = s_bad[order]
            first = np.ones(s_sorted.size, bool)
            first[1:] = s_sorted[1:] != s_sorted[:-1]
            chosen = idx_bad[order][first]

            room = max_points - ys.size
            if room <= 0:
                break
            if chosen.size > room:
                chosen = chosen[:room]
            ys = np.concatenate([ys, py[chosen]])
            xs = np.concatenate([xs, px[chosen]])

        unique = np.unique(np.stack([ys, xs], axis=1), axis=0)
        out_y = unique[:, 0].astype(np.int32)
        out_x = unique[:, 1].astype(np.int32)

        # Poda cuando se acumulan demasiados: se van los de menor curvatura, que
        # son los que menos aportan a la forma. El criterio es determinista, asi
        # que dos frames parecidos podan los mismos y no reaparece el parpadeo.
        if out_y.size > max_points:
            curvature = np.abs(cv2.Laplacian(depth, cv2.CV_32F, ksize=3))
            score = curvature[out_y, out_x]
            keep = np.argpartition(-score, max_points - 1)[:max_points]
            keep.sort()
            out_y, out_x = out_y[keep], out_x[keep]

        self._previous = (out_y, out_x, (h, w))
        return out_y, out_x

    @staticmethod
    def _sample_points(level: np.ndarray, usable: np.ndarray, max_level: int
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Puntos de muestreo con separacion variable segun el nivel de detalle.

        Cada punto de la rejilla hereda el nivel MAS FINO de las celdas que toca,
        de modo que junto a una zona con detalle la densidad sube antes de llegar
        a ella y la transicion no queda abrupta.
        """
        cells_y, cells_x = level.shape
        h, w = cells_y + 1, cells_x + 1

        point_level = np.full((h, w), np.int8(max_level))
        for dy in (0, 1):
            for dx in (0, 1):
                view = point_level[dy:dy + cells_y, dx:dx + cells_x]
                np.minimum(view, level, out=view)

        point_usable = np.zeros((h, w), bool)
        for dy in (0, 1):
            for dx in (0, 1):
                point_usable[dy:dy + cells_y, dx:dx + cells_x] |= usable

        yy, xx = np.mgrid[0:h, 0:w]
        step = (1 << point_level.astype(np.int32))
        take = (yy % step == 0) & (xx % step == 0) & point_usable

        # El borde del encuadre siempre entra: sin el, la envolvente convexa se
        # queda corta y la malla no llega a los lados de la imagen.
        border = np.zeros((h, w), bool)
        edge_step = 1 << max(max_level - 1, 1)
        border[0, ::edge_step] = border[-1, ::edge_step] = True
        border[::edge_step, 0] = border[::edge_step, -1] = True
        border[[0, 0, -1, -1], [0, -1, 0, -1]] = True
        take |= border & point_usable

        ys, xs = np.nonzero(take)
        return ys.astype(np.int32), xs.astype(np.int32)

    @staticmethod
    def _filter(depth: np.ndarray, usable: np.ndarray, ty: np.ndarray,
                tx: np.ndarray, spread_tolerance: float) -> np.ndarray:
        """Descarta triangulos que cruzan un salto o cubren zona sin datos.

        Delaunay rellena TODA la envolvente convexa, incluidos los huecos que la
        mascara habia dejado a proposito. Por eso no basta con mirar los
        vertices: se comprueba tambien el interior del triangulo -su centro y los
        puntos medios de sus lados- contra la mascara de celdas utilizables.
        """
        cells_y, cells_x = usable.shape
        z = depth[ty, tx]
        z_min = z.min(axis=1)
        keep = (z.max(axis=1) - z_min) < (spread_tolerance * z_min + 0.03)

        def cell_ok(py: np.ndarray, px: np.ndarray) -> np.ndarray:
            iy = np.clip(py.astype(np.int32), 0, cells_y - 1)
            ix = np.clip(px.astype(np.int32), 0, cells_x - 1)
            return usable[iy, ix]

        # Cuantos de los puntos de muestra del interior caen en zona utilizable.
        #
        # Antes se exigia que lo fueran TODOS, y con triangulos grandes eso es
        # demasiado severo: uno que cubre medio suelo se caia entero por rozar un
        # hueco de dos pixeles, y de ahi venian los claros en la malla. Ahora se
        # pide mayoria, que descarta igual los triangulos que tapan un agujero de
        # verdad sin castigar a los que solo lo rozan.
        votes = cell_ok(ty.mean(axis=1), tx.mean(axis=1)).astype(np.int8) * 2
        for a, b in ((0, 1), (1, 2), (2, 0)):
            votes += cell_ok((ty[:, a] + ty[:, b]) * 0.5, (tx[:, a] + tx[:, b]) * 0.5)
        return keep & (votes >= 3)  # el centro vale doble
