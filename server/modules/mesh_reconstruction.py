"""MeshReconstruction - generacion de la malla triangular.

Fase 1 (esto): `LiveDepthMesher` triangula directamente la rejilla del mapa de
profundidad del frame actual. Es lo que da la malla que se pega al suelo, las
paredes, la mesa o el coche que tienes delante, con latencia minima.

La clave para que no parezca un trapo colgado es *cortar* los triangulos que
cruzan un salto de profundidad: sin ese corte, el borde de una silla y la pared
del fondo quedarian unidos por una cortina de poligonos falsos.

Fase 2 (prevista): `SpatialMemory` acumulara TSDF y esta misma clase mallara por
marching cubes los chunks sucios. La firma de salida (`MeshChunkPayload`) ya es
la definitiva para que el visor no cambie.
"""

from __future__ import annotations

import numpy as np

from ..core.camera import Intrinsics
from ..core.types import MeshChunkPayload
from .adaptive_mesh import AdaptiveMesher, DelaunayMesher
from .point_cloud import PointCloud


class LiveDepthMesher:
    """Malla de rejilla a partir de un unico mapa de profundidad."""

    def __init__(self, point_cloud: PointCloud | None = None):
        self.pc = point_cloud or PointCloud()
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        # "quadtree" | "delaunay" | "uniforme"
        self.mesh_style = "delaunay"
        self.quadtree_mesher = AdaptiveMesher()
        self.delaunay_mesher = DelaunayMesher()
        self.adaptive_max_level = 4
        # Umbral de error que dispara la insercion de un vertice. Es EL control
        # de cuanto detalle se detecta: con 0.02 la malla se daba por satisfecha
        # antes de tiempo y objetos pequenos no aparecian.
        self.adaptive_tolerance = 0.010
        # Vertices por columna de la rejilla. 5.5 sale de medir el
        # compromiso entre triangulos, coste y estabilidad temporal.
        self.points_per_column = 16.0

    def _cell_corners(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Indices de los 4 vertices de cada celda y de sus 2 triangulos."""
        cached = self._grid_cache.get((h, w))
        if cached is not None:
            return cached
        idx = np.arange(h * w, dtype=np.int32).reshape(h, w)
        a = idx[:-1, :-1].ravel()  # arriba-izquierda
        b = idx[:-1, 1:].ravel()  # arriba-derecha
        c = idx[1:, 1:].ravel()  # abajo-derecha
        d = idx[1:, :-1].ravel()  # abajo-izquierda
        corners = np.stack([a, b, c, d], axis=1)  # Nx4
        tris = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], axis=1)  # Nx2x3
        self._grid_cache[(h, w)] = (corners, tris)
        return corners, tris

    def build(self, depth: np.ndarray, confidence: np.ndarray, intr: Intrinsics,
              near: float, far: float, edge_tolerance: float = 0.055,
              min_conf: float = 0.08, T_wc: np.ndarray | None = None,
              detail_cm: float | None = None,
              plane_mask: np.ndarray | None = None) -> MeshChunkPayload:
        h, w = depth.shape
        pts = self.pc.unproject(depth, intr).reshape(-1, 3)
        valid = self.pc.valid_mask(depth, confidence, near, far, min_conf).ravel()
        z = depth.ravel()

        corners, tris = self._cell_corners(h, w)

        # Una celda solo genera geometria si sus 4 esquinas son validas y entre
        # ellas no hay un salto de profundidad. El criterio principal no es la
        # diferencia de profundidad sino la TORSION de la celda,
        # |z00 + z11 - z01 - z10|, que vale cero sobre cualquier plano aunque
        # este muy inclinado. Con la simple diferencia, el suelo visto en
        # escorzo se troceaba igual que un borde real; con la torsion, medida
        # sobre escenas reales, la mediana es 0.0005 y los bordes autenticos
        # pasan de 0.6: separacion de tres ordenes de magnitud.
        cz = z[corners]
        cell_ok = valid[corners].all(axis=1)
        z_min = cz.min(axis=1)

        twist = np.abs(cz[:, 0] + cz[:, 2] - cz[:, 1] - cz[:, 3])
        cell_ok &= twist < (edge_tolerance * z_min + 0.005)

        # Tope duro adicional: aunque una celda sea plana, un salto enorme entre
        # sus esquinas es siempre un objeto delante de un fondo, nunca una
        # superficie. Es lo que evita las cortinas de poligonos.
        spread = cz.max(axis=1) - z_min
        cell_ok &= spread < (edge_tolerance * 8.0 * z_min + 0.03)

        # `cell_ok` ya lleva los cortes de borde, asi que sirve directamente
        # como mascara de celdas utilizables para cualquiera de los malladores.
        cell_mask = cell_ok.reshape(h - 1, w - 1)
        if self.mesh_style == "delaunay":
            # El tope de vertices se ata a la densidad para que el deslizador
            # de la interfaz siga controlando el detalle de la malla.
            faces = self.delaunay_mesher.build(
                depth, cell_mask, max_level=self.adaptive_max_level,
                tolerance=(detail_cm / 300.0) if detail_cm else self.adaptive_tolerance,
                spread_tolerance=edge_tolerance * 8.0,
                max_points=int(w * self.points_per_column),
                plane_mask=plane_mask,
            )
        elif self.mesh_style == "quadtree":
            faces = self.quadtree_mesher.build(
                depth, cell_mask, max_level=self.adaptive_max_level,
                tolerance=self.adaptive_tolerance,
            )
        else:
            faces = tris[cell_ok].reshape(-1, 3)

        if faces.size == 0:
            empty_v = np.zeros((0, 3), np.float32)
            return MeshChunkPayload(key=(0, 0, 0), vertices=empty_v,
                                    indices=np.zeros((0, 3), np.uint32))

        # Compactar: solo viajan al visor los vertices realmente usados.
        used = np.zeros(h * w, dtype=bool)
        used[faces.ravel()] = True
        remap = np.full(h * w, -1, dtype=np.int32)
        n_used = int(used.sum())
        remap[used] = np.arange(n_used, dtype=np.int32)

        vertices = pts[used].astype(np.float32)
        indices = remap[faces].astype(np.uint32)
        flat = np.nonzero(used)[0].astype(np.int32)
        pixels = np.stack([flat % w, flat // w], axis=1)  # (col, fila)

        if T_wc is not None:
            R, t = T_wc[:3, :3].astype(np.float32), T_wc[:3, 3].astype(np.float32)
            vertices = vertices @ R.T + t

        return MeshChunkPayload(key=(0, 0, 0), vertices=vertices, indices=indices,
                                pixels=pixels)


class MeshReconstruction:
    """Fachada del modulo: hoy delega en el mallador vivo.

    En fase 2 recibira ademas la `SpatialMemory` y emitira chunks del mapa
    acumulado; el resto del sistema no tendra que cambiar.
    """

    def __init__(self):
        self.live = LiveDepthMesher()

    def build_live(self, *args, **kwargs) -> MeshChunkPayload:
        return self.live.build(*args, **kwargs)
