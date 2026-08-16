"""SpatialMemory - mapa 3D persistente.

ESTADO: esqueleto. Guarda estadisticas y sirve el guardado del mapa, pero
todavia no acumula geometria entre frames. La pieza que faltaba, la pose de
`CameraTracking`, ya esta: `integrate` recibe un `T_wc` de verdad, no la
identidad. Lo que falta es la fusion en si.

FASE 2, diseno ya cerrado: TSDF con hash de voxeles.

  * El mundo se divide en chunks dispersos de `chunk_voxels`^3 voxeles, guardados
    en un diccionario indexado por su coordenada entera. Solo existen en memoria
    los chunks que la camara ha visto: un piso entero cabe de sobra.
  * Cada voxel guarda distancia con signo truncada y un peso. Integrar un frame
    es proyectar los centros de voxel candidatos sobre el mapa de profundidad y
    actualizar con media ponderada. El peso satura en `max_weight`, y de ahi sale
    la estabilidad temporal: una superficie vista veinte veces deja de temblar.
  * Los chunks candidatos NO se buscan recorriendo el frustum (carisimo), sino
    derivandolos de los propios puntos observados y dilatando por la banda de
    truncamiento. El coste pasa a ser proporcional a la superficie vista.
  * Cada chunk tocado se marca como sucio, y `MeshReconstruction` remalla por
    marching cubes solo esos, con un borde de un voxel para que las costuras
    entre chunks encajen.
  * `min_weight_to_mesh` descarta lo observado una sola vez, que es de donde
    salen los poligonos fantasma.
"""

from __future__ import annotations

import threading

import numpy as np

from ..config import RuntimeConfig
from ..core.types import PointCloudChunk


class SpatialMemory:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._points_seen = 0
        self._frames_integrated = 0
        # Ultima nube observada (fase 1: solo la del frame actual).
        self._last_cloud: PointCloudChunk | None = None

    # -- estado -------------------------------------------------------------
    @property
    def points_total(self) -> int:
        return self._points_seen

    @property
    def chunk_count(self) -> int:
        return 0  # fase 2

    def reset(self) -> None:
        with self._lock:
            self._points_seen = 0
            self._frames_integrated = 0
            self._last_cloud = None

    # -- integracion --------------------------------------------------------
    def integrate(self, cloud: PointCloudChunk, T_wc: np.ndarray) -> None:
        """Fase 1: solo contabiliza. Fase 2: fusion TSDF."""
        with self._lock:
            self._last_cloud = cloud
            self._points_seen = int(cloud.points.shape[0])
            self._frames_integrated += 1

    def snapshot_points(self) -> np.ndarray:
        with self._lock:
            if self._last_cloud is None:
                return np.zeros((0, 3), np.float32)
            return self._last_cloud.points.copy()
