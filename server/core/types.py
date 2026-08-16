"""Tipos compartidos por todo el pipeline.

Convenciones (fijas en todo Odversa):
  * Sistema de camara: OpenCV -> x derecha, y abajo, z hacia delante.
  * Pose `T_wc`: matriz 4x4 homogenea "world from camera", es decir
    `p_world = T_wc @ p_cam`. El primer frame define el origen del mundo.
  * Profundidad: metros, float32, 0 o NaN = invalido.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TrackingState(str, Enum):
    INIT = "INIT"
    GOOD = "GOOD"
    WEAK = "WEAK"
    LOST = "LOST"


@dataclass
class ImuSample:
    """Muestra inercial del movil (ejes del dispositivo, WebAPI DeviceMotion)."""

    t: float
    # rad/s
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    # m/s^2 sin gravedad
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    # Orientacion absoluta del dispositivo (alpha, beta, gamma) en radianes.
    orientation: np.ndarray | None = None


@dataclass
class Frame:
    """Un frame recibido del movil, ya decodificado."""

    index: int
    # Instante de captura en el movil (segundos, reloj del movil).
    t_device: float
    # Instante de recepcion en el servidor.
    t_recv: float
    bgr: np.ndarray  # HxWx3 uint8
    jpeg: bytes | None = None
    imu: ImuSample | None = None
    # Latencia extremo a extremo estimada en ms (si hay reloj sincronizado).
    latency_ms: float = 0.0

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.bgr.shape[:2]
        return w, h


@dataclass
class DepthResult:
    """Salida del estimador de profundidad."""

    depth: np.ndarray  # HxW float32, metros
    confidence: np.ndarray  # HxW float32 en [0,1]
    raw_disparity: np.ndarray | None = None
    infer_ms: float = 0.0


@dataclass
class TrackingResult:
    T_wc: np.ndarray  # 4x4 float64
    state: TrackingState
    inliers: int = 0
    tracked: int = 0
    # Traslacion y rotacion respecto al frame anterior.
    delta_t_m: float = 0.0
    delta_r_deg: float = 0.0
    used_imu: bool = False


@dataclass
class PointCloudChunk:
    """Nube de puntos de un frame, ya en coordenadas de mundo."""

    points: np.ndarray  # Nx3 float32
    normals: np.ndarray | None  # Nx3 float32
    confidence: np.ndarray  # N float32


@dataclass
class MeshChunkPayload:
    """Malla de un chunk del mapa, lista para enviar al visor."""

    key: tuple[int, int, int]
    vertices: np.ndarray  # Vx3 float32 (mundo)
    indices: np.ndarray  # Tx3 uint32
    lod: int = 0
    # Pixel (col, fila) de la rejilla del que salio cada vertice. Es lo que
    # permite proyectar una caja 2D (busqueda de objetos) sobre la malla.
    pixels: np.ndarray | None = None

    @property
    def tri_count(self) -> int:
        return int(self.indices.shape[0])


@dataclass
class PipelineStats:
    """Instantanea de metricas que consume la UI."""

    fps_capture: float = 0.0
    fps_pipeline: float = 0.0
    latency_ms: float = 0.0
    depth_ms: float = 0.0
    track_ms: float = 0.0
    integrate_ms: float = 0.0
    mesh_ms: float = 0.0
    points_total: int = 0
    triangles: int = 0
    chunks: int = 0
    voxel_size: float = 0.0
    tracking: str = TrackingState.INIT.value
    inliers: int = 0
    frames: int = 0
    connected: bool = False
    depth_backend: str = ""
    updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def pose_translation(T: np.ndarray) -> np.ndarray:
    return T[:3, 3]


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angulo de la rotacion en grados, numericamente seguro."""
    cos = (np.trace(R[:3, :3]) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
