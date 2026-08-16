"""Configuracion central de Odversa Spatial Mesh.

Todos los parametros ajustables en caliente viven en `RuntimeConfig`. La UI del visor
puede modificarlos vía WebSocket sin reiniciar el servidor.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
MODELS_DIR = Path(os.environ.get("ODVERSA_MODELS_DIR", ROOT / "models"))
MAPS_DIR = ROOT / "maps"
CERTS_DIR = ROOT / "certs"


@dataclass
class ServerConfig:
    """Parametros de arranque (no cambian en caliente)."""

    host: str = "0.0.0.0"
    port: int = int(os.environ.get("ODVERSA_PORT", 10000))
    use_https: bool = os.environ.get("ODVERSA_HTTPS", "1") != "0"
    cert_file: Path = CERTS_DIR / "odversa.crt"
    key_file: Path = CERTS_DIR / "odversa.key"


@dataclass
class RuntimeConfig:
    """Parametros del pipeline ajustables desde la UI.

    Los nombres coinciden con los controles de la interfaz para que el mensaje
    `config` del WebSocket sea un mapeo directo.
    """

    # --- Profundidad -------------------------------------------------------
    depth_backend: str = "onnx"  # "onnx" | "stub"
    depth_model: str = "depth-anything-v2-small"
    # Lado equivalente de entrada de la red (el alto y el ancho reales respetan
    # el aspecto de la camara y son multiplos de 14, manteniendo esta area).
    # Medido en M5 (imagen 16:9): 266 -> 32 ms, 308 -> 44 ms, 336 -> 58 ms,
    # 448 -> 117 ms por frame.
    #
    # OJO con bajar de 266: ahi la red no se degrada, se DESMORONA. El error
    # de forma medido salta del 19% al 136%. Por eso el minimo esta acotado
    # tanto aqui como en el deslizador de la interfaz.
    depth_input_size: int = 308
    # Anclajes de la calibracion metrica. `near_m` fija la ESCALA del mundo
    # (la profundidad monocular no la tiene) y `far_m` el ALCANCE util.
    near_m: float = 1.0
    far_m: float = 8.0
    depth_smoothing: float = 0.55  # 0 = sin suavizado temporal, 1 = congelado
    spatial_smoothing: float = 0.25  # filtro bilateral: aplana sin fundir bordes
    # Filtro guiado por la imagen: pega los bordes de profundidad a los bordes
    # reales del objeto. Es lo que hace que un mando tenga silueta y no bulto.
    guided_filter: bool = True
    guided_radius: int = 4
    # Realce del relieve fino (mascara de enfoque sobre la profundidad).
    detail_boost: float = 0.45
    min_confidence: float = 0.08  # descarta pixeles poco fiables
    # Relieve minimo que la malla se molesta en representar, en centimetros a
    # 3 m de distancia (escala con la distancia: a 6 m seria el doble).
    # Es EL control de "cuanto detecta": con 3 cm, un teclado sobre una mesa no
    # genera ni un vertice. Medido: 3.0 cm -> 1.691 triangulos y 14 ms;
    # 1.2 cm -> 2.757 y 22 ms; 0.8 cm -> 3.279 y 27 ms.
    mesh_detail_cm: float = 1.2

    # Salto de profundidad relativo a partir del cual se corta el triangulo.
    # Es lo que evita las "cortinas" entre un objeto y el fondo.
    edge_tolerance: float = 0.055

    # --- IA de superficies (suelo, pared, mesa, techo) --------------------
    # Deteccion geometrica por planos: sin red extra, ~3 ms. Ademas de
    # etiquetar, ESTABILIZA: la profundidad se pega al plano detectado y las
    # superficies grandes dejan de temblar.
    surface_ai: bool = True
    surface_snap: float = 0.85  # cuanto se pega la profundidad al plano
    surface_max_planes: int = 4

    # --- Resolucion de proceso geometrico ---------------------------------
    # Ancho al que se reduce el mapa de profundidad para generar geometria.
    # Control de "densidad de malla": 224 => rejilla de 224x126 puntos.
    # Se puede permitir ser alta porque la malla es ADAPTATIVA: solo subdivide
    # donde hay detalle. Medido, 224 adaptativa sale mas barata que 128
    # uniforme (8.241 triangulos frente a 14.054).
    proc_width: int = 224
    # Submuestreo adicional al retroproyectar (1 = todos los pixeles).
    point_stride: int = 1
    max_points_per_frame: int = 40000

    # --- Tracking ----------------------------------------------------------
    track_max_features: int = 700
    track_min_inliers: int = 25
    track_use_imu: bool = True
    # Rechaza saltos de pose imposibles entre frames consecutivos.
    max_translation_per_frame_m: float = 0.45
    max_rotation_per_frame_deg: float = 25.0

    # --- Memoria espacial (TSDF por voxel-hash) ---------------------------
    voxel_size: float = 0.04  # "densidad de malla": metros por voxel
    chunk_voxels: int = 8  # voxeles por lado de cada chunk
    truncation_voxels: float = 3.0  # banda TSDF en unidades de voxel
    max_integration_depth_m: float = 6.0
    min_integration_depth_m: float = 0.25
    max_weight: float = 24.0  # saturacion del peso => estabilidad temporal
    min_weight_to_mesh: float = 2.0  # descarta geometria poco observada
    max_chunks: int = 120000  # techo de memoria del mapa

    # --- Mallado -----------------------------------------------------------
    mesh_enabled: bool = True
    mesh_hz: float = 8.0  # frecuencia maxima de re-mallado
    max_chunks_per_pass: int = 220
    mesh_smoothing_iters: int = 1  # laplaciano sobre vertices (estabilidad)
    decimate_distance_m: float = 4.0  # a partir de aqui se simplifica el chunk
    decimate_factor: float = 2.0  # tamano de celda de clustering / voxel_size

    # --- Streaming ---------------------------------------------------------
    relay_video: bool = True
    jpeg_relay_quality: int = 72

    def as_dict(self) -> dict:
        return asdict(self)

    def apply(self, updates: dict) -> list[str]:
        """Aplica cambios validados. Devuelve las claves realmente modificadas."""
        changed: list[str] = []
        for key, value in updates.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, str):
                    value = str(value)
            except (TypeError, ValueError):
                continue
            if value != current:
                setattr(self, key, value)
                changed.append(key)
        return changed


# Claves cuyo cambio obliga a reconstruir el mapa desde cero.
REBUILD_KEYS = {"voxel_size", "chunk_voxels", "truncation_voxels"}

DEFAULT_CONFIG_FIELDS = [f for f in RuntimeConfig().as_dict()]
_ = field  # mantiene la importacion util si se anaden campos con default_factory
