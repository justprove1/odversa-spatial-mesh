"""DepthEstimation - estimacion de profundidad monocular.

Este modulo define la *interfaz* y el post-proceso comun. Los modelos concretos
viven en `depth_backends/` y se registran aqui, de forma que cambiar de red
(Depth Anything, ZipDepth, un modelo CoreML propio, un sensor LiDAR real...)
no obliga a tocar el resto de Odversa: basta con implementar `DepthBackend`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..config import RuntimeConfig
from ..core.types import DepthResult

#: Margen por encima de `far_m` antes de dar la profundidad por perdida.
FAR_HEADROOM = 2.5


class DepthBackend(ABC):
    """Contrato minimo de cualquier estimador de profundidad."""

    #: nombre legible que se muestra en la UI
    name: str = "backend"
    #: True si la salida ya esta en metros; False si es disparidad relativa
    metric: bool = False

    @abstractmethod
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Devuelve un mapa float32 HxW.

        Si `metric` es False, el valor es disparidad relativa (mayor = mas cerca).
        Si es True, son metros directamente.
        """

    def close(self) -> None:  # pragma: no cover - opcional
        pass


def build_backend(cfg: RuntimeConfig) -> DepthBackend:
    """Fabrica de backends. Cae al stub si el modelo real no esta disponible."""
    if cfg.depth_backend == "stub":
        from .depth_backends.stub import StubBackend

        return StubBackend()

    if cfg.depth_backend == "onnx":
        try:
            from .depth_backends.onnx_depth_anything import OnnxDepthAnything

            return OnnxDepthAnything(model=cfg.depth_model, input_size=cfg.depth_input_size)
        except Exception as exc:  # modelo no descargado, ORT roto, etc.
            print(f"[depth] backend onnx no disponible ({exc}); usando stub")
            from .depth_backends.stub import StubBackend

            return StubBackend()

    if cfg.depth_backend == "torch":  # extra opcional
        from .depth_backends.torch_depth_anything import TorchDepthAnything

        return TorchDepthAnything(model=cfg.depth_model)

    raise ValueError(f"backend de profundidad desconocido: {cfg.depth_backend}")


class DepthEstimation:
    """Envuelve un backend y produce profundidad metrica y estable en el tiempo.

    Los modelos monoculares relativos devuelven disparidad sin escala y con un
    rango que baila entre frames. Aqui se hace:

    1. Ajuste afin en disparidad, que es lo fisicamente correcto: la salida de
       la red es proporcional a 1/z salvo una transformacion afin desconocida,
       asi que se resuelve `1/z = a*d + b` con dos anclajes.
    2. Los anclajes son los percentiles 10 y 90, no el minimo y el maximo. Con
       los extremos, un reflejo o un pixel de cielo movia toda la escala; con
       los percentiles robustos la geometria deja de respirar. Ademas se
       suavizan con EMA, de modo que meter la mano delante del objetivo no
       reescala la habitacion entera.
    3. Suavizado temporal por pixel y confianza a partir del gradiente (los
       bordes de profundidad son justo donde el modelo se inventa cosas).
    """

    #: percentiles usados como anclajes de la calibracion
    ANCHOR_LOW = 10.0
    ANCHOR_HIGH = 90.0

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.backend = build_backend(cfg)
        self._p_low: float | None = None
        self._p_high: float | None = None
        self._prev_depth: np.ndarray | None = None
        self._last_ms = 0.0

    # -- ciclo de vida ------------------------------------------------------
    @property
    def backend_name(self) -> str:
        return self.backend.name

    def reset(self) -> None:
        self._p_low = None
        self._p_high = None
        self._prev_depth = None

    def rebuild(self, cfg: RuntimeConfig) -> None:
        """Recrea el backend (cambio de modelo o de resolucion de entrada)."""
        try:
            self.backend.close()
        except Exception:
            pass
        self.cfg = cfg
        self.backend = build_backend(cfg)
        self.reset()

    # -- inferencia ---------------------------------------------------------
    def estimate(self, bgr: np.ndarray, out_size: tuple[int, int]) -> DepthResult:
        """`out_size` es (ancho, alto) del mapa de profundidad devuelto."""
        t0 = time.perf_counter()
        raw = self.backend.infer(bgr)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        w, h = out_size
        if raw.shape[1] != w or raw.shape[0] != h:
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        raw = raw.astype(np.float32)

        if self.backend.metric:
            depth = np.clip(raw, 0.0, self.cfg.far_m)
            disparity = None
        else:
            disparity = raw
            depth = self._disparity_to_depth(raw)

        depth = self._temporal_filter(depth)
        conf = self._confidence(depth)

        self._last_ms = infer_ms
        return DepthResult(depth=depth, confidence=conf, raw_disparity=disparity, infer_ms=infer_ms)

    # -- internos -----------------------------------------------------------
    def _anchors(self, raw: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(raw, (self.ANCHOR_LOW, self.ANCHOR_HIGH))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
            lo, hi = float(raw.min()), float(raw.max() + 1e-6)
        k = 0.15  # EMA: la escala cambia despacio aunque la escena cambie rapido
        self._p_low = lo if self._p_low is None else (1 - k) * self._p_low + k * lo
        self._p_high = hi if self._p_high is None else (1 - k) * self._p_high + k * hi
        return float(self._p_low), float(self._p_high)

    def _disparity_to_depth(self, raw: np.ndarray) -> np.ndarray:
        """Resuelve `1/z = a*d + b` anclando p90 -> `near_m` y p10 -> `far_m`.

        `near_m` fija la escala del mundo (la profundidad monocular no la tiene)
        y `far_m` el alcance. Los pixeles que caen mas alla del alcance se
        saturan a `far_m` y el filtro de validez los descarta despues: son
        justo las zonas -cielo, fondo de un pasillo- donde el modelo no sabe.
        """
        near = max(self.cfg.near_m, 0.1)
        far = max(self.cfg.far_m, near * 1.5)
        p_lo, p_hi = self._anchors(raw)

        a = (1.0 / near - 1.0 / far) / max(p_hi - p_lo, 1e-6)
        b = 1.0 / far - a * p_lo
        inv_z = a * raw + b

        # El limite duro NO es `far`, sino bastante mas alla.
        #
        # Los anclajes son percentiles: p10 va a `far_m` por construccion, o sea
        # que en CUALQUIER escena el 10% mas lejano cae justo en `far`. Si ahi
        # mismo se recorta y se descarta, se esta tirando la pared del fondo de
        # todas las habitaciones del mundo, siempre. `far_m` es "la distancia
        # tipica del fondo", no "el mundo se acaba aqui".
        limit = far * FAR_HEADROOM
        depth = 1.0 / np.maximum(inv_z, 1.0 / limit)
        return np.clip(depth, near * 0.4, limit).astype(np.float32)

    def _temporal_filter(self, depth: np.ndarray) -> np.ndarray:
        a = float(np.clip(self.cfg.depth_smoothing, 0.0, 0.95))
        if self._prev_depth is None or self._prev_depth.shape != depth.shape or a <= 0.0:
            self._prev_depth = depth.copy()
            return depth
        # Mezcla adaptativa: donde la profundidad cambia mucho (movimiento real)
        # se confia en el frame nuevo; donde apenas cambia, se promedia y el
        # temblor desaparece.
        diff = np.abs(depth - self._prev_depth)
        rel = diff / np.maximum(depth, 0.2)

        # El peso del historico CRECE con la distancia. Motivo medido: en la
        # zona lejana la red amplifica cualquier variacion (alli 1/z es
        # minusculo y un pelo de disparidad son 60 cm de temblor), y a la vez es
        # la zona donde menos plausible es que la escena cambie rapido de
        # verdad: una pared a 15 m no se mueve. Reforzar la memoria solo alli
        # quita el temblor sin poner lag donde si hay movimiento cercano.
        far = max(self.cfg.far_m, 1.0)
        far_boost = np.clip(depth / far - 0.35, 0.0, 1.0)
        a_eff = np.minimum(a + (0.95 - a) * far_boost, 0.95)

        w = a_eff * np.exp(-(rel / 0.12) ** 2)  # peso del historico
        out = w * self._prev_depth + (1.0 - w) * depth
        self._prev_depth = out
        return out.astype(np.float32)

    def _confidence(self, depth: np.ndarray) -> np.ndarray:
        """Baja confianza en discontinuidades reales y a larga distancia.

        Se usa el laplaciano, no el gradiente. Un suelo visto en escorzo tiene
        un gradiente de profundidad enorme por pixel y sin embargo es la
        superficie mas fiable de la escena: penalizarlo por pendiente borraba el
        suelo entero. El laplaciano vale cero en cualquier plano, este inclinado
        o no, y solo se dispara en los saltos, que es lo que hay que castigar.
        Medido sobre escenas reales, la mediana del laplaciano relativo es ~0.02
        y los bordes autenticos viven por encima de 0.6: hay margen de sobra.
        """
        lap = np.abs(cv2.Laplacian(depth, cv2.CV_32F, ksize=3))
        edge_term = np.exp(-(lap / np.maximum(0.25 * depth, 0.02)) ** 2)
        # La distancia solo empieza a restar confianza cerca del limite real, no
        # al llegar a `far_m`: ahi todavia hay geometria buena.
        reference = max(self.cfg.far_m * FAR_HEADROOM, 1.0)
        range_term = np.clip(1.0 - (depth / reference) ** 1.5, 0.05, 1.0)
        return (edge_term * range_term).astype(np.float32)

    @property
    def last_ms(self) -> float:
        return self._last_ms
