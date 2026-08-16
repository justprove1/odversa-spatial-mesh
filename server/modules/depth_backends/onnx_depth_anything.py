"""Backend ONNX Runtime para modelos tipo Depth Anything V2.

Se usa ONNX Runtime con el Execution Provider de CoreML, que en Apple Silicon
descarga la inferencia al Neural Engine / GPU sin arrastrar PyTorch. El modelo
por defecto es `onnx-community/depth-anything-v2-small` (~25M parametros), el
mejor compromiso actual entre calidad geometrica y latencia para tiempo real.

Cambiar de modelo es cambiar `model=`: mientras la red reciba una imagen NCHW
normalizada con estadisticas ImageNet y devuelva un mapa de disparidad, encaja.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ...config import MODELS_DIR
from ..depth_estimation import DepthBackend

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Repos de Hugging Face conocidos, por nombre corto.
KNOWN_MODELS = {
    "depth-anything-v2-small": ("onnx-community/depth-anything-v2-small", "onnx/model.onnx"),
    "depth-anything-v2-small-q8": ("onnx-community/depth-anything-v2-small", "onnx/model_int8.onnx"),
    "depth-anything-v2-base": ("onnx-community/depth-anything-v2-base", "onnx/model.onnx"),
}


def resolve_model_path(model: str) -> Path:
    """Devuelve la ruta local del .onnx, descargandolo si hace falta."""
    direct = Path(model)
    if direct.suffix == ".onnx" and direct.exists():
        return direct

    repo, filename = KNOWN_MODELS.get(model, (model, "onnx/model.onnx"))
    local_dir = MODELS_DIR / model
    cached = local_dir / filename
    if cached.exists():
        return cached

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(local_dir))
    return Path(path)


class OnnxDepthAnything(DepthBackend):
    metric = False

    def __init__(self, model: str = "depth-anything-v2-small", input_size: int = 308,
                 provider: str | None = None):
        self.model_path = resolve_model_path(model)
        self.name = f"onnx:{model}"
        self.input_size = self._round14(max(input_size, self.MIN_INPUT_SIZE))
        self.provider = (provider or os.environ.get("ODVERSA_DEPTH_PROVIDER", "cpu")).lower()

        # La sesion no se crea aqui: espera al primer frame, cuando ya se conoce
        # la relacion de aspecto real de la camara. Ver `_ensure_session`.
        self.session: ort.InferenceSession | None = None
        self.in_h = self.in_w = self.input_size
        self.active_provider = ""

    def _ensure_session(self, aspect: float) -> None:
        """Crea (o recrea) la sesion para una relacion de aspecto concreta.

        Antes la imagen se metia en una entrada cuadrada: una imagen 4:3 se
        aplastaba horizontalmente antes de estimar la profundidad y se estiraba
        de vuelta despues. Eso deforma la geometria y se lleva por delante el
        detalle de los objetos pequenos, que es justo lo que hay que conservar
        para reconocer un teclado o un mando.

        Aqui se eligen alto y ancho multiplos de 14 (el tamano de parche del
        modelo) que respetan el aspecto de la camara, manteniendo el area
        equivalente a `input_size`^2 para que el coste no se dispare.
        """
        area = float(self.input_size * self.input_size)
        w = self._round14(int(round((area * aspect) ** 0.5)))
        h = self._round14(int(round(w / max(aspect, 1e-3))))
        if self.session is not None and (h, w) == (self.in_h, self.in_w):
            return

        self.in_h, self.in_w = h, w
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 4
        # Clave para el rendimiento: el grafo exportado tiene alto/ancho dinamicos
        # y eso impide casi toda la optimizacion (338 ms/frame en un M5). Fijar
        # las dimensiones libres antes de crear la sesion lo baja a ~43 ms, y es
        # la razon de que la sesion dependa del aspecto.
        for name, value in (("batch_size", 1), ("height", h), ("width", w)):
            try:
                opts.add_free_dimension_override_by_name(name, value)
            except Exception:
                pass

        self.session = self._make_session(opts, self.provider)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_provider = self.session.get_providers()[0]

        # Si el modelo trae el tamano ya congelado, manda el del modelo.
        shape = self.session.get_inputs()[0].shape
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            self.in_h, self.in_w = int(shape[2]), int(shape[3])

    def _make_session(self, opts: ort.SessionOptions, provider: str) -> ort.InferenceSession:
        """Construye la sesion con el EP pedido y cae a CPU si falla.

        Nota medida en Apple Silicon (M5): con las dimensiones ya fijadas, el EP
        de CPU de ONNX Runtime es mas rapido que CoreML para este grafo (ViT con
        muchos `Gather`/`Squeeze` que CoreML no compila entero). Por eso el
        defecto es CPU y CoreML queda como opcion (`ODVERSA_DEPTH_PROVIDER=coreml`).
        """
        candidates: list = []
        if provider == "coreml" and "CoreMLExecutionProvider" in ort.get_available_providers():
            candidates.append(
                [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                 "CPUExecutionProvider"]
            )
        candidates.append(["CPUExecutionProvider"])
        last: Exception | None = None
        for providers in candidates:
            try:
                return ort.InferenceSession(str(self.model_path), opts, providers=providers)
            except Exception as exc:
                last = exc
        raise RuntimeError(f"no se pudo crear la sesion ONNX: {last}")

    #: Por debajo de este lado equivalente la red no se degrada, colapsa: el
    #: error de forma medido pasa del 19% al 136%. No es un umbral estetico.
    MIN_INPUT_SIZE = 266

    @staticmethod
    def _round14(v: int) -> int:
        """Depth Anything usa parches de 14 px: el lado debe ser multiplo de 14."""
        return max(14 * 4, int(round(v / 14.0)) * 14)

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.in_w, self.in_h), interpolation=cv2.INTER_CUBIC)
        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])  # 1x3xHxW

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        self._ensure_session(w / max(h, 1))
        x = self._preprocess(bgr)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        d = np.asarray(out, dtype=np.float32)
        while d.ndim > 2:  # (1,1,H,W) o (1,H,W) -> (H,W)
            d = d[0]
        return d

    @property
    def input_shape(self) -> tuple[int, int]:
        """(alto, ancho) reales de la entrada de la red."""
        return self.in_h, self.in_w

    def close(self) -> None:
        self.session = None
