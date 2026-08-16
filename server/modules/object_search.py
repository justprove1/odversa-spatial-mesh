"""ObjectSearch - busqueda de objetos por texto (vocabulario abierto).

El usuario escribe "vitrina" y, cuando la camara ve una, se resalta en la
malla. Eso no lo puede hacer un detector clasico (solo conoce sus 80 clases
entrenadas): hace falta un modelo texto-imagen. Se usa OWL-ViT, que compara
CUALQUIER texto contra regiones de la imagen y devuelve cajas con puntuacion.

Como encaja sin hundir los fps:

* La deteccion corre en SU PROPIO hilo a ~1-2 Hz sobre el frame mas reciente.
  El pipeline de malla no la espera nunca: consulta las ultimas cajas, que
  siguen valiendo unos frames porque la camara se mueve despacio comparada con
  el ritmo de refresco de la busqueda.
* Sin consulta activa no se ejecuta nada: coste cero cuando no se busca.

El espanol funciona a medias en CLIP (esta entrenado sobre todo en ingles), asi
que cada consulta se lanza por duplicado: tal cual la escribio el usuario y
traducida por un diccionario de objetos domesticos comunes. Se queda la mejor
puntuacion de las dos. Si el usuario escribe en ingles, la primera via ya vale.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import MODELS_DIR

#: media y desviacion de CLIP (no son las de ImageNet)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

#: traduccion rapida de objetos comunes; CLIP entiende mucho mejor el ingles
ES_EN = {
    "vitrina": "display cabinet", "mesa": "table", "silla": "chair",
    "sofa": "sofa", "sofá": "sofa", "puerta": "door", "ventana": "window",
    "cama": "bed", "lampara": "lamp", "lámpara": "lamp", "cuadro": "picture frame",
    "planta": "plant", "maceta": "potted plant", "televisor": "television",
    "tele": "television", "television": "television", "televisión": "television",
    "ordenador": "computer", "portatil": "laptop", "portátil": "laptop",
    "teclado": "keyboard", "raton": "computer mouse", "ratón": "computer mouse",
    "pantalla": "monitor", "monitor": "monitor", "movil": "mobile phone",
    "móvil": "mobile phone", "telefono": "phone", "teléfono": "phone",
    "libro": "book", "libros": "books", "estanteria": "bookshelf",
    "estantería": "bookshelf", "armario": "wardrobe", "nevera": "fridge",
    "frigorifico": "fridge", "frigorífico": "fridge", "horno": "oven",
    "microondas": "microwave", "fregadero": "sink", "lavabo": "sink",
    "espejo": "mirror", "taza": "mug", "vaso": "glass", "botella": "bottle",
    "cojin": "cushion", "cojín": "cushion", "alfombra": "rug",
    "cortina": "curtain", "radiador": "radiator", "enchufe": "power outlet",
    "interruptor": "light switch", "escalera": "stairs", "reloj": "clock",
    "mochila": "backpack", "bolso": "bag", "zapatos": "shoes",
    "guitarra": "guitar", "piano": "piano", "bici": "bicycle",
    "bicicleta": "bicycle", "coche": "car", "moto": "motorcycle",
    "perro": "dog", "gato": "cat", "persona": "person", "gente": "people",
    "mando": "remote control", "consola": "game console", "altavoz": "speaker",
    "auriculares": "headphones", "ventilador": "fan", "papelera": "trash bin",
    "caja": "box", "mueble": "furniture", "banco": "bench", "nave": "spaceship",
    "avion": "airplane", "avión": "airplane", "extintor": "fire extinguisher",
}


class ObjectSearch:
    """Detector de vocabulario abierto en un hilo propio, a bajo ritmo."""

    def __init__(self, model: str = "owlvit-base-patch32", input_size: int = 768,
                 interval: float = 0.6, threshold: float = 0.13):
        self.model_dir = Path(MODELS_DIR) / model
        self.input_size = input_size
        self.interval = interval
        self.threshold = threshold

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._query: str = ""
        self._tokens: tuple[np.ndarray, np.ndarray] | None = None
        self._frame: np.ndarray | None = None
        self._boxes: list[dict] = []
        self._boxes_t = 0.0
        self._status = "apagado"
        self._infer_ms = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session = None
        self._tokenizer = None

    # -- ciclo de vida ------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="odversa-search",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)

    # -- entrada ------------------------------------------------------------
    def set_query(self, text: str) -> None:
        """Cambia (o borra) lo que se busca. Vacio = apagar la busqueda."""
        text = (text or "").strip().lower()[:64]
        with self._cond:
            if text == self._query:
                return
            self._query = text
            self._tokens = None  # se retokeniza en el hilo
            self._boxes = []
            self._status = "buscando…" if text else "apagado"
            self._cond.notify_all()

    def submit(self, bgr: np.ndarray) -> None:
        """El pipeline deja aqui el frame mas reciente; no bloquea nunca."""
        if not self._query:
            return
        with self._cond:
            self._frame = bgr
            self._cond.notify_all()

    # -- salida -------------------------------------------------------------
    def current(self) -> dict:
        """Cajas vigentes + estado, para el pipeline y el visor."""
        with self._lock:
            fresh = (time.time() - self._boxes_t) < 2.5
            return {
                "query": self._query,
                "boxes": list(self._boxes) if fresh else [],
                "status": self._status,
                "ms": round(self._infer_ms),
            }

    # -- hilo de trabajo ----------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while not self._stop.is_set() and (not self._query or self._frame is None):
                    self._cond.wait(0.5)
                if self._stop.is_set():
                    return
                frame, self._frame = self._frame, None
                query = self._query

            try:
                self._ensure_model()
                boxes, ms = self._detect(frame, query)
            except Exception as exc:
                with self._lock:
                    self._status = f"error: {str(exc)[:60]}"
                time.sleep(2.0)
                continue

            with self._lock:
                if self._query == query:  # la consulta no cambio mientras tanto
                    self._boxes = boxes
                    self._boxes_t = time.time()
                    self._infer_ms = ms
                    self._status = (f"{len(boxes)} coincidencia" +
                                    ("s" if len(boxes) != 1 else "")) if boxes \
                                   else "sin coincidencias"
            time.sleep(self.interval)

    # -- modelo -------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = self.model_dir / "onnx" / "model.onnx"
        tok_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tok_path.exists():
            raise RuntimeError("modelo de busqueda no descargado")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(str(model_path), opts,
                                             providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_padding(length=16)
        self._tokenizer.enable_truncation(max_length=16)

    def _prompts(self, query: str) -> list[str]:
        prompts = [f"a photo of a {query}"]
        translated = ES_EN.get(query)
        if translated and translated != query:
            prompts.append(f"a photo of a {translated}")
        return prompts

    def _detect(self, bgr: np.ndarray, query: str) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        side = self.input_size
        rgb = cv2.cvtColor(cv2.resize(bgr, (side, side)), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0) - CLIP_MEAN) / CLIP_STD
        pixel_values = np.ascontiguousarray(x.transpose(2, 0, 1)[None])

        prompts = self._prompts(query)
        encodings = [self._tokenizer.encode(p) for p in prompts]
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        outputs = self._session.run(None, {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention,
        })
        # Salidas OWL-ViT: logits (1, P, Q) y pred_boxes (1, P, 4) en cxcywh.
        logits, pred_boxes = outputs[0], outputs[1]
        scores = 1.0 / (1.0 + np.exp(-logits[0]))  # sigmoide
        best = scores.max(axis=1)  # la mejor de las dos redacciones
        keep = best > self.threshold
        ms = (time.perf_counter() - t0) * 1000.0
        if not keep.any():
            return [], ms

        boxes = pred_boxes[0][keep]
        conf = best[keep]
        order = np.argsort(-conf)[:8]
        boxes, conf = boxes[order], conf[order]

        # Supresion de solapados: OWL devuelve racimos de cajas casi identicas.
        chosen: list[int] = []
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            duplicate = False
            for j in chosen:
                cx2, cy2, w2, h2 = boxes[j]
                if abs(cx - cx2) < 0.5 * (w + w2) / 2 and abs(cy - cy2) < 0.5 * (h + h2) / 2:
                    duplicate = True
                    break
            if not duplicate:
                chosen.append(i)
            if len(chosen) >= 3:
                break

        out = []
        for i in chosen:
            cx, cy, w, h = (float(v) for v in boxes[i])
            out.append({
                "x0": max(0.0, cx - w / 2), "y0": max(0.0, cy - h / 2),
                "x1": min(1.0, cx + w / 2), "y1": min(1.0, cy + h / 2),
                "score": round(float(conf[i]), 3),
            })
        return out, ms
