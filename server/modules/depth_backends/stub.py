"""Backend de profundidad sin red neuronal.

Sirve para dos cosas: arrancar el pipeline completo aunque no se haya descargado
ningun modelo, y como referencia de latencia cero al medir el resto del sistema.

La heuristica es deliberadamente tonta (los pixeles bajos de la imagen se
consideran suelo cercano y el brillo modula la distancia), pero produce un mapa
plausible y suave, suficiente para validar tracking, fusion y mallado.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..depth_estimation import DepthBackend


class StubBackend(DepthBackend):
    name = "stub-heuristico"
    metric = False

    def __init__(self, work_width: int = 256):
        self.work_width = work_width

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        tw = self.work_width
        th = max(1, int(round(h * tw / w)))
        small = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Gradiente vertical: la parte baja de la imagen suele ser suelo cercano.
        ramp = np.linspace(0.15, 1.0, th, dtype=np.float32)[:, None]
        # La textura fina tambien correlaciona con cercania.
        detail = cv2.GaussianBlur(gray, (0, 0), 1.2) - cv2.GaussianBlur(gray, (0, 0), 7.0)
        disparity = 0.65 * ramp + 0.25 * gray + 2.0 * np.abs(detail)
        disparity = cv2.GaussianBlur(disparity, (0, 0), 2.0)
        return disparity.astype(np.float32)
