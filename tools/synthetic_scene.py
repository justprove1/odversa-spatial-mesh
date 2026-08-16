"""Escena sintetica para probar Odversa sin movil.

Dibuja una habitacion en perspectiva (suelo a cuadros, dos paredes, techo y una
caja en medio) con un punto de fuga que se puede desplazar, de forma que al
animarla parece que la camara gira. No pretende ser realista: sirve para
comprobar que el pipeline produce geometria con estructura y que los tres modos
del visor se comportan.
"""

from __future__ import annotations

import numpy as np
import cv2


def render(width: int = 640, height: int = 480, t: float = 0.0) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    # Punto de fuga que oscila: simula un giro suave de la camara.
    vx = int(width * (0.5 + 0.16 * np.sin(t * 0.6)))
    vy = int(height * 0.52)

    # Techo y fondo.
    cv2.rectangle(img, (0, 0), (width, height), (26, 24, 22), -1)
    cv2.rectangle(img, (vx - 90, vy - 70), (vx + 90, vy + 70), (58, 54, 50), -1)

    # Suelo: lineas que convergen en el punto de fuga.
    for i in range(-14, 15):
        x = int(width * 0.5 + i * width * 0.16)
        cv2.line(img, (x, height), (vx, vy), (78, 74, 68), 1, cv2.LINE_AA)
    for k in range(1, 16):
        # Espaciado que se comprime hacia el horizonte.
        y = int(vy + (height - vy) * (k / 15.0) ** 2.2)
        cv2.line(img, (0, y), (width, y), (70, 66, 62), 1, cv2.LINE_AA)

    # Paredes laterales.
    cv2.fillPoly(img, [np.array([[0, 0], [vx - 90, vy - 70], [vx - 90, vy + 70], [0, height]])],
                 (44, 42, 40))
    cv2.fillPoly(img, [np.array([[width, 0], [vx + 90, vy - 70], [vx + 90, vy + 70],
                                 [width, height]])], (40, 38, 36))

    # Una caja apoyada en el suelo, cerca de la camara.
    bx = int(width * (0.34 + 0.05 * np.sin(t * 0.6)))
    by = int(height * 0.74)
    cv2.rectangle(img, (bx, by), (bx + 120, by + 90), (120, 116, 110), -1)
    cv2.rectangle(img, (bx, by), (bx + 120, by + 90), (150, 146, 140), 2)
    cv2.fillPoly(img, [np.array([[bx, by], [bx + 40, by - 32],
                                 [bx + 160, by - 32], [bx + 120, by]])], (140, 136, 130))

    # Textura fina: sin ella los modelos de profundidad se quedan sin pistas.
    noise = np.random.default_rng(int(t * 30) % 997).integers(0, 14, (height, width, 1), np.uint8)
    return cv2.add(img, np.repeat(noise, 3, axis=2))


def encode_jpeg(img: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("no se pudo codificar el JPEG")
    return buf.tobytes()
