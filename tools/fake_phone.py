"""Movil simulado: alimenta el servidor sin necesidad de telefono.

Se conecta al WebSocket del movil y manda frames como lo haria la pagina de
captura, incluidos los 8 bytes de marca de tiempo. Sirve para probar el visor y
para medir el pipeline de punta a punta.

    .venv/bin/python -m tools.fake_phone --image foto.jpg --fps 20
    .venv/bin/python -m tools.fake_phone --scene --url ws://localhost:8420/ws/phone
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time

import cv2
import numpy as np
import websockets

from tools.synthetic_scene import encode_jpeg, render


def pan_crop(img: np.ndarray, t: float, out_w: int, out_h: int) -> np.ndarray:
    """Recorte que se desplaza: imita un movimiento suave de camara."""
    h, w = img.shape[:2]
    scale = min(w / out_w, h / out_h) * 0.82
    cw, ch = int(out_w * scale), int(out_h * scale)
    max_dx, max_dy = w - cw, h - ch
    x = int((0.5 + 0.42 * np.sin(t * 0.5)) * max_dx)
    y = int((0.5 + 0.30 * np.sin(t * 0.31)) * max_dy)
    crop = img[y:y + ch, x:x + cw]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


async def stream(url: str, image: str | None, fps: float, width: int, seconds: float) -> None:
    src = cv2.imread(image) if image else None
    if image and src is None:
        raise SystemExit(f"no se pudo leer {image}")

    height = int(round(width * 3 / 4))
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "width": width, "height": height,
                                  "hfov_deg": None, "device": "fake_phone"}))
        print(f"conectado a {url} · {width}x{height} @ {fps} fps")

        # Igual que el movil real: no se envia mas de `max_in_flight` frames sin
        # que el servidor confirme. Sin esto la prueba mide una latencia que el
        # movil de verdad nunca tendria.
        state = {"in_flight": 0}

        async def read_acks() -> None:
            try:
                async for message in ws:
                    if isinstance(message, str) and '"ack"' in message:
                        state["in_flight"] = max(0, state["in_flight"] - 1)
            except Exception:
                pass

        reader = asyncio.create_task(read_acks())
        t0 = time.time()
        sent = 0
        period = 1.0 / fps
        while time.time() - t0 < seconds:
            if state["in_flight"] >= 2:
                await asyncio.sleep(0.004)
                continue
            t = time.time() - t0
            frame = pan_crop(src, t, width, height) if src is not None else render(width, height, t)
            payload = struct.pack("<d", time.time()) + encode_jpeg(frame, 72)
            await ws.send(payload)
            state["in_flight"] += 1
            sent += 1
            await asyncio.sleep(period)
        reader.cancel()
        print(f"enviados {sent} frames en {time.time() - t0:.1f} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8420/ws/phone")
    ap.add_argument("--image", help="foto a emitir (se desplaza para simular movimiento)")
    ap.add_argument("--scene", action="store_true", help="usar la escena sintetica")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--seconds", type=float, default=600.0)
    args = ap.parse_args()
    if not args.image and not args.scene:
        args.scene = True
    asyncio.run(stream(args.url, args.image, args.fps, args.width, args.seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
