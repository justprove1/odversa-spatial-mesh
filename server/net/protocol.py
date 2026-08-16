"""Protocolo binario entre servidor y visor.

Formato de cada mensaje:

    [uint32 LE: longitud de la cabecera][cabecera JSON utf-8][carga binaria]

La cabecera indica el tipo y los tamanos de cada bloque de la carga. Video y
malla viajan en el *mismo* mensaje: asi el visor nunca pinta una malla de un
frame sobre la imagen de otro, que es lo que hace que el modo REAL+MALLA se vea
desincronizado.
"""

from __future__ import annotations

import json
import struct

import numpy as np

_HEADER = struct.Struct("<I")


def encode(msg_type: str, header: dict | None = None, blocks: list[bytes] | None = None) -> bytes:
    head = {"type": msg_type}
    if header:
        head.update(header)
    if blocks:
        head["blocks"] = [len(b) for b in blocks]
    raw = json.dumps(head, separators=(",", ":")).encode("utf-8")
    # La cabecera se rellena hasta multiplo de 4 para que los bloques binarios
    # queden alineados: asi el visor crea Float32Array/Uint32Array como vistas
    # directas sobre el buffer recibido, sin copiar nada.
    pad = (-len(raw)) % 4
    raw += b" " * pad
    out = [_HEADER.pack(len(raw)), raw]
    if blocks:
        out.extend(blocks)
    return b"".join(out)


def encode_json(msg_type: str, payload: dict) -> str:
    """Mensajes de control (texto), para lo que no lleva binario."""
    return json.dumps({"type": msg_type, **payload}, separators=(",", ":"))


def encode_frame(jpeg: bytes | None, vertices: np.ndarray, indices: np.ndarray,
                 header: dict, highlight: np.ndarray | None = None) -> bytes:
    """Mensaje `frame`: imagen + malla (+ resaltado de busqueda) del mismo instante."""
    # Orden deliberado: primero los bloques que el visor mapea como arrays
    # tipados (necesitan alineacion a 4 bytes), luego el resaltado (uint8, le da
    # igual la alineacion) y el JPEG al final, que se entrega como Blob.
    blocks: list[bytes] = [
        np.ascontiguousarray(vertices, dtype=np.float32).tobytes(),
        np.ascontiguousarray(indices, dtype=np.uint32).tobytes(),
    ]
    order = ["vertices", "indices"]
    if highlight is not None:
        blocks.append(np.ascontiguousarray(highlight, dtype=np.uint8).tobytes())
        order.append("highlight")
    if jpeg:
        blocks.append(jpeg)
        order.append("jpeg")

    head = dict(header)
    head["order"] = order
    head["vcount"] = int(vertices.shape[0])
    head["tcount"] = int(indices.shape[0])
    return encode("frame", head, blocks)


def decode(data: bytes) -> tuple[dict, list[bytes]]:
    """Solo se usa en pruebas: el visor lo hace en JavaScript."""
    (hlen,) = _HEADER.unpack_from(data, 0)
    head = json.loads(data[4 : 4 + hlen].decode("utf-8"))
    offset = 4 + hlen
    blocks = []
    for size in head.get("blocks", []):
        blocks.append(data[offset : offset + size])
        offset += size
    return head, blocks
