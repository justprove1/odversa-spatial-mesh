"""Guardado del mapa 3D a disco, sin dependencias externas."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import MAPS_DIR


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_ply(vertices: np.ndarray, faces: np.ndarray, path: Path | None = None) -> Path:
    """PLY binario little-endian: lo abre Blender, MeshLab, CloudCompare..."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or MAPS_DIR / f"odversa-mesh-{_stamp()}.ply"

    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Odversa Spatial Mesh\n"
        f"element vertex {v.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {f.shape[0]}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    face_rec = np.empty(f.shape[0], dtype=[("n", "u1"), ("i", "<u4", (3,))])
    face_rec["n"] = 3
    face_rec["i"] = f

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(v.tobytes())
        fh.write(face_rec.tobytes())
    return path


def save_obj(vertices: np.ndarray, faces: np.ndarray, path: Path | None = None) -> Path:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or MAPS_DIR / f"odversa-mesh-{_stamp()}.obj"
    with open(path, "w") as fh:
        fh.write("# Odversa Spatial Mesh\n")
        np.savetxt(fh, vertices, fmt="v %.5f %.5f %.5f")
        np.savetxt(fh, faces + 1, fmt="f %d %d %d")
    return path
