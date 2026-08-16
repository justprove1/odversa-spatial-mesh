"""Previsualiza fuera del navegador lo que vera el visor.

Pasa una imagen por el pipeline y genera un PNG con los tres modos mas una
vista orbitada, usando el mismo criterio de oclusion que el visor: relleno
opaco por delante del alambre, dibujado de atras hacia delante.

    .venv/bin/python -m tools.render_preview foto.jpg salida.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from server.config import RuntimeConfig
from server.core.camera import Intrinsics
from server.modules.depth_estimation import DepthEstimation
from server.modules.mesh_optimization import drop_degenerate, laplacian_smooth
from server.modules.mesh_reconstruction import MeshReconstruction
from server.modules.point_cloud import PointCloud

MESH_BGR = (255, 224, 95)  # el cian del visor, en BGR


def rasterize(vertices: np.ndarray, faces: np.ndarray, intr: Intrinsics,
              R: np.ndarray, t: np.ndarray, fill: bool) -> tuple[np.ndarray, np.ndarray]:
    """Dibuja la malla con algoritmo del pintor. Devuelve (color, mascara)."""
    canvas = np.zeros((intr.height, intr.width, 3), np.uint8)
    mask = np.zeros((intr.height, intr.width), np.uint8)
    if faces.shape[0] == 0:
        return canvas, mask

    cam = vertices @ R.T + t
    z = cam[:, 2]
    uv = np.empty((cam.shape[0], 2), np.float32)
    safe_z = np.where(z > 1e-3, z, 1e-3)
    uv[:, 0] = cam[:, 0] / safe_z * intr.fx + intr.cx
    uv[:, 1] = cam[:, 1] / safe_z * intr.fy + intr.cy

    tri_z = z[faces].mean(axis=1)
    visible = (z[faces] > 0.05).all(axis=1)
    faces, tri_z = faces[visible], tri_z[visible]
    if faces.shape[0] == 0:
        return canvas, mask

    # Sombreado por normales, el mismo criterio que el shader del visor: cuanto
    # mas de canto se ve un triangulo, mas apagada su linea. Sin esto una malla
    # vista desde la camara original se proyecta como una rejilla uniforme y no
    # se distingue una superficie de otra, asi que la previsualizacion mentiria.
    a, b, c = cam[faces[:, 0]], cam[faces[:, 1]], cam[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-9)
    centers = (a + b + c) / 3.0
    view = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    facing = np.abs((normals * view).sum(axis=1))
    shade = 0.28 + 0.72 * np.clip(facing, 0.0, 1.0) ** 0.7

    # De atras hacia delante: lo cercano tapa lo lejano.
    order = np.argsort(-tri_z)
    pts = uv[faces[order]].astype(np.int32)
    shade = shade[order]

    for tri, s in zip(pts, shade):
        color = tuple(float(ch) * float(s) for ch in MESH_BGR)
        if fill:
            cv2.fillConvexPoly(canvas, tri, (0, 0, 0))
            cv2.fillConvexPoly(mask, tri, 0)
        cv2.polylines(canvas, [tri], True, color, 1, cv2.LINE_AA)
        cv2.polylines(mask, [tri], True, 255, 1, cv2.LINE_AA)
    return canvas, mask


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (10, 8, 6), -1)
    cv2.putText(out, text, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 235, 240), 1,
                cv2.LINE_AA)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    src = cv2.imread(argv[0])
    if src is None:
        print(f"no se pudo leer {argv[0]}")
        return 1

    cfg = RuntimeConfig()
    if len(argv) > 2:
        cfg.proc_width = int(argv[2])
    depth_estimator = DepthEstimation(cfg)
    pc = PointCloud()
    mesher = MeshReconstruction()

    h0, w0 = src.shape[:2]
    view_w = 720
    view_h = int(round(view_w * h0 / w0))
    real = cv2.resize(src, (view_w, view_h), interpolation=cv2.INTER_AREA)

    pw = cfg.proc_width
    ph = int(round(pw * h0 / w0))
    intr_proc = Intrinsics.from_fov(w0, h0).scaled(pw, ph)
    intr_view = Intrinsics.from_fov(w0, h0).scaled(view_w, view_h)

    # Dos pasadas: la calibracion por EMA necesita un frame para asentarse.
    for _ in range(2):
        result = depth_estimator.estimate(src, (pw, ph))
    depth = pc.refine_depth(result.depth, src, cfg)

    payload = mesher.build_live(depth, result.confidence, intr_proc, cfg.near_m, cfg.far_m,
                                edge_tolerance=cfg.edge_tolerance, min_conf=cfg.min_confidence)
    v = laplacian_smooth(payload.vertices, payload.indices, cfg.mesh_smoothing_iters, 0.45)
    v, f = drop_degenerate(v, payload.indices)
    print(f"malla: {v.shape[0]} vertices, {f.shape[0]} triangulos")
    print(f"profundidad: {depth.min():.2f} .. {depth.max():.2f} m (media {depth.mean():.2f})")
    print(f"extension XYZ: {np.round(v.max(0) - v.min(0), 2)}")

    eye = np.eye(3)
    zero = np.zeros(3)

    mesh_only, _ = rasterize(v, f, intr_view, eye, zero, fill=True)
    _, lines = rasterize(v, f, intr_view, eye, zero, fill=True)
    overlay = real.copy()
    overlay[lines > 0] = (0.25 * overlay[lines > 0] + 0.75 * np.array(MESH_BGR)).astype(np.uint8)

    # Vista orbitada: 32 grados alrededor del centro de la geometria.
    center = np.median(v, axis=0)
    angle = np.radians(32.0)
    Ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                   [-np.sin(angle), 0, np.cos(angle)]])
    orbit, _ = rasterize(v, f, intr_view, Ry, center - Ry @ center, fill=True)

    top = np.hstack([label(real, "SOLO REAL"), label(overlay, "REAL + MALLA")])
    bottom = np.hstack([label(mesh_only, "SOLO MALLA"),
                        label(orbit, "SOLO MALLA · orbita 32 grados")])
    out = np.vstack([top, bottom])

    dest = Path(argv[1])
    cv2.imwrite(str(dest), out)
    print(f"escrito {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
