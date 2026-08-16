"""Prueba del pipeline completo sin red ni navegador.

Recorre profundidad -> nube de puntos -> malla -> optimizacion -> protocolo,
sobre la escena sintetica, e informa de tiempos y de la geometria producida.

    .venv/bin/python -m tools.check_pipeline
"""

from __future__ import annotations

import time

import numpy as np

from server.config import RuntimeConfig
from server.core.camera import Intrinsics
from server.modules.depth_estimation import DepthEstimation
from server.modules.mesh_optimization import cluster_decimate, drop_degenerate, laplacian_smooth
from server.modules.mesh_reconstruction import MeshReconstruction
from server.modules.point_cloud import PointCloud
from server.net import protocol
from tools.synthetic_scene import encode_jpeg, render


def main() -> int:
    cfg = RuntimeConfig()
    depth = DepthEstimation(cfg)
    pc = PointCloud()
    mesher = MeshReconstruction()

    print(f"backend de profundidad: {depth.backend_name}")

    src = Intrinsics.from_fov(640, 480)
    pw = cfg.proc_width
    ph = int(round(pw * 480 / 640))
    intr = src.scaled(pw, ph)
    print(f"rejilla de proceso: {pw}x{ph}  vfov={intr.vfov_deg:.1f} grados")

    timings = {"depth": [], "cloud": [], "mesh": [], "opt": []}
    last = None

    for i in range(6):
        img = render(640, 480, t=i * 0.25)

        t0 = time.perf_counter()
        result = depth.estimate(img, (pw, ph))
        timings["depth"].append((time.perf_counter() - t0) * 1000)

        d = pc.refine_depth(result.depth, img, cfg)

        t0 = time.perf_counter()
        cloud = pc.build(d, result.confidence, intr, None, cfg.near_m, cfg.far_m)
        timings["cloud"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        payload = mesher.build_live(d, result.confidence, intr, cfg.near_m, cfg.far_m,
                                    edge_tolerance=cfg.edge_tolerance,
                                    min_conf=cfg.min_confidence)
        timings["mesh"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        v = laplacian_smooth(payload.vertices, payload.indices, cfg.mesh_smoothing_iters, 0.45)
        v, f = drop_degenerate(v, payload.indices)
        timings["opt"].append((time.perf_counter() - t0) * 1000)

        last = (result, cloud, v, f)

    result, cloud, v, f = last
    d = result.depth
    print(f"\nprofundidad: min={d.min():.2f} m  max={d.max():.2f} m  media={d.mean():.2f} m")
    print(f"nube de puntos: {cloud.points.shape[0]} puntos")
    print(f"malla: {v.shape[0]} vertices, {f.shape[0]} triangulos")
    assert v.shape[0] > 500, "la malla ha salido practicamente vacia"
    assert f.shape[0] > 500, "no se han generado triangulos"
    assert int(f.max()) < v.shape[0], "hay indices fuera de rango"

    # La malla debe tener extension real en los tres ejes.
    extent = v.max(axis=0) - v.min(axis=0)
    print(f"extension XYZ: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
    assert (extent > 0.2).all(), "la geometria es plana: algo va mal en la retroproyeccion"

    vd, fd = cluster_decimate(v, f, cfg.voxel_size * 2)
    print(f"decimada: {vd.shape[0]} vertices, {fd.shape[0]} triangulos "
          f"({100 * fd.shape[0] / max(f.shape[0], 1):.0f}% del original)")
    assert fd.shape[0] < f.shape[0], "la decimacion no ha reducido nada"

    # Protocolo: ida y vuelta con alineacion correcta.
    jpeg = encode_jpeg(render(640, 480))
    msg = protocol.encode_frame(jpeg, v, f, {"frame": 1, "w": pw, "h": ph,
                                             "src_w": 640, "src_h": 480,
                                             "vfov": intr.vfov_deg})
    head, blocks = protocol.decode(msg)
    assert head["vcount"] == v.shape[0] and head["tcount"] == f.shape[0]
    assert (4 + len(protocol.json.dumps(head, separators=(",", ":")).encode())) % 4 <= 3
    v_back = np.frombuffer(blocks[0], np.float32).reshape(-1, 3)
    assert np.allclose(v_back, v), "los vertices no sobreviven al protocolo"
    header_len = int(np.frombuffer(msg[:4], "<u4")[0])
    assert (4 + header_len) % 4 == 0, "los bloques binarios quedarian desalineados"
    print(f"protocolo: {len(msg) / 1024:.0f} kB por frame  "
          f"(cabecera alineada a {4 + header_len} bytes)")

    print("\ntiempos medios (ms):")
    for name, values in timings.items():
        print(f"  {name:<6} {np.mean(values):6.1f}")
    total = sum(np.mean(v) for v in timings.values())
    print(f"  {'TOTAL':<6} {total:6.1f}  ->  {1000 / total:.1f} fps teoricos")
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
