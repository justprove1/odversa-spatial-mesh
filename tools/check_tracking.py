"""Prueba de la odometria contra un recorrido de pose CONOCIDA.

No hay forma de validar un tracker mirando el visor: una trayectoria bonita
puede estar equivocada de medio metro. Aqui se construye una habitacion
sintetica, se mueve la camara por un camino que sabemos exactamente, y se
compara lo que estima `CameraTracking` con la verdad.

El truco para tener imagen Y profundidad coherentes en cada pose es no dibujar
nada: se retroproyecta la vista inicial a una nube de puntos con color, y cada
vista posterior es esa nube reproyectada con la pose de turno. Asi la geometria
y la textura se mueven juntas, que es justo lo que ve una camara real.

    .venv/bin/python -m tools.check_tracking
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from server.config import RuntimeConfig
from server.core.camera import Intrinsics, pixel_rays
from server.core.types import Frame, ImuSample, TrackingState
from server.modules.camera_tracking import CameraTracking

FPS = 17.0
SUPERSAMPLE = 2  # puntos por pixel al construir la nube: menos agujeros


# -- escena -----------------------------------------------------------------
def room_depth(intr: Intrinsics) -> np.ndarray:
    """Habitacion: suelo, techo, dos paredes y el fondo. Profundidad exacta."""
    rays = pixel_rays(intr).astype(np.float64)
    rx, ry = rays[..., 0], rays[..., 1]
    z = np.full(rx.shape, 6.0)  # pared del fondo a 6 m

    def closer(candidate: np.ndarray, valid: np.ndarray) -> None:
        take = valid & (candidate > 0.1) & (candidate < z)
        z[take] = candidate[take]

    with np.errstate(divide="ignore", invalid="ignore"):
        closer(1.40 / ry, ry > 1e-6)    # suelo, camara a 1,40 m
        closer(-1.10 / ry, ry < -1e-6)  # techo
        closer(-2.50 / rx, rx < -1e-6)  # pared izquierda
        closer(2.50 / rx, rx > 1e-6)    # pared derecha
    return z.astype(np.float32)


def texture(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Textura con esquinas de verdad: sin ellas no hay flujo optico que valga."""
    rng = np.random.default_rng(seed)
    small = rng.integers(20, 235, (h // 4 + 2, w // 4 + 2, 3), np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    grain = rng.integers(0, 26, (h, w, 1), np.uint8)
    return cv2.add(img, np.repeat(grain, 3, axis=2))


def build_cloud(intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Nube de puntos con color de la vista inicial (que define el mundo)."""
    hi = intr.scaled(intr.width * SUPERSAMPLE, intr.height * SUPERSAMPLE)
    depth = room_depth(hi)
    colors = texture(hi.height, hi.width)
    points = pixel_rays(hi) * depth[..., None]
    return points.reshape(-1, 3).astype(np.float64), colors.reshape(-1, 3)


def render(points: np.ndarray, colors: np.ndarray, T_wc: np.ndarray,
           intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Vista desde `T_wc`: pinta la nube con z-buffer y tapa los agujeros."""
    T_cw = np.linalg.inv(T_wc)
    cam = points @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = cam[:, 2]
    front = z > 0.05
    cam, z, col = cam[front], z[front], colors[front]

    u = np.round(cam[:, 0] / z * intr.fx + intr.cx).astype(np.int64)
    v = np.round(cam[:, 1] / z * intr.fy + intr.cy).astype(np.int64)
    inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    u, v, z, col = u[inside], v[inside], z[inside], col[inside]

    img = np.zeros((intr.height, intr.width, 3), np.uint8)
    depth = np.zeros((intr.height, intr.width), np.float32)
    order = np.argsort(-z)  # pintor: lo cercano se escribe al final y gana
    img[v[order], u[order]] = col[order]
    depth[v[order], u[order]] = z[order].astype(np.float32)

    holes = (depth <= 0).astype(np.uint8)
    if holes.any():
        far = np.where(holes.astype(bool), 1e6, depth)
        depth = np.where(holes.astype(bool), cv2.erode(far, np.ones((3, 3), np.float32)), depth)
        depth[depth > 1e5] = 6.0
        img = cv2.inpaint(img, holes, 2, cv2.INPAINT_TELEA)
    return img, depth


# -- recorrido ---------------------------------------------------------------
def pose(step: int) -> np.ndarray:
    """Verdad de campo: desplazamiento lateral y hacia delante, girando a la vez."""
    T = np.eye(4)
    yaw = np.radians(0.6 * step)
    T[:3, :3] = cv2.Rodrigues(np.array([0.0, yaw, 0.0]))[0]
    T[:3, 3] = [0.020 * step, 0.004 * step, 0.014 * step]
    return T


def imu_for(step: int, dt: float) -> ImuSample:
    """Giroscopio coherente con el recorrido (ejes del dispositivo)."""
    rate = np.radians(0.6) / dt  # yaw del mundo -> giro del dispositivo
    return ImuSample(t=step * dt, gyro=np.array([0.0, -rate, 0.0], np.float32))


def run(tracker: CameraTracking, frames: list[tuple[np.ndarray, np.ndarray]],
        intr: Intrinsics, with_imu: bool = False) -> tuple[list[np.ndarray], list, float]:
    dt = 1.0 / FPS
    poses, results, elapsed = [], [], 0.0
    for i, (img, depth) in enumerate(frames):
        frame = Frame(index=i, t_device=i * dt, t_recv=i * dt, bgr=img,
                      imu=imu_for(i, dt) if with_imu else None)
        t0 = time.perf_counter()
        res = tracker.track(frame, depth, intr)
        elapsed += (time.perf_counter() - t0) * 1000.0
        poses.append(res.T_wc.copy())
        results.append(res)
    return poses, results, elapsed / max(len(frames), 1)


def main() -> int:
    cfg = RuntimeConfig()
    src = Intrinsics.from_fov(640, 360)
    pw = cfg.proc_width
    intr = src.scaled(pw, int(round(pw * 360 / 640)))
    print(f"rejilla {intr.width}x{intr.height}  vfov={intr.vfov_deg:.1f} grados")

    points, colors = build_cloud(intr)
    steps = 30
    truth = [pose(i) for i in range(steps)]
    frames = [render(points, colors, T, intr) for T in truth]
    print(f"escena: {points.shape[0]} puntos, {steps} frames sinteticos")

    # 1. Recorrido conocido -------------------------------------------------
    tracker = CameraTracking(cfg)
    poses, results, ms = run(tracker, frames, intr)

    end_err = float(np.linalg.norm(poses[-1][:3, 3] - truth[-1][:3, 3]))
    path = float(np.linalg.norm(truth[-1][:3, 3]))
    R_err = poses[-1][:3, :3].T @ truth[-1][:3, :3]
    yaw_err = abs(np.degrees(np.arctan2(R_err[0, 2], R_err[2, 2])))
    good = sum(1 for r in results if r.state is TrackingState.GOOD)
    inliers = int(np.median([r.inliers for r in results if r.inliers]))

    print(f"\nrecorrido real:   {path:.3f} m,  giro {0.6 * (steps - 1):.1f} grados")
    print(f"estimado:         {np.linalg.norm(poses[-1][:3, 3]):.3f} m")
    print(f"error final:      {end_err:.3f} m  ({100 * end_err / path:.1f}% del recorrido)")
    print(f"error de giro:    {yaw_err:.2f} grados")
    print(f"estado GOOD en    {good}/{steps} frames,  inliers (mediana) {inliers}")
    print(f"coste:            {ms:.2f} ms por frame")

    assert good >= steps - 3, "el tracker no consigue engancharse a la escena"
    assert end_err < 0.10 * path + 0.02, "la trayectoria se desvia demasiado"
    assert yaw_err < 3.0, "la rotacion acumulada no sigue al recorrido"
    assert ms < 12.0, "el tracking se come el presupuesto del frame"

    # 2. Camara quieta: no puede inventar movimiento ------------------------
    still = [frames[0] for _ in range(12)]
    tracker = CameraTracking(cfg)
    poses_still, _, _ = run(tracker, still, intr)
    drift = float(np.linalg.norm(poses_still[-1][:3, 3]))
    print(f"\ncamara quieta:    deriva {drift * 100:.2f} cm en 12 frames")
    assert drift < 0.02, "la pose se mueve con la camara parada"

    # 3. Escena sin textura: debe degradar, no mentir ni romperse ------------
    flat = (np.full_like(frames[0][0], 30), np.full_like(frames[0][1], 3.0))
    tracker = CameraTracking(cfg)
    _, results_flat, _ = run(tracker, [flat] * 10, intr)
    states = {r.state.value for r in results_flat}
    print(f"pared lisa:       estados {sorted(states)}")
    assert TrackingState.GOOD.value not in states, "dice GOOD sobre una pared sin textura"

    # 4. IMU como respaldo: con la escena inservible, la rotacion sobrevive --
    tracker = CameraTracking(cfg)
    _, results_imu, _ = run(tracker, [flat] * 10, intr, with_imu=True)
    used = sum(1 for r in results_imu if r.used_imu)
    turned = sum(r.delta_r_deg for r in results_imu)
    print(f"respaldo IMU:     {used}/10 frames por giroscopio, {turned:.1f} grados recuperados")
    assert used >= 8 and turned > 2.0, "el giroscopio no esta entrando como respaldo"

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
