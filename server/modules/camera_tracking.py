"""CameraTracking - pose 6DoF de la camara (odometria RGB-D + IMU).

Fase 2. La camara se localiza contra el frame anterior, no contra un mapa: es
odometria, no SLAM. No hay cierre de bucle, asi que el error se acumula; a
cambio cuesta ~2 ms y no necesita ninguna red mas.

La cadena, en el orden en que corre:

  1. `goodFeaturesToTrack` sobre el frame anterior, enmascarado por profundidad
     valida. No se redetecta cada frame: los puntos se arrastran con el flujo
     mientras sobrevivan, y solo se rellena cuando quedan pocos. Ademas de
     ahorrar, evita que la pose salte cada vez que cambia el juego de esquinas.
  2. Flujo optico Lucas-Kanade piramidal hacia el frame actual, con
     verificacion inversa: se vuelve del punto encontrado al de partida y se
     descarta lo que no regresa a menos de un pixel. Es el filtro que quita las
     correspondencias que se van por un borde de oclusion.
  3. `solvePnPRansac` entre los puntos 3D del frame anterior (retroproyectados
     con SU profundidad) y sus proyecciones 2D en el actual. Sale `T_cp`, la
     transformacion que lleva del sistema del frame anterior al actual, y de ahi
     `T_wc <- T_wc @ inv(T_cp)`.
  4. El giroscopio del movil entra por dos sitios: como semilla de rotacion para
     RANSAC (`useExtrinsicGuess`) y como respaldo cuando la escena se queda sin
     textura y el PnP no converge. Con IMU la rotacion aguanta; la traslacion no,
     porque integrar el acelerometro dos veces se va en segundos.
  5. Validacion de movimiento (`max_translation_per_frame_m`,
     `max_rotation_per_frame_deg`) y estados GOOD / WEAK / LOST.

Sobre la ESCALA: la profundidad es monocular, calibrada por los anclajes
`near_m`/`far_m`. La traslacion que sale de aqui hereda esa escala, asi que es
consistente consigo misma pero no es metrica absoluta. Cambiar `near_m` a mitad
de recorrido cambia el tamano del mundo, y por eso la trayectoria se reinicia.

Trabaja a la resolucion de proceso (la de la profundidad), que es la que le
corresponde a `intr`. Subir de ahi no mejoraria la pose: el limite lo pone el
ruido de la profundidad, no el de los pixeles.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import RuntimeConfig
from ..core.camera import Intrinsics
from ..core.types import Frame, TrackingResult, TrackingState, identity_pose, rotation_angle_deg

# Verificacion inversa: cuanto puede alejarse el punto al volver, en pixeles.
BACKWARD_TOLERANCE_PX = 1.0
# Error de reproyeccion que RANSAC considera inlier, en pixeles.
REPROJECTION_PX = 2.0
# Fraccion del objetivo de esquinas por debajo de la cual se redetecta.
REFILL_RATIO = 0.55
# Frames seguidos sin pose valida antes de declarar LOST.
LOST_AFTER = 6
# Suavizado del incremento de pose. Bajo a proposito: filtra el temblor de la
# profundidad sin llegar a arrastrar la pose por detras de la mano.
DELTA_EMA = 0.35

# Ejes del dispositivo (x derecha, y arriba, z hacia el usuario) a ejes de
# camara OpenCV (x derecha, y abajo, z hacia la escena).
DEVICE_TO_CAMERA = np.diag([1.0, -1.0, -1.0])

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.03),
)


def _pose_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)
    return T


class CameraTracking:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.T_wc = identity_pose()
        self.state = TrackingState.INIT
        self._frames = 0
        self._reset_reference()

    def _reset_reference(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_depth: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None  # Nx1x2 float32
        self._prev_t: float | None = None
        self._intr_key: tuple | None = None
        self._lost_streak = 0
        self._delta_smooth: np.ndarray | None = None

    def reset(self) -> None:
        self.T_wc = identity_pose()
        self.state = TrackingState.INIT
        self._frames = 0
        self._reset_reference()

    # -- ciclo ---------------------------------------------------------------
    def track(self, frame: Frame, depth: np.ndarray, intr: Intrinsics) -> TrackingResult:
        self._frames += 1
        gray = self._to_process_gray(frame.bgr, intr)
        imu_delta = self._imu_rotation(frame)

        # Un cambio de intrinsecos (el deslizador de densidad) invalida las
        # referencias: los pixeles anteriores ya no significan lo mismo.
        key = (intr.width, intr.height, round(intr.fx, 4))
        if key != self._intr_key:
            self._intr_key = key
            self._prev_gray = None
            self._prev_points = None

        if self._prev_gray is None or self._prev_depth is None:
            self._store_reference(gray, depth, frame)
            self.state = TrackingState.INIT
            return TrackingResult(T_wc=self.T_wc, state=self.state)

        p0 = self._reference_corners(depth_prev=self._prev_depth)
        tracked = 0
        survivors: np.ndarray | None = None
        T_cp: np.ndarray | None = None
        inliers = 0

        if p0 is not None:
            p0, p1 = self._flow(self._prev_gray, gray, p0)
            tracked = int(p0.shape[0])
            if tracked >= 6:
                T_cp, inliers = self._solve(p0, p1, intr, imu_delta)
                survivors = p1

        used_imu = False
        if T_cp is None or inliers < 6:
            # Sin geometria fiable: la rotacion del giroscopio es mejor que nada
            # y no inventa traslacion, que es el error que se nota.
            if imu_delta is not None:
                T_cp, used_imu = imu_delta, True
            else:
                T_cp = None

        state = self._classify(T_cp, inliers, used_imu)
        delta_t = delta_r = 0.0

        if T_cp is not None and state is not TrackingState.LOST:
            T_cp = self._smooth(T_cp)
            delta_t = float(np.linalg.norm(T_cp[:3, 3]))
            delta_r = rotation_angle_deg(T_cp[:3, :3])
            # `T_cp` va del frame anterior al actual; la pose acumulada va al
            # reves, de ahi la inversa.
            self.T_wc = self.T_wc @ np.linalg.inv(T_cp)

        self.state = state
        # Los puntos que sobreviven se arrastran al siguiente frame, ya en sus
        # coordenadas nuevas. Si la pose no fue buena se sueltan: seguir tirando
        # de un juego de esquinas dudoso solo propaga el error.
        self._store_reference(gray, depth, frame,
                              points=survivors if state is TrackingState.GOOD else None)

        return TrackingResult(T_wc=self.T_wc, state=state, inliers=inliers, tracked=tracked,
                              delta_t_m=delta_t, delta_r_deg=delta_r, used_imu=used_imu)

    # -- etapas --------------------------------------------------------------
    def _to_process_gray(self, bgr: np.ndarray, intr: Intrinsics) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != intr.width or gray.shape[0] != intr.height:
            gray = cv2.resize(gray, (intr.width, intr.height), interpolation=cv2.INTER_AREA)
        return gray

    def _valid_mask(self, depth: np.ndarray) -> np.ndarray:
        """Pixeles con profundidad utilizable, encogidos para huir de los bordes.

        El erosionado no es cosmetico: una esquina justo en un salto de
        profundidad recibe la distancia del fondo o la del objeto segun el
        redondeo, y ese punto envenena el PnP.
        """
        far = self.cfg.far_m * 2.5
        mask = np.isfinite(depth) & (depth > max(0.15, self.cfg.near_m * 0.25)) & (depth < far)
        return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))

    def _reference_corners(self, depth_prev: np.ndarray) -> np.ndarray | None:
        """Esquinas del frame anterior: las que sobreviven, mas relleno si faltan."""
        target = max(60, int(self.cfg.track_max_features))
        have = 0 if self._prev_points is None else int(self._prev_points.shape[0])
        if have >= target * REFILL_RATIO:
            return self._prev_points

        mask = self._valid_mask(depth_prev)
        h, w = depth_prev.shape[:2]
        fresh = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=target, qualityLevel=0.01,
            minDistance=max(3.0, w / 90.0), mask=mask, blockSize=5,
        )
        if fresh is None:
            return self._prev_points
        if self._prev_points is None or have == 0:
            return fresh.astype(np.float32)
        # Se anaden a las que ya se seguian, sin duplicar las que caen encima.
        keep = np.ones(fresh.shape[0], bool)
        old = self._prev_points.reshape(-1, 2)
        for i, pt in enumerate(fresh.reshape(-1, 2)):
            if np.min(np.abs(old - pt).max(axis=1)) < 3.0:
                keep[i] = False
        return np.concatenate([self._prev_points, fresh[keep]], axis=0).astype(np.float32)

    def _flow(self, prev_gray: np.ndarray, gray: np.ndarray,
              p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
        if p1 is None:
            return p0[:0], p0[:0]
        p0r, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK_PARAMS)
        if p0r is None:
            return p0[:0], p0[:0]
        err = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
        h, w = gray.shape[:2]
        inside = ((p1.reshape(-1, 2) >= 1.0) & (p1.reshape(-1, 2) < [w - 1, h - 1])).all(axis=1)
        ok = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1) & (err < BACKWARD_TOLERANCE_PX) & inside
        return p0[ok], p1[ok]

    def _solve(self, p0: np.ndarray, p1: np.ndarray, intr: Intrinsics,
               imu_delta: np.ndarray | None) -> tuple[np.ndarray | None, int]:
        """PnP entre los 3D del frame anterior y sus pixeles en el actual."""
        depth_prev = self._prev_depth
        assert depth_prev is not None
        uv = p0.reshape(-1, 2)
        cols = np.clip(np.round(uv[:, 0]).astype(int), 0, depth_prev.shape[1] - 1)
        rows = np.clip(np.round(uv[:, 1]).astype(int), 0, depth_prev.shape[0] - 1)
        z = depth_prev[rows, cols].astype(np.float64)

        far = self.cfg.far_m * 2.5
        good = np.isfinite(z) & (z > max(0.15, self.cfg.near_m * 0.25)) & (z < far)
        if int(good.sum()) < 6:
            return None, 0

        uv, z = uv[good].astype(np.float64), z[good]
        obj = np.stack([(uv[:, 0] - intr.cx) / intr.fx * z,
                        (uv[:, 1] - intr.cy) / intr.fy * z, z], axis=1)
        img = p1.reshape(-1, 2)[good].astype(np.float64)

        guess = imu_delta is not None
        rvec = cv2.Rodrigues(imu_delta[:3, :3])[0] if guess else np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, intr.K, None, rvec=rvec, tvec=tvec, useExtrinsicGuess=guess,
                iterationsCount=100, reprojectionError=REPROJECTION_PX, confidence=0.995,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None, 0
        if not ok or inl is None:
            return None, 0

        count = int(inl.shape[0])
        if count >= 6:
            # Refinado con solo los inliers: RANSAC da el consenso, no el mejor
            # ajuste. Este paso quita la mayor parte del temblor de la pose.
            sel = inl.reshape(-1)
            ok2, rvec, tvec = cv2.solvePnP(obj[sel], img[sel], intr.K, None, rvec=rvec,
                                           tvec=tvec, useExtrinsicGuess=True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok2:
                return None, 0
        return _pose_from_rt(rvec, tvec), count

    # -- fusion y estados ----------------------------------------------------
    def _imu_rotation(self, frame: Frame) -> np.ndarray | None:
        """Rotacion entre frames integrando el giroscopio, en ejes de camara."""
        if not self.cfg.track_use_imu or frame.imu is None or self._prev_t is None:
            return None
        dt = float(frame.t_device - self._prev_t)
        if not (1e-3 < dt < 0.5):
            return None
        gyro = np.asarray(frame.imu.gyro, dtype=np.float64).reshape(3)
        if not np.isfinite(gyro).all() or float(np.abs(gyro).max()) < 1e-3:
            return None
        # El giroscopio mide como gira el DISPOSITIVO; la escena gira al reves.
        omega = DEVICE_TO_CAMERA @ gyro
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues((-omega * dt).reshape(3, 1))[0]
        return T

    def _classify(self, T_cp: np.ndarray | None, inliers: int, used_imu: bool) -> TrackingState:
        if T_cp is None:
            self._lost_streak += 1
            return TrackingState.LOST if self._lost_streak >= LOST_AFTER else TrackingState.WEAK

        # Un salto imposible entre dos frames consecutivos no es movimiento: es
        # una correspondencia mala que ha colado. Se descarta el incremento
        # entero antes que meterlo en la pose acumulada.
        if (np.linalg.norm(T_cp[:3, 3]) > self.cfg.max_translation_per_frame_m
                or rotation_angle_deg(T_cp[:3, :3]) > self.cfg.max_rotation_per_frame_deg):
            self._lost_streak += 1
            return TrackingState.LOST if self._lost_streak >= LOST_AFTER else TrackingState.WEAK

        self._lost_streak = 0
        if used_imu or inliers < int(self.cfg.track_min_inliers):
            return TrackingState.WEAK
        return TrackingState.GOOD

    def _smooth(self, T_cp: np.ndarray) -> np.ndarray:
        """EMA sobre el incremento (traslacion y eje-angulo), no sobre la pose.

        Suavizar la pose acumulada la frenaria y dejaria un error permanente;
        suavizar el incremento solo recorta el ruido de cada paso.
        """
        vec = np.concatenate([T_cp[:3, 3], cv2.Rodrigues(T_cp[:3, :3])[0].reshape(3)])
        if self._delta_smooth is None:
            self._delta_smooth = vec
        else:
            self._delta_smooth = DELTA_EMA * self._delta_smooth + (1.0 - DELTA_EMA) * vec
        return _pose_from_rt(self._delta_smooth[3:].reshape(3, 1), self._delta_smooth[:3])

    def _store_reference(self, gray: np.ndarray, depth: np.ndarray, frame: Frame,
                         points: np.ndarray | None = None) -> None:
        self._prev_gray = gray
        self._prev_depth = depth
        self._prev_t = float(frame.t_device)
        self._prev_points = points if points is not None and points.shape[0] > 0 else None
