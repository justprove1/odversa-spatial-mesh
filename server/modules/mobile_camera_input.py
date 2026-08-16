"""MobileCameraInput - entrada de la camara del movil.

Responsabilidad: recibir frames y muestras inerciales de *una* fuente, sea cual
sea el transporte, decodificarlos y dejarlos en una cola de un solo hueco (el
pipeline siempre trabaja con el frame mas reciente; los atrasados se tiran, que
es lo correcto cuando lo que importa es la latencia).

El transporte concreto (WebSocket sobre Wi-Fi hoy; WebRTC, USB/UVC o un sensor
con LiDAR manana) se inyecta desde fuera llamando a `submit_*`. Esta clase no
sabe nada de red.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from ..core.camera import DEFAULT_HFOV_DEG, Intrinsics
from ..core.types import Frame, ImuSample


class MobileCameraInput:
    def __init__(self, imu_history: int = 240):
        # Aviso al movil de que un frame ya se ha consumido. Sirve para que
        # module el ritmo de envio: ver `set_ack_callback`.
        self._ack: object | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: Frame | None = None
        self._imu: deque[ImuSample] = deque(maxlen=imu_history)
        self._frame_index = 0
        self._connected = False
        self._intrinsics: Intrinsics | None = None
        self._hfov_deg = DEFAULT_HFOV_DEG
        # Offset entre el reloj del movil y el del servidor (para la latencia).
        self._clock_offset: float | None = None
        self._recv_times: deque[float] = deque(maxlen=60)
        self._dropped = 0
        self.device = ""

    # -- estado -------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected

    def set_connected(self, value: bool) -> None:
        with self._cond:
            self._connected = value
            if not value:
                self._pending = None
            self._cond.notify_all()

    def set_device(self, user_agent: str | None) -> None:
        """Cadena de usuario del movil, para mostrarla en el visor."""
        self.device = (user_agent or "").strip()

    def set_source_info(self, width: int, height: int, hfov_deg: float | None = None) -> Intrinsics:
        """El movil anuncia resolucion y (si puede) el FOV real de la lente."""
        if hfov_deg and 20.0 < hfov_deg < 160.0:
            self._hfov_deg = float(hfov_deg)
        self._intrinsics = Intrinsics.from_fov(width, height, self._hfov_deg)
        return self._intrinsics

    @property
    def intrinsics(self) -> Intrinsics | None:
        return self._intrinsics

    @property
    def capture_fps(self) -> float:
        if len(self._recv_times) < 2:
            return 0.0
        span = self._recv_times[-1] - self._recv_times[0]
        return (len(self._recv_times) - 1) / span if span > 1e-6 else 0.0

    @property
    def dropped(self) -> int:
        return self._dropped

    # -- entrada ------------------------------------------------------------
    def submit_imu(self, sample: ImuSample) -> None:
        with self._cond:
            self._imu.append(sample)

    def submit_jpeg(self, data: bytes, t_device: float | None = None) -> bool:
        """Decodifica un JPEG del movil y lo deja como frame pendiente."""
        buf = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return False

        now = time.time()
        self._recv_times.append(now)
        if t_device is not None:
            # Estimacion conservadora del desfase de relojes: nos quedamos con el
            # minimo observado, que corresponde al frame de menor latencia.
            offset = now - t_device
            self._clock_offset = offset if self._clock_offset is None else min(self._clock_offset, offset)
            latency_ms = max(0.0, (offset - self._clock_offset)) * 1000.0
        else:
            latency_ms = 0.0

        h, w = bgr.shape[:2]
        if self._intrinsics is None or self._intrinsics.width != w or self._intrinsics.height != h:
            self.set_source_info(w, h)

        with self._cond:
            # Si el hueco ya estaba ocupado, ese frame muere aqui. Hay que
            # acusarlo igualmente: el movil lleva la cuenta de frames en vuelo y
            # un acuse que no llega le hace creer que el servidor sigue ocupado,
            # dejando de enviar hasta atascarse.
            replaced = self._pending is not None
            if replaced:
                self._dropped += 1
            self._frame_index += 1
            self._pending = Frame(
                index=self._frame_index,
                t_device=t_device if t_device is not None else now,
                t_recv=now,
                bgr=bgr,
                jpeg=data,
                imu=self._latest_imu_locked(),
                latency_ms=latency_ms,
            )
            self._connected = True
            self._cond.notify_all()
        if replaced and self._ack is not None:
            self._ack()  # fuera del lock: el envio es asincrono
        return True

    def _latest_imu_locked(self) -> ImuSample | None:
        return self._imu[-1] if self._imu else None

    def imu_between(self, t0: float, t1: float) -> list[ImuSample]:
        """Muestras inerciales en un intervalo (reloj del movil)."""
        with self._cond:
            return [s for s in self._imu if t0 < s.t <= t1]

    # -- salida -------------------------------------------------------------
    def set_ack_callback(self, fn) -> None:
        """Funcion que avisa al movil de que puede mandar el siguiente frame.

        Sin esto, el movil emite a su ritmo y los frames se acumulan en el buffer
        del socket: el sistema sigue dando los mismos fps pero cada imagen llega
        con varios frames de retraso, y al mover el movil se nota muchisimo.
        """
        self._ack = fn

    def next_frame(self, timeout: float = 0.5) -> Frame | None:
        """Bloquea hasta que haya un frame nuevo. Devuelve None si expira."""
        with self._cond:
            if self._pending is None:
                self._cond.wait(timeout)
            frame, self._pending = self._pending, None
        if frame is not None and self._ack is not None:
            self._ack()
        return frame

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()
