"""Orquestador del pipeline de Odversa.

    frame del movil -> profundidad -> nube de puntos -> tracking
                    -> memoria espacial -> malla -> optimizacion -> difusion

Cada etapa es un modulo independiente; este fichero solo los encadena y mide.

Corre en DOS hilos encadenados, no en uno. El motivo esta medido: la red de
profundidad se lleva el 86% del tiempo (58,8 ms de 68) y todo lo demas suma
9,3 ms. En un solo hilo esos dos tiempos se suman; separandolos, mientras la
etapa de geometria trabaja con el frame N la de profundidad ya esta con el N+1,
y el ritmo pasa a ser el del cuello de botella en vez de la suma:

    un hilo:  58,8 + 9,3 = 68,1 ms  ->  14,7 fps
    dos hilos: max(58,8; 9,3) = 58,8 ms  ->  17,0 fps

El traspaso entre etapas es un unico hueco: si la geometria va con retraso, el
frame viejo se tira en vez de encolarse. Acumular cola solo sirve para ir cada
vez mas por detras de la mano que mueve el movil.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .config import REBUILD_KEYS, RuntimeConfig
from .core.camera import Intrinsics
from .core.types import PipelineStats, PointCloudChunk
from .modules.camera_tracking import CameraTracking
from .modules.depth_estimation import DepthEstimation
from .modules.mesh_optimization import drop_degenerate, enforce_budget, laplacian_smooth
from .modules.mesh_reconstruction import MeshReconstruction
from .modules.mobile_camera_input import MobileCameraInput
from .modules.point_cloud import PointCloud
from .modules.spatial_memory import SpatialMemory
from .modules.object_search import ObjectSearch
from .modules.surface_detection import SurfaceDetection
from .modules.video_streaming import VideoStreaming
from .net import protocol

MAX_TRIANGLES = 90000


def _euler_deg(T: np.ndarray) -> list[float]:
    """Yaw, pitch y roll en grados a partir de la pose (convenio OpenCV)."""
    R = T[:3, :3]
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        return [float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                float(np.degrees(np.arctan2(-R[2, 0], sy))),
                float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))]
    return [0.0, float(np.degrees(np.arctan2(-R[2, 0], sy))),
            float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))]


class _Slot:
    """Traspaso de un solo hueco entre las dos etapas.

    Si la etapa de geometria no ha recogido el frame anterior, el nuevo lo
    sustituye. Es deliberado: en un sistema que sigue el movimiento de la mano,
    un frame reciente vale mas que la garantia de procesarlos todos.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self.dropped = 0

    def put(self, item) -> None:
        with self._cond:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cond.notify()

    def get(self, timeout: float = 0.4):
        with self._cond:
            if self._item is None:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()


class Pipeline:
    def __init__(self, cfg: RuntimeConfig, camera: MobileCameraInput, streaming: VideoStreaming):
        self.cfg = cfg
        self.camera = camera
        self.streaming = streaming

        self.depth = DepthEstimation(cfg)
        self.tracking = CameraTracking(cfg)
        self.point_cloud = PointCloud()
        self.memory = SpatialMemory(cfg)
        self.mesh = MeshReconstruction()
        self.surfaces = SurfaceDetection()
        self.search = ObjectSearch()
        self._surface_list: list[dict] = []

        self.stats = PipelineStats(depth_backend=self.depth.backend_name)
        self._stats_lock = threading.Lock()
        self._cfg_lock = threading.Lock()
        self._pending_cfg: dict = {}

        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._handoff = _Slot()
        self._times: deque[float] = deque(maxlen=40)
        self._last_stats_sent = 0.0
        self._last_mesh: tuple[np.ndarray, np.ndarray] | None = None
        self._depth_ms = 0.0

    # -- ciclo de vida ------------------------------------------------------
    def start(self) -> None:
        if any(t.is_alive() for t in self._threads):
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._run_depth, name="odversa-depth", daemon=True),
            threading.Thread(target=self._run_geometry, name="odversa-geometry", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self.search.start()

    def stop(self) -> None:
        self._stop.set()
        self.search.stop()
        self.camera.wake()
        self._handoff.wake()
        for thread in self._threads:
            thread.join(timeout=2.0)

    # -- control ------------------------------------------------------------
    def update_config(self, updates: dict) -> None:
        with self._cfg_lock:
            self._pending_cfg.update(updates)

    def reset(self) -> None:
        self.depth.reset()
        self.tracking.reset()
        self.memory.reset()
        # El mallador conserva los vertices entre frames para que la malla no
        # tiemble; al reiniciar hay que olvidarlos tambien.
        self.mesh.live.delaunay_mesher.reset()
        self.surfaces.reset()
        self._last_mesh = None

    def _drain_config(self) -> None:
        with self._cfg_lock:
            if not self._pending_cfg:
                return
            updates, self._pending_cfg = self._pending_cfg, {}
        changed = self.cfg.apply(updates)
        if not changed:
            return
        if "depth_input_size" in changed or "depth_backend" in changed or "depth_model" in changed:
            self.depth.rebuild(self.cfg)
            with self._stats_lock:
                self.stats.depth_backend = self.depth.backend_name
        if "proc_width" in changed:
            self.mesh.live.delaunay_mesher.reset()
        if REBUILD_KEYS.intersection(changed):
            self.memory.reset()

    # -- etapa 1: profundidad -----------------------------------------------
    def _run_depth(self) -> None:
        """Hilo de inferencia. Es el cuello de botella, asi que no hace nada mas.

        La configuracion se aplica aqui porque este hilo es el unico que muta
        `cfg` y el unico que puede reconstruir el backend.
        """
        while not self._stop.is_set():
            frame = self.camera.next_frame(timeout=0.4)
            if frame is None:
                self._publish_stats(force=True)
                continue
            try:
                self._drain_config()
                self.search.submit(frame.bgr)
                intr, size = self._proc_intrinsics(frame)
                t0 = time.perf_counter()
                depth_result = self.depth.estimate(frame.bgr, size)
                self._depth_ms = (time.perf_counter() - t0) * 1000.0
                self._handoff.put((frame, depth_result, intr, size))
            except Exception as exc:  # un frame malo no puede tumbar el pipeline
                print(f"[pipeline] error estimando profundidad {frame.index}: {exc!r}")

    # -- etapa 2: geometria --------------------------------------------------
    def _run_geometry(self) -> None:
        """Hilo de refinado, malla y difusion. Solapa con la inferencia del siguiente."""
        while not self._stop.is_set():
            item = self._handoff.get(timeout=0.4)
            if item is None:
                continue
            frame = item[0]
            try:
                self._process(*item)
            except Exception as exc:
                print(f"[pipeline] error procesando frame {frame.index}: {exc!r}")

    def _proc_intrinsics(self, frame) -> tuple[Intrinsics, tuple[int, int]]:
        src = self.camera.intrinsics
        w, h = frame.size
        if src is None:
            src = Intrinsics.from_fov(w, h)
        pw = max(48, min(int(self.cfg.proc_width), 480))
        ph = max(36, int(round(pw * h / w)))
        return src.scaled(pw, ph), (pw, ph)

    def _process(self, frame, depth_result, intr: Intrinsics, size: tuple[int, int]) -> None:
        t_start = time.perf_counter()
        pw, ph = size
        depth_ms = self._depth_ms

        depth = self.point_cloud.refine_depth(depth_result.depth, frame.bgr, self.cfg)
        conf = depth_result.confidence

        # IA de superficies: detecta suelo/pared/mesa/techo por planos y pega la
        # profundidad al plano -> las superficies grandes dejan de temblar.
        if self.cfg.surface_ai:
            depth, self._surface_list = self.surfaces.process(
                depth, self.point_cloud.rays_for(intr),
                self.cfg.surface_snap, self.cfg.far_m * 2.4,
            )
        else:
            self._surface_list = []
        # Quietud final: EMA sobre la profundidad YA refinada y pegada a planos.
        depth = self.surfaces.settle(depth, self.cfg.far_m)

        # 2. Tracking: odometria RGB-D contra el frame anterior.
        #    La malla se sigue emitiendo en el sistema de la CAMARA, no en el del
        #    mundo: el visor la superpone sobre el video y transformarla aqui la
        #    despegaria de la imagen. La pose viaja aparte, para el panel y para
        #    la memoria espacial.
        t0 = time.perf_counter()
        track = self.tracking.track(frame, depth, intr)
        track_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Malla + optimizacion
        t0 = time.perf_counter()
        if self.cfg.mesh_enabled:
            payload = self.mesh.build_live(
                depth, conf, intr, self.cfg.near_m, self.cfg.far_m,
                edge_tolerance=self.cfg.edge_tolerance, min_conf=self.cfg.min_confidence,
                detail_cm=self.cfg.mesh_detail_cm,
                plane_mask=self.surfaces.plane_mask if self.cfg.surface_ai else None,
            )
            vertices, faces = payload.vertices, payload.indices
            if self.cfg.mesh_smoothing_iters > 0:
                vertices = laplacian_smooth(vertices, faces, self.cfg.mesh_smoothing_iters, 0.45)
            # `build_live` solo emite celdas de rejilla con las cuatro esquinas
            # validas, asi que no puede producir degenerados. Solo hay que
            # limpiar si la decimacion ha llegado a actuar.
            vertices, faces, lod = enforce_budget(vertices, faces, MAX_TRIANGLES, self.cfg.voxel_size)
            if lod > 0:
                vertices, faces = drop_degenerate(vertices, faces)
            faces = faces.astype(np.uint32, copy=False)
        else:
            vertices = np.zeros((0, 3), np.float32)
            faces = np.zeros((0, 3), np.uint32)
        mesh_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Nube de puntos + memoria espacial.
        #    Los puntos son los propios vertices de la malla: construir una nube
        #    aparte repetiria la retroproyeccion que el mallador ya ha hecho.
        t0 = time.perf_counter()
        cloud = PointCloudChunk(points=vertices, normals=None,
                                confidence=np.ones(vertices.shape[0], np.float32))
        self.memory.integrate(cloud, track.T_wc)
        integrate_ms = (time.perf_counter() - t0) * 1000.0

        self._last_mesh = (vertices, faces)

        # 4b. Busqueda de objetos: proyectar las cajas 2D sobre la malla.
        search = self.search.current()
        highlight = None
        if (search["query"] and search["boxes"] and payload.pixels is not None
                and vertices.shape[0] == payload.pixels.shape[0]):
            highlight = self._project_boxes(search["boxes"], payload.pixels,
                                            vertices, pw, ph)

        # 5. Difusion al visor
        header = {
            "frame": frame.index,
            "w": pw,
            "h": ph,
            "src_w": frame.size[0],
            "src_h": frame.size[1],
            "vfov": intr.vfov_deg,
            "intr": intr.to_dict(),
            "t_recv": frame.t_recv,
            "latency_ms": round(frame.latency_ms, 1),
            "tracking": track.state.value,
            "search": {"query": search["query"], "status": search["status"],
                        "boxes": search["boxes"], "ms": search["ms"]},
        }
        jpeg = frame.jpeg if self.cfg.relay_video else None
        self.streaming.broadcast(
            protocol.encode_frame(jpeg, vertices, faces, header, highlight=highlight))

        # 6. Metricas
        geometry_ms = (time.perf_counter() - t_start) * 1000.0
        self._times.append(time.time())
        with self._stats_lock:
            s = self.stats
            s.fps_capture = self.camera.capture_fps
            s.fps_pipeline = self._pipeline_fps()
            # Latencia real de extremo a extremo: aunque las dos etapas se
            # solapen entre frames distintos, UN frame concreto sigue pasando por
            # las dos en serie. Sumar solo la etapa de geometria daria un numero
            # bonito y falso.
            s.latency_ms = frame.latency_ms + depth_ms + geometry_ms
            s.depth_ms = depth_ms
            s.track_ms = track_ms
            s.integrate_ms = integrate_ms
            s.mesh_ms = mesh_ms
            s.points_total = int(cloud.points.shape[0])
            s.triangles = int(faces.shape[0])
            s.chunks = self.memory.chunk_count
            s.voxel_size = self.cfg.voxel_size
            s.tracking = track.state.value
            s.inliers = track.inliers
            s.frames = frame.index
            s.connected = self.camera.connected
            s.updated = time.time()
        self._publish_stats()

    @staticmethod
    def _project_boxes(boxes: list[dict], pixels: np.ndarray, vertices: np.ndarray,
                       pw: int, ph: int) -> np.ndarray:
        """Marca (0/255) los vertices que caen dentro de alguna caja detectada.

        La caja 2D sola no basta: encierra tambien la pared de detras del
        objeto. Por eso se estima la profundidad del objeto con la mediana del
        nucleo central de la caja y solo se resalta lo que esta a esa distancia:
        el objeto se separa de su fondo.
        """
        highlight = np.zeros(vertices.shape[0], np.uint8)
        px = pixels[:, 0].astype(np.float32)
        py = pixels[:, 1].astype(np.float32)
        z = vertices[:, 2]
        for box in boxes:
            x0, x1 = box["x0"] * pw, box["x1"] * pw
            y0, y1 = box["y0"] * ph, box["y1"] * ph
            inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            if not inside.any():
                continue
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            core = inside & (np.abs(px - cx) <= (x1 - x0) * 0.3) \
                          & (np.abs(py - cy) <= (y1 - y0) * 0.3)
            z_obj = float(np.median(z[core])) if core.any() else float(np.median(z[inside]))
            highlight[inside & (z > z_obj * 0.62) & (z < z_obj * 1.45)] = 255
        return highlight

    def _pipeline_fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 1e-6 else 0.0

    def _publish_stats(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_stats_sent < 0.25:
            return
        self._last_stats_sent = now
        with self._stats_lock:
            data = self.stats.to_dict()
        data["connected"] = self.camera.connected
        data["viewers"] = self.streaming.viewer_count
        data["dropped"] = self.camera.dropped
        data["device"] = self.camera.device
        data["surfaces"] = self._surface_list
        # Pose tal y como la conoce el tracker, sin adornos: si el estado es
        # WEAK o LOST el visor lo dice en vez de fingir una posicion firme.
        T = self.tracking.T_wc
        data["position"] = [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])]
        data["rotation"] = _euler_deg(T)
        self.streaming.broadcast(
            protocol.encode_json("stats", {"stats": data, "config": self.cfg.as_dict()})
        )

    # -- utilidades ---------------------------------------------------------
    def current_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self._last_mesh is None:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
        return self._last_mesh
