"""Servidor de Odversa Spatial Mesh.

Sirve tres cosas:
  * `/phone`  -> pagina que abre la camara del movil y la manda al PC.
  * `/`       -> visor de escritorio (la interfaz con los tres modos).
  * `/ws/...` -> los dos WebSockets, el del movil y el del visor.

Se arranca con HTTPS y certificado autofirmado porque los navegadores solo dan
acceso a la camara en contexto seguro; sin eso, el movil no dejaria abrirla.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import MAPS_DIR, WEB_DIR, RuntimeConfig, ServerConfig
from .core.types import ImuSample
from .modules import map_io
from .modules.mobile_camera_input import MobileCameraInput
from .modules.video_streaming import VideoStreaming
from .net import protocol
from .net.qr import qr_svg
from .net.redirect import start_redirect_server
from .pipeline import Pipeline

import numpy as np

runtime_cfg = RuntimeConfig()
camera = MobileCameraInput()
streaming = VideoStreaming()
pipeline = Pipeline(runtime_cfg, camera, streaming)


_cached_ip: str | None = None
#: puerto que el redirector http consiguio abrir de verdad (ver `lifespan`)
_http_port: int | None = None


def local_ip() -> str:
    """IP de la maquina en la red local (la que hay que teclear en el movil).

    Se cachea: la llamaban `/api/info`, `/api/qr.svg` y cada conexion del visor,
    y abrir un socket por peticion no aporta nada -la IP no cambia sola- y le da
    a macOS motivos de sobra para volver a preguntar por el permiso de red local.
    """
    global _cached_ip
    if _cached_ip is not None:
        return _cached_ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        _cached_ip = s.getsockname()[0]
    except Exception:
        _cached_ip = "127.0.0.1"
    finally:
        s.close()
    return _cached_ip


@asynccontextmanager
async def lifespan(app: FastAPI):
    streaming.bind_loop(asyncio.get_running_loop())
    pipeline.start()
    # Redirector http -> https, para que teclear la direccion sin esquema no
    # acabe en un "no se puede acceder a este sitio" sin explicacion.
    global _http_port
    port = int(os.environ.get("ODVERSA_PORT", 10000))
    redirect, _http_port = start_redirect_server(
        int(os.environ.get("ODVERSA_HTTP_PORT", 10001)), port)
    if _http_port:
        print(f"  Camara por http (redirige):  http://{local_ip()}:{_http_port}/phone")
    yield
    if redirect:
        redirect.shutdown()
    pipeline.stop()


app = FastAPI(title="Odversa Spatial Mesh", lifespan=lifespan)


# ---------------------------------------------------------------- paginas ---
def _asset_version() -> str:
    """Sello que cambia cuando cambia cualquier fichero de `web/`."""
    newest = 0.0
    for path in WEB_DIR.rglob("*"):
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
    return f"{int(newest)}"


#: Prefijo versionado de los estaticos.
#:
#: Los navegadores guardan los modulos de JavaScript en un mapa por URL y los
#: reutilizan aunque el servidor mande `no-store`, asi que editar un fichero y
#: recargar puede dejar mezclados modulos nuevos y viejos: errores que no
#: corresponden a ningun codigo que exista. Poniendo la version EN LA RUTA, cada
#: cambio produce URLs nuevas, y como los `import` relativos se resuelven contra
#: la URL del modulo, la version se propaga sola a todos sin tocar el JavaScript.
ASSET_PREFIX = f"/s{_asset_version()}"


def _page(path) -> Response:
    html = path.read_text(encoding="utf-8").replace("/static/", f"{ASSET_PREFIX}/")
    return Response(html, media_type="text/html; charset=utf-8",
                    headers={"Cache-Control": "no-store"})


@app.get("/")
async def viewer_page():
    return _page(WEB_DIR / "viewer" / "index.html")


@app.get("/phone")
async def phone_page():
    return _page(WEB_DIR / "phone" / "index.html")


def phone_url(request: Request) -> str:
    """URL https de captura, la que se muestra escrita.

    El puerto se toma de la cabecera Host, no de la configuracion: asi la URL
    es correcta aunque el servidor se haya arrancado en otro puerto.
    """
    port = request.headers.get("host", "").rpartition(":")[2] or "10000"
    scheme = request.url.scheme if request.url.scheme in ("http", "https") else "https"
    return f"{scheme}://{local_ip()}:{port}/phone"


def phone_url_http() -> str | None:
    """URL http de captura, que redirige a la https.

    Devuelve None si no se pudo abrir ningun puerto: mejor no anunciar una
    direccion que no existe.
    """
    if not _http_port:
        return None
    return f"http://{local_ip()}:{_http_port}/phone"


@app.get("/api/qr.svg")
async def qr_code(request: Request):
    """QR de la pagina de captura, para no teclear la IP en el movil.

    Codifica la direccion HTTPS del puerto principal, no la del redirector.

    El redirector http nacio para que teclear la direccion sin esquema no
    acabara en un error mudo, y para eso sirve. Pero su puerto es negociado: si
    esta ocupado se busca otro, asi que puede cambiar entre arranques. Un QR que
    apunta a un puerto movil es un QR que caduca -y eso es justo lo que pasaba.

    El puerto https, en cambio, es el que el usuario ha arrancado: fijo y
    siempre correcto. Ademas no se pierde nada: el salto por http terminaba
    igualmente en https, con el mismo aviso de certificado.
    """
    return Response(qr_svg(phone_url(request)),
                    media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@app.get("/api/info")
async def info():
    return JSONResponse({
        "ip": local_ip(),
        "config": runtime_cfg.as_dict(),
        "depth_backend": pipeline.depth.backend_name,
    })


@app.post("/api/save")
async def save_map():
    vertices, faces = pipeline.current_mesh()
    if vertices.shape[0] == 0:
        return JSONResponse({"ok": False, "error": "no hay malla que guardar"}, status_code=400)
    path = map_io.save_ply(vertices, faces)
    return JSONResponse({"ok": True, "path": str(path), "vertices": int(vertices.shape[0]),
                         "triangles": int(faces.shape[0])})


# ------------------------------------------------------------- websockets ---
@app.websocket("/ws/phone")
async def ws_phone(ws: WebSocket):
    """Entrada desde el movil: JPEG binarios + JSON de control/IMU."""
    await ws.accept()
    camera.set_connected(True)

    # Acuse de recibo: el pipeline avisa (desde su hilo) cada vez que consume un
    # frame, y el movil usa ese aviso para no ir por delante. Es lo que mantiene
    # la latencia pegada al tiempo de proceso en vez de crecer sin freno.
    loop = asyncio.get_running_loop()

    def ack() -> None:
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(_send_ack(ws)))

    camera.set_ack_callback(ack)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                # Cabecera de 8 bytes con el timestamp del movil (float64 LE).
                if len(data) > 8:
                    t_device = float(np.frombuffer(data[:8], dtype="<f8")[0])
                    camera.submit_jpeg(data[8:], t_device=t_device)
            elif (text := msg.get("text")) is not None:
                await _handle_phone_text(ws, text)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        camera.set_ack_callback(None)
        camera.set_connected(False)


async def _send_ack(ws: WebSocket) -> None:
    try:
        await ws.send_text('{"type":"ack"}')
    except Exception:
        pass  # el movil se ha ido; no es un error que deba propagarse


async def _handle_phone_text(ws: WebSocket, text: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return
    kind = payload.get("type")
    if kind == "hello":
        camera.set_device(payload.get("device"))
        intr = camera.set_source_info(
            int(payload.get("width", 640)), int(payload.get("height", 480)),
            payload.get("hfov_deg"),
        )
        await ws.send_text(protocol.encode_json("ready", {"intrinsics": intr.to_dict()}))
    elif kind == "imu":
        camera.submit_imu(ImuSample(
            t=float(payload.get("t", 0.0)),
            gyro=np.array(payload.get("gyro", [0, 0, 0]), np.float32),
            accel=np.array(payload.get("accel", [0, 0, 0]), np.float32),
            orientation=(np.array(payload["orientation"], np.float32)
                         if payload.get("orientation") else None),
        ))


@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    """Salida hacia el visor: frames (imagen+malla) y estadisticas."""
    await ws.accept()
    client = streaming.add_client(ws)
    pump = asyncio.create_task(streaming.pump(client))
    try:
        await ws.send_text(protocol.encode_json("hello", {
            "config": runtime_cfg.as_dict(),
            "ip": local_ip(),
            "http_url": phone_url_http(),
            "depth_backend": pipeline.depth.backend_name,
        }))
        while True:
            text = await ws.receive_text()
            await _handle_viewer_text(ws, text)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        pump.cancel()
        streaming.remove_client(client)


async def _handle_viewer_text(ws: WebSocket, text: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return
    kind = payload.get("type")
    if kind == "config":
        pipeline.update_config(payload.get("values", {}))
    elif kind == "search":
        pipeline.search.set_query(payload.get("query", ""))
    elif kind == "reset":
        pipeline.reset()
    elif kind == "save":
        vertices, faces = pipeline.current_mesh()
        if vertices.shape[0] == 0:
            await ws.send_text(protocol.encode_json("saved", {"ok": False,
                                                              "error": "malla vacia"}))
            return
        path = map_io.save_ply(vertices, faces)
        await ws.send_text(protocol.encode_json("saved", {
            "ok": True, "path": str(path), "triangles": int(faces.shape[0])}))


class NoCacheStatic(StaticFiles):
    """Estaticos sin cache.

    El visor y la pagina de captura se editan a menudo, y un modulo JavaScript
    servido desde la cache junto a otro recien cambiado produce errores que no
    corresponden a ningun codigo existente. En una herramienta local no hay nada
    que ganar cacheando.
    """

    def is_not_modified(self, *args, **kwargs) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app.mount(ASSET_PREFIX, NoCacheStatic(directory=str(WEB_DIR)), name="assets")
app.mount("/static", NoCacheStatic(directory=str(WEB_DIR)), name="static")
MAPS_DIR.mkdir(parents=True, exist_ok=True)
