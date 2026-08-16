"""VideoStreaming - transporte y difusion.

Dos direcciones:
  * movil -> servidor: hoy WebSocket binario sobre la red local (Wi-Fi o el
    cable USB con "compartir conexion", que en la practica es la misma ruta IP
    con menos latencia). La clase no depende del transporte: cualquier fuente
    que llame a `MobileCameraInput.submit_jpeg` encaja igual, incluido un futuro
    WebRTC o una camara UVC por USB.
  * servidor -> visor(es): difusion del mensaje `frame` con imagen y malla.

Regla de latencia: si un cliente no da abasto, se le tiran los mensajes viejos
en vez de acumularlos. Mas vale saltar un frame que ir tres segundos por detras.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


# `eq=False` conserva la identidad por objeto: los clientes se guardan en un
# set, y el __eq__ que genera dataclass por defecto anula __hash__.
@dataclass(eq=False)
class ViewerClient:
    ws: object
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=2))
    dropped: int = 0


class VideoStreaming:
    """Hub de difusion, seguro de usar desde el hilo del pipeline."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[ViewerClient] = set()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # -- clientes -----------------------------------------------------------
    def add_client(self, ws) -> ViewerClient:
        client = ViewerClient(ws=ws)
        self._clients.add(client)
        return client

    def remove_client(self, client: ViewerClient) -> None:
        self._clients.discard(client)

    @property
    def viewer_count(self) -> int:
        return len(self._clients)

    # -- envio --------------------------------------------------------------
    def broadcast(self, data: bytes | str) -> None:
        """Llamable desde cualquier hilo."""
        loop = self._loop
        if loop is None or not self._clients:
            return
        loop.call_soon_threadsafe(self._enqueue, data)

    def _enqueue(self, data: bytes | str) -> None:
        for client in list(self._clients):
            if client.queue.full():
                try:
                    client.queue.get_nowait()  # descarta el mas viejo
                    client.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                client.queue.put_nowait(data)
            except asyncio.QueueFull:
                client.dropped += 1

    async def pump(self, client: ViewerClient) -> None:
        """Tarea por cliente: vacia su cola hacia el WebSocket."""
        while True:
            data = await client.queue.get()
            if isinstance(data, str):
                await client.ws.send_text(data)
            else:
                await client.ws.send_bytes(data)
