"""Pequeno servidor HTTP que redirige a HTTPS.

Motivo: el servidor real solo habla TLS, asi que una peticion `http://` al
puerto 8443 no falla con un mensaje util, falla en seco. Y como los navegadores
asumen `http://` cuando tecleas una direccion sin esquema, escribir
`192.168.1.14:8443/phone` en el movil acaba en "no se puede acceder a este
sitio" sin ninguna pista de por que.

Con esto, `http://IP:8080/phone` responde una redireccion a la URL https
correcta. Tambien sirve de diagnostico: si el puerto 8080 responde y el 8443 no,
el problema es el certificado; si no responde ninguno, es la red.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def start_redirect_server(http_port: int, https_port: int
                          ) -> tuple[ThreadingHTTPServer | None, int | None]:
    """Arranca el redirector en un hilo aparte.

    Devuelve `(servidor, puerto_real)`. El puerto real puede no ser el pedido:
    si esta ocupado se prueban alternativas y, como ultimo recurso, se deja que
    el sistema asigne uno libre. Quien llama DEBE usar el puerto devuelto para
    construir la URL, porque anunciar un puerto que no es el que se abrio manda
    al movil al servidor de otro proyecto -y lo que ve es un 404 sin sentido.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _redirect(self) -> None:
            host = self.headers.get("Host", "").split(":")[0] or "localhost"
            target = f"https://{host}:{https_port}{self.path}"
            body = (
                f"<!doctype html><meta charset=utf-8>"
                f'<meta http-equiv="refresh" content="0;url={target}">'
                f'<body style="background:#05060a;color:#e8ecf4;font-family:sans-serif;'
                f'padding:40px;text-align:center">Odversa usa HTTPS.<br><br>'
                f'<a style="color:#4fd6ff" href="{target}">{target}</a></body>'
            ).encode("utf-8")
            self.send_response(307)
            self.send_header("Location", target)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _redirect
        do_POST = _redirect
        do_HEAD = _redirect

        def log_message(self, *args) -> None:  # sin ruido en la consola
            pass

    # 0 al final = "el que sea, con tal de que este libre".
    for candidate in (http_port, 8442, 8444, 8899, 0):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", candidate), Handler)
        except OSError:
            continue
        bound = server.server_address[1]
        threading.Thread(target=server.serve_forever, name="odversa-redirect",
                         daemon=True).start()
        if bound != http_port:
            print(f"[odversa] el puerto {http_port} estaba ocupado; "
                  f"redireccion http en el {bound}")
        return server, bound
    print("[odversa] no se pudo abrir ningun puerto para la redireccion http")
    return None, None
