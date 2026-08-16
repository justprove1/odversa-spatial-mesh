"""Codigo QR en SVG para abrir la pagina de captura en el movil.

Teclear `https://192.168.1.14:8443/phone` en un telefono es la parte mas
propensa a fallar de todo el montaje: un digito mal y no carga. El visor pinta
este QR mientras espera, y con escanearlo basta.

Se genera SVG a mano en vez de PNG para no arrastrar Pillow y para que escale
sin pixelarse en cualquier tamano.
"""

from __future__ import annotations

import qrcode


def qr_svg(data: str, module_px: int = 8, quiet_zone: int = 4) -> str:
    """Devuelve un SVG con el QR de `data`.

    `quiet_zone` es el margen blanco alrededor, en modulos. El estandar pide 4 y
    los lectores fallan con menos: sin ese margen, el borde del QR se confunde
    con lo que haya detras en la pantalla.
    """
    code = qrcode.QRCode(box_size=1, border=quiet_zone,
                         error_correction=qrcode.constants.ERROR_CORRECT_Q)
    code.add_data(data)
    code.make(fit=True)
    matrix = code.get_matrix()

    size = len(matrix)
    side = size * module_px

    # Se agrupan los modulos contiguos de cada fila en un solo rectangulo: el
    # SVG queda en torno a diez veces mas pequeno que pintando cuadro a cuadro.
    rects: list[str] = []
    for y, row in enumerate(matrix):
        x = 0
        while x < size:
            if not row[x]:
                x += 1
                continue
            run = x
            while run < size and row[run]:
                run += 1
            rects.append(
                f'<rect x="{x * module_px}" y="{y * module_px}" '
                f'width="{(run - x) * module_px}" height="{module_px}"/>'
            )
            x = run

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
        f'viewBox="0 0 {side} {side}" shape-rendering="crispEdges">'
        f'<rect width="{side}" height="{side}" fill="#ffffff"/>'
        f'<g fill="#000000">{"".join(rects)}</g>'
        f"</svg>"
    )
