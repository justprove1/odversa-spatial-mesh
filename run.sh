#!/usr/bin/env bash
# Arranca Odversa Spatial Mesh.
#   ./run.sh            -> HTTPS (necesario para que el movil abra la camara)
#   ODVERSA_PORT=9443 ./run.sh   (para cambiar de puerto)
set -euo pipefail

cd "$(dirname "$0")"
PORT="${ODVERSA_PORT:-10000}"
HTTP_PORT="${ODVERSA_HTTP_PORT:-10001}"
export ODVERSA_PORT HTTP_PORT
export ODVERSA_HTTP_PORT="$HTTP_PORT"
CERT_DIR="certs"
CRT="$CERT_DIR/odversa.crt"
KEY="$CERT_DIR/odversa.key"

# Si el puerto ya esta cogido, uvicorn muere con un "Errno 48" que no dice
# quien lo tiene ni que hacer. Casi siempre es otro Odversa que sigue vivo.
BUSY_PID="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
if [ -n "$BUSY_PID" ]; then
  BUSY_CMD="$(ps -p "$BUSY_PID" -o comm= 2>/dev/null || echo desconocido)"
  cat <<EOF

  El puerto $PORT ya esta ocupado por el proceso $BUSY_PID ($BUSY_CMD).

  Si es otro Odversa que dejaste abierto, ciérralo con:
      kill $BUSY_PID
  o abre directamente el que ya esta corriendo:
      https://localhost:$PORT/

  Tambien puedes arrancar en otro puerto:
      ODVERSA_PORT=10100 ./run.sh

EOF
  exit 1
fi

# IP en la red local: es la que hay que teclear en el movil.
IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"

# Certificado autofirmado, con la IP dentro para que el navegador del movil
# acepte la excepcion. Se regenera si cambia la IP de la maquina.
if [ ! -f "$CRT" ] || ! openssl x509 -in "$CRT" -noout -text 2>/dev/null | grep -q "IP Address:$IP"; then
  echo "==> generando certificado para $IP"
  mkdir -p "$CERT_DIR"
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout "$KEY" -out "$CRT" \
    -subj "/CN=odversa-spatial-mesh" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$IP" >/dev/null 2>&1
fi

cat <<EOF

  ODVERSA SPATIAL MESH

  Visor (en este ordenador):   https://localhost:$PORT/
  Camara (en el movil):        https://$IP:$PORT/phone

  Lo mas comodo es escanear el codigo QR que sale en el visor.

  Si tecleas la direccion a mano, el "https://" NO es opcional: sin el, el
  navegador prueba por http, no encuentra nada y dice "no se puede acceder
  a este sitio". Por si acaso, la direccion http que aparece abajo redirige al sitio
  correcto.

  El movil avisara de que el certificado no es de fiar: es el autofirmado
  de esta maquina. Acepta la excepcion y deja abrir la camara.

EOF

exec .venv/bin/python -m uvicorn server.main:app \
  --host 0.0.0.0 --port "$PORT" \
  --ssl-certfile "$CRT" --ssl-keyfile "$KEY" \
  --ws-max-size 33554432 \
  --log-level warning
