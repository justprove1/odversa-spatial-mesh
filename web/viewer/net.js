/**
 * Cliente de red del visor: decodifica el protocolo binario del servidor.
 *
 * Formato: [uint32 tamano de cabecera][cabecera JSON][bloques binarios]
 * La cabecera viene rellenada a multiplo de 4, asi que los vertices y los
 * indices se leen como vistas directas sobre el ArrayBuffer recibido: cero
 * copias entre la red y la GPU.
 */

const decoder = new TextDecoder();

export class NetClient {
  constructor(handlers = {}) {
    this.handlers = handlers;
    this.ws = null;
    this.reconnectDelay = 700;
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/ws/viewer`);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') this._onText(ev.data);
      else this._onBinary(ev.data);
    };
    this.ws.onopen = () => {
      this.reconnectDelay = 700;  // conexion buena: se reinicia la espera
      this.handlers.onOpen?.();
    };
    this.ws.onclose = () => {
      this.handlers.onClose?.();
      // Espera progresiva hasta 8 s. Insistir cada 700 ms mientras el servidor
      // esta parado solo llena la consola de errores y gasta bateria.
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.6, 8000);
    };
    this.ws.onerror = () => this.ws.close();
  }

  _onText(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    if (msg.type === 'stats') this.handlers.onStats?.(msg.stats, msg.config);
    else if (msg.type === 'hello') this.handlers.onHello?.(msg);
    else if (msg.type === 'saved') this.handlers.onSaved?.(msg);
  }

  _onBinary(buffer) {
    const view = new DataView(buffer);
    const headerLength = view.getUint32(0, true);
    const header = JSON.parse(decoder.decode(new Uint8Array(buffer, 4, headerLength)));
    let offset = 4 + headerLength;

    const blocks = {};
    header.order.forEach((name, i) => {
      const size = header.blocks[i];
      blocks[name] = { offset, size };
      offset += size;
    });

    if (header.type === 'frame') {
      const v = blocks.vertices;
      const t = blocks.indices;
      const frame = {
        header,
        vertices: new Float32Array(buffer, v.offset, v.size / 4),
        indices: new Uint32Array(buffer, t.offset, t.size / 4),
        highlight: blocks.highlight
          ? new Uint8Array(buffer, blocks.highlight.offset, blocks.highlight.size)
          : null,
        jpeg: blocks.jpeg
          ? new Blob([new Uint8Array(buffer, blocks.jpeg.offset, blocks.jpeg.size)],
                     { type: 'image/jpeg' })
          : null,
      };
      this.handlers.onFrame?.(frame);
    }
  }

  send(type, payload = {}) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, ...payload }));
    }
  }

  setConfig(values) { this.send('config', { values }); }
}
