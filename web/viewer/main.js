/**
 * ODVERSA SPATIAL MESH - visor de escritorio.
 *
 * Une las tres piezas del cliente: la red (`NetClient`), el dibujado
 * (`Renderer`) y los controles (`UI`). Aqui no hay logica de reconstruccion:
 * todo eso vive en el servidor.
 */

import { NetClient } from './net.js';
import { Renderer } from './renderer.js';
import { UI } from './ui.js';

const views = document.getElementById('views');
const renderer = new Renderer({
  glCanvas: document.getElementById('gl-canvas'),
  videoCanvas: document.getElementById('video-canvas'),
  views,
  bodyA: document.getElementById('body-a'),
  bodyB: document.getElementById('body-b'),
});

let lastFrameAt = 0;
let lastGrid = '';

const net = new NetClient({
  onOpen: () => ui.note('conectado al servidor'),
  onClose: () => { ui.setWaiting(true); ui.note('servidor desconectado'); },

  onHello: (msg) => {
    ui.syncConfig(msg.config);
    ui.setPhoneUrl(msg.ip, location.port || '443', msg.http_url);
    renderer.setFar(msg.config.far_m);
    ui.note(`profundidad: ${msg.depth_backend}`);
  },

  onFrame: (frame) => {
    lastFrameAt = performance.now();
    ui.setWaiting(false);
    renderer.applyFrame(frame);
    ui.setSearchStatus(frame.header.search);
    const grid = `${frame.header.w}×${frame.header.h}`;
    if (grid !== lastGrid) { lastGrid = grid; ui.setGrid(grid); }
  },

  onStats: (stats, config) => {
    ui.update(stats);
    ui.syncConfig(config);
  },

  onSaved: (msg) => {
    ui.note(msg.ok ? `guardado: ${msg.path.split('/').pop()} (${msg.triangles} tri)`
                   : `no se pudo guardar: ${msg.error}`);
  },
});

const ui = new UI({
  onConfig: {
    send: (values) => net.setConfig(values),
    onFarChange: (far) => renderer.setFar(far),
  },
  onMode: (mode) => renderer.setMode(mode),
  onMeshToggle: (on) => renderer.setMeshVisible(on),
  onOrbitToggle: (on) => {
    if (on && renderer.mode !== 'mesh') {
      // La orbita rompe la superposicion sobre la imagen: se cambia de modo.
      document.querySelector('[data-mode="mesh"]').click();
    }
    renderer.setOrbit(on);
    ui.setOrbitActive(renderer.orbitEnabled);
  },
  onReset: () => { net.send('reset'); ui.note('reconstrucción reiniciada'); },
  onSave: () => net.send('save'),
  onSearch: (query) => net.send('search', { query }),
});

// Si dejan de llegar frames, se avisa: es el sintoma de que el movil se ha
// dormido, ha cambiado de red o ha cerrado la pestana.
setInterval(() => {
  if (lastFrameAt && performance.now() - lastFrameAt > 2500) ui.setWaiting(true);
}, 1000);

window.addEventListener('resize', () => renderer.resize());
new ResizeObserver(() => renderer.resize()).observe(views);
renderer.resize();
renderer.setMode('real-mesh');

(function loop() {
  renderer.render();
  requestAnimationFrame(loop);
})();

net.connect();
