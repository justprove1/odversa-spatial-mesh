/**
 * Odversa · captura en el movil.
 *
 * El movil no reconstruye nada: abre la camara, comprime cada frame y lo manda
 * al ordenador. Todo el trabajo pesado (profundidad, malla) ocurre alli.
 *
 * Decisiones de latencia:
 *  - JPEG por WebSocket binario. Sobre la red local va por debajo de los 30 ms
 *    y, a diferencia de WebRTC, entrega frames intactos y con marca de tiempo,
 *    que es justo lo que necesita la reconstruccion.
 *  - Si el socket se atasca (`bufferedAmount`), se salta el frame en vez de
 *    encolarlo: mas vale perder uno que ir medio segundo por detras.
 *  - Cada frame lleva delante 8 bytes con el instante de captura, para medir la
 *    latencia extremo a extremo en el servidor.
 */

const CAPTURE_WIDTH = 640;      // ancho al que se envia (el sensor captura mas)
const JPEG_QUALITY = 0.62;
const MAX_BUFFERED = 96 * 1024; // umbral de atasco del socket
const IMU_HZ = 30;
// Frames enviados sin que el servidor haya confirmado que los consumio. Con 2
// hay siempre uno listo para entrar en cuanto el pipeline termina el anterior,
// sin llegar a formar cola. Con mas, la latencia crece; con 1, se pierde ritmo.
const MAX_IN_FLIGHT = 2;

const els = {
  video: document.getElementById('video'),
  canvas: document.getElementById('scratch'),
  start: document.getElementById('start'),
  state: document.getElementById('state'),
  res: document.getElementById('m-res'),
  fps: document.getElementById('m-fps'),
  kbps: document.getElementById('m-kbps'),
  hint: document.querySelector('.hint'),
};

const ctx = els.canvas.getContext('2d', { alpha: false, willReadFrequently: false });

let ws = null;
let stream = null;
let running = false;
let inFlight = 0;
let sentBytes = 0;
let sentFrames = 0;
let lastReport = performance.now();
let wakeLock = null;

function setState(text, cls = '') {
  els.state.textContent = text;
  els.state.className = 'state ' + cls;
}

function setHint(text) {
  els.hint.innerHTML = text;
}

/**
 * Traduce los fallos de `getUserMedia` a algo accionable.
 *
 * El caso que mas despista es "Permission denied by system": no lo manda la
 * pagina ni el navegador, lo manda el sistema operativo, que tiene bloqueada la
 * camara para la app del navegador. Por eso ni siquiera aparece el dialogo de
 * permiso, y por eso no sirve de nada recargar la pagina.
 */
function explainCameraError(err) {
  const name = err?.name || '';
  const message = err?.message || String(err);
  const system = /denied by system|not allowed by system/i.test(message);
  const android = /Android/i.test(navigator.userAgent);
  const ios = /iPhone|iPad|iPod/i.test(navigator.userAgent);

  if (system) {
    if (android) {
      return ['el sistema bloquea la cámara',
        'El <b>sistema operativo</b> tiene la cámara bloqueada para el navegador.<br>' +
        'Ajustes → Aplicaciones → tu navegador → Permisos → Cámara → <b>Permitir</b>, ' +
        'y vuelve aquí.'];
    }
    if (ios) {
      return ['el sistema bloquea la cámara',
        'El <b>sistema</b> tiene la cámara bloqueada para el navegador.<br>' +
        'Ajustes → tu navegador → <b>Cámara</b> → Permitir. Mira también ' +
        'Tiempo de uso → Restricciones de contenido y privacidad → Cámara.'];
    }
    return ['el sistema bloquea la cámara',
      'El <b>sistema operativo</b> tiene la cámara bloqueada para el navegador. ' +
      'Actívala en los ajustes de privacidad del sistema y vuelve aquí.'];
  }

  if (name === 'NotAllowedError') {
    return ['permiso denegado',
      'Has denegado el permiso de cámara a esta página. Tócalo en el candado de ' +
      'la barra de direcciones y ponlo en <b>Permitir</b>.'];
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return ['sin cámara disponible',
      'No se encuentra ninguna cámara trasera utilizable en este dispositivo.'];
  }
  if (name === 'NotReadableError') {
    return ['cámara ocupada',
      'Otra aplicación está usando la cámara. Ciérrala y pulsa REINTENTAR.'];
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return ['contexto no seguro',
      'El navegador no ofrece la cámara aquí. Entra por <b>https://</b> y acepta ' +
      'el aviso de certificado antes de pulsar el botón.'];
  }
  return [message || 'error', message];
}

// -------------------------------------------------------------- conexion ---
function connect() {
  return new Promise((resolve, reject) => {
    const url = `wss://${location.host}/ws/phone`;
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { inFlight = 0; resolve(); };
    ws.onmessage = (ev) => {
      // El servidor confirma cada frame consumido; ese aviso es lo que marca
      // el ritmo de envio.
      if (typeof ev.data === 'string' && ev.data.includes('"ack"')) {
        inFlight = Math.max(0, inFlight - 1);
      }
    };
    ws.onerror = () => reject(new Error('no se pudo conectar'));
    ws.onclose = () => {
      if (running) {
        setState('reconectando', 'err');
        setTimeout(() => connect().then(sendHello).catch(() => {}), 1200);
      }
    };
  });
}

function sendHello() {
  const track = stream?.getVideoTracks()[0];
  const settings = track ? track.getSettings() : {};
  ws.send(JSON.stringify({
    type: 'hello',
    width: els.canvas.width,
    height: els.canvas.height,
    // Si el navegador expone el FOV real de la lente lo usamos; si no, el
    // servidor asume ~65 grados, tipico de una camara trasera.
    hfov_deg: settings.fieldOfView || null,
    device: navigator.userAgent,
  }));
}

// ---------------------------------------------------------------- camara ---
async function openCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
    },
    audio: false,
  });
  els.video.srcObject = stream;
  await els.video.play();

  const vw = els.video.videoWidth || 1280;
  const vh = els.video.videoHeight || 720;
  els.canvas.width = CAPTURE_WIDTH;
  els.canvas.height = Math.round(CAPTURE_WIDTH * vh / vw);
  els.res.textContent = `${els.canvas.width}×${els.canvas.height}`;

  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* opcional */ }
}

// Un solo buffer reutilizado: evita que el recolector de basura entre a saco
// en mitad del streaming.
const stampBuffer = new ArrayBuffer(8);
const stampView = new DataView(stampBuffer);

function sendFrame(blob) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  blob.arrayBuffer().then((buf) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    stampView.setFloat64(0, Date.now() / 1000, true);
    const out = new Uint8Array(8 + buf.byteLength);
    out.set(new Uint8Array(stampBuffer), 0);
    out.set(new Uint8Array(buf), 8);
    ws.send(out);
    inFlight += 1;
    sentBytes += out.byteLength;
    sentFrames += 1;
  });
}

function captureLoop() {
  if (!running) return;

  const ready = els.video.readyState >= 2;
  const clear = ws && ws.readyState === WebSocket.OPEN
    && ws.bufferedAmount < MAX_BUFFERED
    && inFlight < MAX_IN_FLIGHT;

  if (ready && clear) {
    ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
    els.canvas.toBlob(sendFrame, 'image/jpeg', JPEG_QUALITY);
  }

  // `requestVideoFrameCallback` sincroniza con los frames reales de la camara;
  // sin el, un temporizador a 30 Hz hace el mismo trabajo.
  if (els.video.requestVideoFrameCallback) {
    els.video.requestVideoFrameCallback(() => captureLoop());
  } else {
    setTimeout(captureLoop, 33);
  }
}

function reportLoop() {
  const now = performance.now();
  const dt = (now - lastReport) / 1000;
  if (dt >= 1) {
    els.fps.textContent = (sentFrames / dt).toFixed(0);
    els.kbps.textContent = (sentBytes / dt / 1024).toFixed(0);
    sentFrames = 0; sentBytes = 0; lastReport = now;
  }
  if (running) requestAnimationFrame(reportLoop);
}

// ------------------------------------------------------------------- imu ---
async function startImu() {
  // iOS exige pedir permiso desde un gesto del usuario.
  for (const api of [window.DeviceMotionEvent, window.DeviceOrientationEvent]) {
    if (api && typeof api.requestPermission === 'function') {
      try { await api.requestPermission(); } catch { /* el usuario dijo que no */ }
    }
  }

  let orientation = null;
  window.addEventListener('deviceorientation', (e) => {
    if (e.alpha == null) return;
    const d = Math.PI / 180;
    orientation = [e.alpha * d, e.beta * d, e.gamma * d];
  });

  let lastSent = 0;
  window.addEventListener('devicemotion', (e) => {
    const now = performance.now();
    if (now - lastSent < 1000 / IMU_HZ) return;
    lastSent = now;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const r = e.rotationRate || {};
    const a = e.acceleration || {};
    const d = Math.PI / 180;
    ws.send(JSON.stringify({
      type: 'imu',
      t: Date.now() / 1000,
      gyro: [(r.beta || 0) * d, (r.gamma || 0) * d, (r.alpha || 0) * d],
      accel: [a.x || 0, a.y || 0, a.z || 0],
      orientation,
    }));
  });
}

// ---------------------------------------------------------------- arranque --
els.start.addEventListener('click', async () => {
  els.start.disabled = true;
  setState('conectando');
  try {
    await openCamera();
    await connect();
    sendHello();
    startImu();
    running = true;
    setState('transmitiendo', 'on');
    els.start.textContent = 'TRANSMITIENDO';
    captureLoop();
    reportLoop();
  } catch (err) {
    const [short, detail] = explainCameraError(err);
    setState(short, 'err');
    setHint(detail);
    els.start.disabled = false;
    els.start.textContent = 'REINTENTAR';
    console.error('[odversa] fallo al abrir la camara:', err);
  }
});

// Al volver a la app tras bloquear la pantalla hay que repedir el wake lock.
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState === 'visible' && running && wakeLock === null) {
    try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* opcional */ }
  }
});
