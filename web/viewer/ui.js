/**
 * UI - controles e indicadores.
 *
 * Cada deslizador se traduce a una clave de la configuracion del servidor y se
 * manda con retardo (`debounce`): mover el control no debe disparar veinte
 * reconfiguraciones del pipeline por segundo.
 */

const $ = (id) => document.getElementById(id);

/**
 * Combinaciones de calidad.
 *
 * Van juntas a proposito: subir la densidad de malla sin subir la resolucion de
 * la red solo dibuja mas triangulos sobre la misma informacion borrosa, y subir
 * el realce con mucho suavizado se anula solo.
 */
const PRESETS = {
  fast: {
    values: { proc_width: 160, depth_input_size: 266, mesh_detail_cm: 2.0, detail_boost: 0.30,
              spatial_smoothing: 0.30, guided_radius: 3, depth_smoothing: 0.60 },
  },
  balanced: {
    values: { proc_width: 224, depth_input_size: 308, mesh_detail_cm: 1.2, detail_boost: 0.45,
              spatial_smoothing: 0.25, guided_radius: 4, depth_smoothing: 0.55 },
  },
  detail: {
    values: { proc_width: 288, depth_input_size: 448, mesh_detail_cm: 0.7, detail_boost: 0.60,
              spatial_smoothing: 0.15, guided_radius: 5, depth_smoothing: 0.45 },
  },
};

const SLIDERS = [
  { id: 'c-density',  key: 'proc_width',        label: 'v-density',  fmt: (v) => v.toFixed(0) },
  { id: 'c-res',      key: 'depth_input_size',  label: 'v-res',      fmt: (v) => v.toFixed(0) },
  { id: 'c-detailcm', key: 'mesh_detail_cm',    label: 'v-detailcm', fmt: (v) => `${v.toFixed(1)} cm` },
  { id: 'c-smooth',   key: 'spatial_smoothing', label: 'v-smooth',   fmt: (v) => v.toFixed(2) },
  { id: 'c-detail',   key: 'detail_boost',      label: 'v-detail',   fmt: (v) => v.toFixed(2) },
  { id: 'c-near',     key: 'near_m',            label: 'v-near',     fmt: (v) => `${v.toFixed(1)} m` },
  { id: 'c-far',      key: 'far_m',             label: 'v-far',      fmt: (v) => `${v.toFixed(1)} m` },
  { id: 'c-edge',     key: 'edge_tolerance',    label: 'v-edge',     fmt: (v) => v.toFixed(3) },
  { id: 'c-temporal', key: 'depth_smoothing',   label: 'v-temporal', fmt: (v) => v.toFixed(2) },
];

/**
 * Estados del tracking: etiqueta y clase de color.
 *
 * Se traducen porque "WEAK" no le dice nada a quien mira el panel, y lo que
 * necesita saber es si puede fiarse de la posicion o no.
 */
const TRACK_STATE = {
  INIT: ['Fijando origen', ''],
  GOOD: ['Siguiendo', 'ok'],
  WEAK: ['Referencia débil', 'warn'],
  LOST: ['Sin referencia', 'bad'],
};

const TRACK_NOTE = {
  INIT: 'esperando textura para engancharse',
  GOOD: 'odometría relativa al punto de partida',
  WEAK: 'poca textura: la posición puede derivar',
  LOST: 'mueve el móvil más despacio o apunta a algo con relieve',
};

/** Nombre corto y legible del movil a partir de su cadena de usuario. */
function deviceName(ua) {
  if (!ua) return 'desconocido';
  if (/iPad/i.test(ua)) return 'iPad';
  if (/iPhone/i.test(ua)) return 'iPhone';
  const android = ua.match(/Android[^;]*;\s*([^;)]+)/i);
  if (android) return android[1].trim().slice(0, 22);
  if (/Macintosh/i.test(ua)) return 'Mac';
  if (/Windows/i.test(ua)) return 'Windows';
  return 'cámara';
}

export class UI {
  constructor({ onConfig, onMode, onReset, onSave, onMeshToggle, onOrbitToggle, onSearch }) {
    this.onConfig = onConfig;
    this._timer = null;
    // Recorrido en planta (x, z) y orientacion actual, para el panel de tracking.
    this._trail = [[0, 0]];
    this._yaw = 0;
    this._pending = {};

    $('modes').addEventListener('click', (e) => {
      const button = e.target.closest('button[data-mode]');
      if (!button) return;
      for (const b of $('modes').children) b.classList.toggle('active', b === button);
      onMode(button.dataset.mode);
    });

    $('presets').addEventListener('click', (e) => {
      const button = e.target.closest('button[data-preset]');
      if (!button) return;
      const preset = PRESETS[button.dataset.preset];
      if (!preset) return;
      for (const b of $('presets').children) b.classList.toggle('on', b === button);
      this.onConfig.send(preset.values);
      this.syncConfig(preset.values);
    });

    for (const slider of SLIDERS) {
      const input = $(slider.id);
      const label = $(slider.label);
      const paint = () => {
        label.textContent = slider.fmt(parseFloat(input.value));
        this._paintTrack(input);
      };
      paint();
      input.addEventListener('input', () => {
        paint();
        this._queue(slider.key, parseFloat(input.value));
        if (slider.key === 'far_m') onConfig.onFarChange?.(parseFloat(input.value));
        if (slider.key in PRESETS.balanced.values) this._clearPreset();
      });
    }

    this.meshButton = $('t-mesh');
    this.meshButton.addEventListener('click', () => {
      const on = this.meshButton.classList.toggle('on');
      this.meshButton.textContent = on ? 'MALLA ACTIVA' : 'MALLA APAGADA';
      this._queue('mesh_enabled', on);
      onMeshToggle(on);
    });

    this.orbitButton = $('t-orbit');
    this.orbitButton.addEventListener('click', () => {
      onOrbitToggle(this.orbitButton.classList.toggle('on'));
    });

    $('a-reset').addEventListener('click', onReset);
    $('a-save').addEventListener('click', onSave);

    // Busqueda de objetos: se envia con retardo para no lanzar una consulta
    // por cada tecla; Escape la borra.
    const searchInput = $('search-input');
    let searchTimer = null;
    searchInput.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => onSearch(searchInput.value.trim()), 450);
    });
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        searchInput.value = '';
        clearTimeout(searchTimer);
        onSearch('');
      } else if (e.key === 'Enter') {
        clearTimeout(searchTimer);
        onSearch(searchInput.value.trim());
      }
    });

    this._drawTracking();
  }

  /** Pinta la parte recorrida del deslizador en verde. */
  _paintTrack(input) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const pct = ((parseFloat(input.value) - min) / (max - min)) * 100;
    input.style.setProperty('--fill', `${pct}%`);
  }

  _clearPreset() {
    for (const b of $('presets').children) b.classList.remove('on');
  }

  _queue(key, value) {
    this._pending[key] = value;
    clearTimeout(this._timer);
    this._timer = setTimeout(() => {
      const values = this._pending;
      this._pending = {};
      this.onConfig.send(values);
    }, 110);
  }

  setOrbitActive(on) {
    this.orbitButton.classList.toggle('on', on);
  }

  note(text) {
    $('note').textContent = text;
    clearTimeout(this._noteTimer);
    this._noteTimer = setTimeout(() => { $('note').textContent = ''; }, 6000);
  }

  setPhoneUrl(ip, port, httpUrl) {
    $('phone-url').textContent = `https://${ip}:${port}/phone`;
    if (httpUrl) $('phone-http').textContent = httpUrl;
  }

  setWaiting(waiting) {
    $('waiting').classList.toggle('hidden', !waiting);
  }

  setGrid(text) { $('s-grid').textContent = text; }

  /** Estado de la busqueda bajo la barra: coincidencias o silencio. */
  setSearchStatus(search) {
    const el = $('search-status');
    if (!search || !search.query) {
      el.textContent = '';
      el.className = 'search-status';
      return;
    }
    const hits = (search.boxes || []).length;
    el.textContent = `${search.status}${search.ms ? ` · ${search.ms} ms` : ''}`;
    el.className = 'search-status' + (hits ? ' hit' : '');
  }

  /** Vuelca las estadisticas del servidor en los indicadores. */
  update(stats) {
    const connected = stats.connected && stats.fps_pipeline > 0.2;
    $('c-state').textContent = connected ? 'Conectado' : 'Sin señal';
    $('c-dot').classList.toggle('on', connected);
    $('c-device').textContent = connected ? deviceName(stats.device) : '—';

    $('s-fps').textContent = stats.fps_pipeline.toFixed(1);
    $('s-lat').textContent = `${stats.latency_ms.toFixed(0)} ms`;
    $('s-points').textContent = stats.points_total.toLocaleString('es-ES');
    $('s-tris').textContent = stats.triangles.toLocaleString('es-ES');

    const track = $('s-track');
    track.textContent = stats.tracking === 'INIT' ? 'Relativo' : stats.tracking;
    track.className = { GOOD: 'ok', WEAK: 'warn', LOST: 'bad' }[stats.tracking] ?? '';

    this._updateTracking(stats);
    this._updateSurfaces(stats.surfaces);
  }

  /** Pinta las superficies que la IA geometrica ha detectado. */
  _updateSurfaces(surfaces) {
    const box = $('surfaces');
    if (!surfaces || !surfaces.length) {
      box.innerHTML = '<span class="none">sin superficies detectadas</span>';
      return;
    }
    const NAMES = { suelo: 'Suelo', pared: 'Pared', mesa: 'Mesa',
                    techo: 'Techo', superficie: 'Superficie' };
    box.innerHTML = surfaces.map((s) =>
      `<span class="chip">${NAMES[s.tipo] ?? s.tipo} <i>${s.area.toFixed(0)}%</i></span>`
    ).join('');
  }

  /**
   * Panel de tracking.
   *
   * La pose viene de la odometria RGB-D del servidor. Es RELATIVA: el origen es
   * donde estaba la camara al empezar y la escala la fija la calibracion de
   * profundidad, asi que se etiqueta como tal en vez de venderla como metrica
   * absoluta. El estado se muestra siempre, porque una posicion con el tracking
   * en LOST es un numero viejo, no una medida.
   */
  _updateTracking(stats) {
    const p = stats.position ?? [0, 0, 0];
    const r = stats.rotation ?? [0, 0, 0];
    $('t-pos').textContent = `[X: ${p[0].toFixed(2)}, Y: ${p[1].toFixed(2)}, Z: ${p[2].toFixed(2)}]`;
    $('t-rot').textContent = `[Yaw: ${r[0].toFixed(1)}°, Pitch: ${r[1].toFixed(1)}°, Roll: ${r[2].toFixed(1)}°]`;

    const state = $('t-state');
    state.textContent = TRACK_STATE[stats.tracking]?.[0] ?? stats.tracking;
    state.className = TRACK_STATE[stats.tracking]?.[1] ?? '';
    $('track-note').textContent = TRACK_NOTE[stats.tracking] ?? '';

    // Rastro del recorrido, visto desde arriba (plano XZ).
    const last = this._trail[this._trail.length - 1];
    if (!last || Math.hypot(p[0] - last[0], p[2] - last[1]) > 0.01) {
      this._trail.push([p[0], p[2]]);
      if (this._trail.length > 400) this._trail.shift();
    }
    if (stats.tracking === 'INIT' && this._trail.length > 1) this._trail = [[p[0], p[2]]];
    this._yaw = (r[0] * Math.PI) / 180;
    this._drawTracking();
  }

  /**
   * Planta del recorrido: rejilla en metros, rastro y la camara mirando a donde
   * mira. Se dibuja en planta y no en perspectiva porque lo que interesa
   * comprobar de un vistazo es si el camino tiene la forma del paseo real.
   */
  _drawTracking() {
    const canvas = $('track-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * devicePixelRatio;
    const h = canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.clearRect(0, 0, w, h);

    const trail = this._trail;
    // Encuadre: al menos 1 m de lado, y crece con el recorrido para que no se
    // salga nunca del cuadro.
    let span = 0.5;
    for (const [x, z] of trail) span = Math.max(span, Math.abs(x), Math.abs(z));
    const scale = (Math.min(w, h) * 0.42) / (span * 1.15);
    const cx = w * 0.5, cy = h * 0.5;
    const toPx = ([x, z]) => [cx + x * scale, cy - z * scale];

    // Rejilla de 1 m, o de 25 cm si el recorrido es corto.
    const step = span < 1.0 ? 0.25 : 1.0;
    ctx.strokeStyle = 'rgba(120,140,130,.16)';
    ctx.lineWidth = devicePixelRatio;
    for (let g = -6; g <= 6; g++) {
      const d = g * step * scale;
      if (Math.abs(d) > Math.max(w, h) * 0.5) continue;
      ctx.beginPath(); ctx.moveTo(cx + d, 0); ctx.lineTo(cx + d, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, cy + d); ctx.lineTo(w, cy + d); ctx.stroke();
    }

    // Origen: donde se empezo a seguir.
    ctx.fillStyle = 'rgba(120,140,130,.55)';
    ctx.beginPath(); ctx.arc(cx, cy, 2.2 * devicePixelRatio, 0, Math.PI * 2); ctx.fill();

    // Rastro.
    if (trail.length > 1) {
      ctx.strokeStyle = 'rgba(47,191,106,.75)';
      ctx.lineWidth = 1.4 * devicePixelRatio;
      ctx.beginPath();
      trail.forEach((p, i) => {
        const [px, py] = toPx(p);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.stroke();
    }

    // Camara: triangulo orientado segun el yaw.
    const [px, py] = toPx(trail[trail.length - 1] ?? [0, 0]);
    const s = Math.min(w, h) * 0.11;
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(-this._yaw);
    ctx.strokeStyle = '#2fbf6a';
    ctx.fillStyle = 'rgba(47,191,106,.20)';
    ctx.lineWidth = 1.6 * devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(0, -s * 0.9);
    ctx.lineTo(-s * 0.6, s * 0.5);
    ctx.lineTo(s * 0.6, s * 0.5);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // Escala, para que las distancias del rastro se puedan leer.
    ctx.fillStyle = 'rgba(150,170,160,.6)';
    ctx.font = `${11 * devicePixelRatio}px ui-monospace, monospace`;
    ctx.fillText(`rejilla ${step < 1 ? '25 cm' : '1 m'}`, 6 * devicePixelRatio,
                 h - 6 * devicePixelRatio);
  }

  /** Sincroniza los controles con la configuracion real del servidor. */
  syncConfig(config) {
    for (const slider of SLIDERS) {
      const value = config[slider.key];
      if (value === undefined) continue;
      const input = $(slider.id);
      if (document.activeElement === input) continue;
      input.value = value;
      $(slider.label).textContent = slider.fmt(parseFloat(value));
      this._paintTrack(input);
    }
  }
}
