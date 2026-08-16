/**
 * Renderer - dibujado de la malla sobre (o en lugar de) la imagen real.
 *
 * La alineacion entre malla e imagen no es un ajuste a ojo: la malla se genera
 * retroproyectando la profundidad con los intrinsecos de ese frame, asi que
 * basta con mirarla desde el origen con el MISMO campo de vision vertical para
 * que cada triangulo caiga sobre el pixel del que salio. Por eso la region de
 * dibujo se recorta al rectangulo exacto que ocupa el video dentro de su
 * tarjeta, con su misma relacion de aspecto.
 *
 * Las dos vistas (real+malla y dimensional) comparten UN solo contexto WebGL y
 * UNA sola copia de la geometria: se dibuja dos veces recortando por region con
 * `setScissor`. Con dos renderizadores habria que subir la malla a la GPU dos
 * veces por frame.
 *
 * Conversion de ejes: el servidor trabaja en convenio OpenCV (x derecha, y
 * abajo, z hacia delante) y WebGL usa y arriba y -z hacia delante. La rotacion
 * de 180 grados sobre X del grupo `world` traduce (x,y,z) -> (x,-y,-z).
 *
 * Oclusion: se dibuja dos veces la misma geometria en cada region. Primero una
 * pasada opaca que solo escribe profundidad (sobre el video) o pinta negro (en
 * la vista dimensional), y encima el alambre. Sin eso se verian las lineas de
 * la pared del fondo a traves de los objetos.
 */

import * as THREE from './vendor/three.module.js';

const MESH_COLOR = 0xffffff;
const INITIAL_VERTS = 40000;
const INITIAL_INDICES = 120000;

export class Renderer {
  constructor({ glCanvas, videoCanvas, views, bodyA, bodyB }) {
    this.views = views;
    this.bodyA = bodyA;
    this.bodyB = bodyB;
    this.videoCanvas = videoCanvas;
    this.videoCtx = videoCanvas.getContext('2d', { alpha: false });
    this.mode = 'real-mesh';
    this.meshVisible = true;
    this.orbitEnabled = false;
    this.aspect = 4 / 3;
    this.farPlane = 8;

    this.renderer = new THREE.WebGLRenderer({
      canvas: glCanvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.autoClear = false;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(52, this.aspect, 0.05, 300);

    // Grupo que traduce del convenio de la camara al de WebGL.
    this.world = new THREE.Group();
    this.world.rotation.x = Math.PI;
    this.scene.add(this.world);

    this._buildMesh();
    this._initOrbit();
    this._resetView();
  }

  // ------------------------------------------------------------ geometria --
  _buildMesh() {
    this.geometry = new THREE.BufferGeometry();
    this.vertexCapacity = INITIAL_VERTS;
    this.indexCapacity = INITIAL_INDICES;
    this.geometry.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(this.vertexCapacity * 3), 3));
    this.geometry.setAttribute('normal',
      new THREE.BufferAttribute(new Float32Array(this.vertexCapacity * 3), 3));
    this.geometry.setAttribute('highlight',
      new THREE.BufferAttribute(new Uint8Array(this.vertexCapacity), 1, true));
    this.geometry.setIndex(
      new THREE.BufferAttribute(new Uint32Array(this.indexCapacity), 1));
    this.geometry.setDrawRange(0, 0);

    // Pasada de oclusion.
    this.fillMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      side: THREE.DoubleSide,
      colorWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: 1.2,
      polygonOffsetUnits: 1.2,
    });

    // Alambre con sombreado geometrico.
    //
    // Desde el punto de vista de la propia camara, una malla sacada de un mapa
    // de profundidad se proyecta siempre como una rejilla regular: la forma no
    // se ve porque cada celda ocupa lo mismo en pantalla. Modulando el brillo
    // de las lineas con la inclinacion de la superficie (cuanto mas de canto
    // esta un poligono, mas apagado) el suelo, las paredes y los objetos se
    // separan solos. Es informacion puramente geometrica: ni textura, ni color
    // del objeto, ni iluminacion simulada.
    this.wireMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(MESH_COLOR) },
        uOpacity: { value: 0.85 },
        uFogNear: { value: 1.0 },
        uFogFar: { value: 12.0 },
        uFogAmount: { value: 0.0 },
        uHlColor: { value: new THREE.Color(0xffb03a) },
      },
      vertexShader: `
        attribute float highlight;
        varying float vShade;
        varying float vDepth;
        varying float vHl;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vec3 n = normalize(normalMatrix * normal);
          vec3 viewDir = normalize(-mv.xyz);
          vShade = abs(dot(n, viewDir));
          vDepth = -mv.z;
          vHl = highlight;
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uOpacity;
        uniform float uFogNear;
        uniform float uFogFar;
        uniform float uFogAmount;
        uniform vec3 uHlColor;
        varying float vShade;
        varying float vDepth;
        varying float vHl;
        void main() {
          float shade = mix(0.30, 1.0, pow(clamp(vShade, 0.0, 1.0), 0.7));
          float fog = smoothstep(uFogNear, uFogFar, vDepth) * uFogAmount;
          vec3 base = uColor * shade;
          // El objeto buscado se pinta naranja, mas brillante que el resto y
          // sin dejar que la niebla de distancia lo apague.
          vec3 color = mix(base * (1.0 - fog), uHlColor, vHl * 0.9);
          float alpha = mix(uOpacity * (1.0 - fog * 0.65), 1.0, vHl * 0.6);
          gl_FragColor = vec4(color, alpha);
        }
      `,
      wireframe: true,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    this.fillMesh = new THREE.Mesh(this.geometry, this.fillMaterial);
    this.wireMesh = new THREE.Mesh(this.geometry, this.wireMaterial);
    this.fillMesh.renderOrder = 0;
    this.wireMesh.renderOrder = 1;
    this.fillMesh.frustumCulled = false;
    this.wireMesh.frustumCulled = false;
    this.world.add(this.fillMesh, this.wireMesh);
  }

  _ensureCapacity(vertexCount, indexCount) {
    if (vertexCount > this.vertexCapacity) {
      this.vertexCapacity = Math.ceil(vertexCount * 1.35);
      this.geometry.setAttribute('position',
        new THREE.BufferAttribute(new Float32Array(this.vertexCapacity * 3), 3));
      this.geometry.setAttribute('normal',
        new THREE.BufferAttribute(new Float32Array(this.vertexCapacity * 3), 3));
      this.geometry.setAttribute('highlight',
        new THREE.BufferAttribute(new Uint8Array(this.vertexCapacity), 1, true));
    }
    if (indexCount > this.indexCapacity) {
      this.indexCapacity = Math.ceil(indexCount * 1.35);
      this.geometry.setIndex(
        new THREE.BufferAttribute(new Uint32Array(this.indexCapacity), 1));
    }
  }

  updateMesh(vertices, indices, highlight) {
    const vertexCount = vertices.length / 3;
    this._ensureCapacity(vertexCount, indices.length);

    const hl = this.geometry.getAttribute('highlight');
    if (highlight && highlight.length === vertexCount) {
      hl.array.set(highlight);
    } else {
      hl.array.fill(0, 0, vertexCount);
    }
    hl.addUpdateRange(0, vertexCount);
    hl.needsUpdate = true;

    const position = this.geometry.getAttribute('position');
    position.array.set(vertices);
    position.addUpdateRange(0, vertices.length);
    position.needsUpdate = true;

    const index = this.geometry.getIndex();
    index.array.set(indices);
    index.addUpdateRange(0, indices.length);
    index.needsUpdate = true;

    this.geometry.setDrawRange(0, indices.length);
    this.geometry.boundingSphere = null;
    this._computeNormals(vertexCount, indices);
  }

  _computeNormals(vertexCount, indices) {
    const position = this.geometry.getAttribute('position');
    const normal = this.geometry.getAttribute('normal');
    const pos = position.array;
    const nor = normal.array;
    nor.fill(0, 0, vertexCount * 3);

    for (let i = 0; i < indices.length; i += 3) {
      const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
      const abx = pos[b] - pos[a], aby = pos[b + 1] - pos[a + 1], abz = pos[b + 2] - pos[a + 2];
      const acx = pos[c] - pos[a], acy = pos[c + 1] - pos[a + 1], acz = pos[c + 2] - pos[a + 2];
      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;
      nor[a] += nx; nor[a + 1] += ny; nor[a + 2] += nz;
      nor[b] += nx; nor[b + 1] += ny; nor[b + 2] += nz;
      nor[c] += nx; nor[c + 1] += ny; nor[c + 2] += nz;
    }
    for (let i = 0; i < vertexCount * 3; i += 3) {
      const x = nor[i], y = nor[i + 1], z = nor[i + 2];
      const len = Math.hypot(x, y, z) || 1;
      nor[i] = x / len; nor[i + 1] = y / len; nor[i + 2] = z / len;
    }
    normal.addUpdateRange(0, vertexCount * 3);
    normal.needsUpdate = true;
  }

  // ---------------------------------------------------------------- video --
  async drawVideo(blob, width, height) {
    if (this.videoCanvas.width !== width || this.videoCanvas.height !== height) {
      this.videoCanvas.width = width;
      this.videoCanvas.height = height;
    }
    const bitmap = await createImageBitmap(blob);
    this.videoCtx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    // Cajas de la busqueda sobre la imagen real: el mismo naranja que la malla.
    if (this._boxes && this._boxes.length) {
      const ctx = this.videoCtx;
      ctx.strokeStyle = '#ffb03a';
      ctx.fillStyle = '#ffb03a';
      ctx.lineWidth = Math.max(2, width / 320);
      ctx.font = `${Math.max(11, width / 45)}px ui-sans-serif, system-ui`;
      for (const b of this._boxes) {
        const x = b.x0 * width, y = b.y0 * height;
        const w = (b.x1 - b.x0) * width, h2 = (b.y1 - b.y0) * height;
        ctx.strokeRect(x, y, w, h2);
        const label = `${this._boxLabel} ${(b.score * 100).toFixed(0)}%`;
        ctx.fillText(label, x + 4, Math.max(14, y - 6));
      }
    }
  }

  // ---------------------------------------------------------------- frame --
  applyFrame(frame) {
    const h = frame.header;
    const aspect = h.src_w / h.src_h;
    if (Math.abs(aspect - this.aspect) > 1e-3) {
      this.aspect = aspect;
      this.resize();
    }
    if (Math.abs(this.camera.fov - h.vfov) > 1e-3) {
      this.camera.fov = h.vfov;
      this.camera.updateProjectionMatrix();
    }
    this.updateMesh(frame.vertices, frame.indices, frame.highlight);
    this._boxes = (h.search && h.search.boxes) || [];
    this._boxLabel = (h.search && h.search.query) || '';
    if (frame.jpeg && this.mode !== 'mesh') {
      this.drawVideo(frame.jpeg, h.src_w, h.src_h);
    }
  }

  // ---------------------------------------------------------------- modos --
  setMode(mode) {
    this.mode = mode;
    // REAL+MALLA muestra las dos tarjetas; los otros dos modos ocupan el ancho
    // completo con una sola.
    this.views.classList.toggle('single', mode !== 'real-mesh');
    this.views.classList.toggle('single-b', mode === 'mesh');
    if (mode !== 'mesh' && this.orbitEnabled) this.setOrbit(false);
    this.resize();
  }

  setMeshVisible(visible) {
    this.meshVisible = visible;
  }

  setFar(far) {
    this.farPlane = far;
  }

  // --------------------------------------------------------------- orbita --
  _initOrbit() {
    this.orbit = { theta: 0, phi: 0, radius: 0, target: new THREE.Vector3(0, 0, 2.2) };
    let dragging = false;
    let lastX = 0, lastY = 0;

    const canvas = this.renderer.domElement;
    canvas.addEventListener('pointerdown', (e) => {
      if (!this.orbitEnabled) return;
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging || !this.orbitEnabled) return;
      this.orbit.theta -= (e.clientX - lastX) * 0.006;
      this.orbit.phi = THREE.MathUtils.clamp(
        this.orbit.phi - (e.clientY - lastY) * 0.006, -1.35, 1.35);
      lastX = e.clientX; lastY = e.clientY;
      this._applyOrbit();
    });
    canvas.addEventListener('pointerup', (e) => {
      dragging = false;
      canvas.releasePointerCapture?.(e.pointerId);
    });
    canvas.addEventListener('wheel', (e) => {
      if (!this.orbitEnabled) return;
      e.preventDefault();
      this.orbit.radius = THREE.MathUtils.clamp(
        this.orbit.radius * (1 + Math.sign(e.deltaY) * 0.09), 0.15, 60);
      this._applyOrbit();
    }, { passive: false });
  }

  setOrbit(enabled) {
    this.orbitEnabled = enabled;
    this.views.classList.toggle('orbit', enabled);
    if (enabled) {
      this.orbit.theta = 0;
      this.orbit.phi = 0;
      this.orbit.radius = this.orbit.target.z;
      this._applyOrbit();
    } else {
      this._resetView();
    }
  }

  _applyOrbit() {
    const { theta, phi, radius, target } = this.orbit;
    const x = target.x + radius * Math.cos(phi) * Math.sin(theta);
    const y = target.y + radius * Math.sin(phi);
    const z = target.z - radius * Math.cos(phi) * Math.cos(theta);
    this.camera.position.set(x, -y, -z);
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(target.x, -target.y, -target.z);
  }

  _resetView() {
    this.camera.position.set(0, 0, 0);
    this.camera.rotation.set(0, 0, 0);
    this.camera.up.set(0, 1, 0);
  }

  // ---------------------------------------------------------------- ciclo --
  /** Rectangulo (en px del lienzo) donde cae la imagen dentro de un elemento. */
  _fitRect(el) {
    const host = this.views.getBoundingClientRect();
    const box = el.getBoundingClientRect();
    if (box.width < 4 || box.height < 4) return null;
    const width = Math.min(box.width, box.height * this.aspect);
    const height = width / this.aspect;
    return {
      x: box.left - host.left + (box.width - width) / 2,
      y: box.top - host.top + (box.height - height) / 2,
      w: width,
      h: height,
      // WebGL cuenta la Y desde abajo.
      yGl: host.height - (box.top - host.top + (box.height - height) / 2) - height,
    };
  }

  resize() {
    const host = this.views.getBoundingClientRect();
    if (host.width < 4 || host.height < 4) return;
    this.renderer.setSize(host.width, host.height, false);
    // El video se recorta al mismo rectangulo que la region A, para que la
    // superposicion cuadre pixel a pixel.
    const rect = this._fitRect(this.bodyA);
    if (rect) {
      const style = this.videoCanvas.style;
      style.position = 'absolute';
      style.left = `${rect.x - (this.bodyA.getBoundingClientRect().left - host.left)}px`;
      style.top = `${rect.y - (this.bodyA.getBoundingClientRect().top - host.top)}px`;
      style.width = `${rect.w}px`;
      style.height = `${rect.h}px`;
    }
  }

  _renderRegion(rect, dimensional) {
    if (!rect) return;
    const u = this.wireMaterial.uniforms;
    if (dimensional) {
      // El relleno pinta negro: el mundo queda reducido a su geometria.
      this.fillMaterial.colorWrite = true;
      u.uOpacity.value = 1.0;
      u.uFogAmount.value = 0.7;
      u.uFogNear.value = this.farPlane * 0.3;
      u.uFogFar.value = this.farPlane * 1.5;
    } else {
      // Sobre la imagen real el relleno solo tapa lo que queda detras.
      this.fillMaterial.colorWrite = false;
      u.uOpacity.value = 0.85;
      u.uFogAmount.value = 0.0;
    }

    this.camera.aspect = rect.w / rect.h;
    this.camera.updateProjectionMatrix();
    this.renderer.setViewport(rect.x, rect.yGl, rect.w, rect.h);
    this.renderer.setScissor(rect.x, rect.yGl, rect.w, rect.h);
    this.renderer.render(this.scene, this.camera);
  }

  render() {
    this.renderer.setScissorTest(false);
    this.renderer.clear();
    if (!this.meshVisible || this.mode === 'real') return;

    this.renderer.setScissorTest(true);
    if (this.mode === 'real-mesh') {
      this._renderRegion(this._fitRect(this.bodyA), false);
      this._renderRegion(this._fitRect(this.bodyB), true);
    } else if (this.mode === 'mesh') {
      this._renderRegion(this._fitRect(this.bodyB), true);
    }
    this.renderer.setScissorTest(false);
  }
}
