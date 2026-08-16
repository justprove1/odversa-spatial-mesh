# ODVERSA SPATIAL MESH

Convierte lo que ve la cámara del móvil en geometría 3D, en tiempo real, en el
ordenador.

El móvil **solo abre la cámara**. Todo el trabajo (profundidad, nube de puntos,
malla) ocurre en el Mac, y la malla se ve en el visor del Mac.

```
móvil (cámara) ──Wi-Fi/WebSocket──▶ Mac ──▶ profundidad ──▶ nube de puntos
                                              └──▶ malla ──▶ visor (3 modos)
```

## Arrancar

```bash
cd ~/odversa-spatial-mesh && ./run.sh
```

Imprime dos direcciones:

- **Visor** (en el Mac): `https://localhost:10000/`
- **Cámara** (en el móvil): `https://192.168.1.14:10000/phone`

El visor muestra un **código QR** mientras espera: escanéalo con el móvil y te
lleva directo a la página de captura, sin teclear la IP.

En el móvil, el navegador avisará de que el certificado no es de fiar: es el
autofirmado que genera `run.sh` para esta máquina. Acepta la excepción, pulsa
**ABRIR CÁMARA** y da permiso. La malla aparece en el Mac.

> La página `/phone` va **en el teléfono**. Abierta en el ordenador funciona,
> pero capturaría la webcam del portátil, que no es la idea.

> HTTPS no es opcional: los navegadores solo dan acceso a la cámara en contexto
> seguro. Sin certificado, el móvil ni siquiera ofrece el permiso.

**Por cable USB**: activa "Compartir conexión" / USB tethering en el móvil. La
ruta pasa a ser el cable en lugar del Wi-Fi, con menos latencia y sin cambiar
nada del código; solo cambia la IP que hay que teclear.

## Los tres modos

| Modo | Qué se ve |
|---|---|
| **REAL + MALLA** | La imagen de la cámara con la malla fina superpuesta y alineada |
| **SOLO MALLA** | Fondo negro y solo líneas: el mundo reducido a geometría |
| **SOLO REAL** | Solo la cámara, sin malla |

En **SOLO MALLA** se activa **ÓRBITA LIBRE**: arrastra para girar alrededor de
la geometría y comprobar que es 3D de verdad, no un dibujo sobre la imagen.

## Controles

Empieza por los tres presets de **CALIDAD**, que mueven a la vez los parámetros
que interactúan entre sí:

Medido de extremo a extremo en el M5, con la máquina en uso normal:

| Preset | Malla | Red | fps | Latencia | Triángulos |
|---|---|---|---|---|---|
| RÁPIDO | 160 | 266 | ~21 | ~55 ms | ~1.200 |
| EQUILIBRADO | 224 | 308 | ~19 | ~105 ms | ~2.100 |
| DETALLE | 288 | 448 | ~9 | ~180 ms | ~3.400 |

Los números bailan según lo ocupada que esté la máquina: la misma prueba da
entre 12,7 y 16,7 fps en EQUILIBRADO según lo que haya de fondo. Están medidos
con uso normal, no en un equipo en reposo.

**El deslizador de resolución no baja de 266 a propósito.** Por debajo la red no
se degrada, colapsa: el error de forma medido salta del 19% al 136%.

Subir la densidad sin subir la resolución de la red solo dibuja más triángulos
sobre la misma información borrosa; por eso van juntos.

- **Densidad de malla** — vértices de la rejilla (176 = 176×132). Por encima de
  ~200 el alambre deja de leerse como líneas y quien cuenta la forma es el
  sombreado.
- **Resolución de profundidad** — lado equivalente de la entrada de la red; el
  alto y el ancho reales respetan el aspecto de la cámara.
- **Detalle mínimo** — el relieve más pequeño que la malla se molesta en
  representar, en cm a 3 m (escala con la distancia). Es **el** control de
  "cuánto detecta": con 3 cm, un teclado sobre una mesa no genera ni un vértice.
  Medido sobre una escena real, con la malla ya asentada:

  | Detalle mínimo | Triángulos | Coste |
  |---|---|---|
  | 3,0 cm | 1.691 | 14 ms |
  | **1,2 cm** | **2.757** | **22 ms** |
  | 0,8 cm | 3.279 | 27 ms |
  | 0,5 cm | 3.916 | 31 ms |

- **Relieve fino** — realza el relieve que la red aplana. Sube esto para que los
  objetos pequeños se despeguen de la superficie donde están apoyados.
- **Suavizado** — filtro bilateral: aplana superficies sin fundir los bordes.
- **Estabilidad temporal** — mezcla con el frame anterior; sube esto si la malla
  tiembla.
- **Corte de bordes** — cuánto se corta en los saltos de profundidad. Bajarlo
  corta más (menos "cortinas" entre objeto y fondo, más agujeros).
- **Escala** — la profundidad monocular no tiene escala; este control fija a qué
  distancia está el objeto más cercano y con eso se calibra todo lo demás.
- **Alcance** — distancia máxima reconstruida.
- **Reiniciar** / **Guardar mapa 3D** (PLY binario en `maps/`, abre en Blender).

## Búsqueda de objetos

Escribe en la barra superior lo que buscas — «vitrina», «silla», «extintor» — y
cuando la cámara lo vea se resalta **en naranja** sobre la malla (en las dos
vistas) y con una caja etiquetada sobre el vídeo. Borrar el texto (o Escape)
apaga la búsqueda.

Cómo funciona (`object_search.py`):

- **OWL-ViT** (vocabulario abierto): compara cualquier texto contra regiones de
  la imagen. No está limitado a una lista de clases.
- Corre en **su propio hilo a ~1-2 Hz** (~120-330 ms por consulta en CPU); el
  pipeline de malla nunca lo espera, así que los fps no se resienten. Sin
  consulta activa, coste cero.
- El español va por **doble vía**: la consulta tal cual y traducida por un
  diccionario interno de ~70 objetos comunes (CLIP entiende mucho mejor el
  inglés). Se queda la mejor puntuación.
- La caja 2D no basta —encierra también la pared de detrás—, así que se estima
  la **profundidad del objeto** (mediana del núcleo de la caja) y solo se
  resaltan los vértices a esa distancia: el objeto se separa de su fondo.
- El resaltado viaja como un byte por vértice en el mismo mensaje binario que
  la malla, y el shader mezcla hacia naranja.

El modelo (612 MB) se descarga la primera vez a `models/owlvit-base-patch32/`.

## Arquitectura

Un módulo por responsabilidad, sustituibles de uno en uno:

```
server/
  pipeline.py                    encadena las etapas, mide tiempos
  core/camera.py, core/types.py  intrínsecos y tipos compartidos
  modules/
    mobile_camera_input.py       recibe frames + IMU, cola de un hueco
    video_streaming.py           transporte y difusión al visor
    depth_estimation.py          interfaz + calibración a métrico
    depth_backends/              onnx_depth_anything.py · stub.py
    camera_tracking.py           pose 6DoF: odometría RGB-D + IMU
    point_cloud.py               retroproyección y filtrado
    spatial_memory.py            mapa persistente [esqueleto, fase 2]
    mesh_reconstruction.py       malla triangular
    object_search.py             busqueda de objetos por texto (OWL-ViT)
    mesh_optimization.py         suavizado, decimación, presupuesto
    map_io.py                    guardado PLY/OBJ
  net/protocol.py                protocolo binario
web/
  phone/                         captura en el móvil
  viewer/                        Renderer, UI, red (Three.js incluido)
```

Cambiar de modelo de profundidad = implementar `DepthBackend` (dos métodos) y
registrarlo en `build_backend`. Nada más del sistema se entera.

## Estado

**Funciona ahora (fase 1)**: cámara del móvil → profundidad → nube de puntos →
malla triangular → los tres modos, a ~14 fps de punta a punta con ~50 ms de
latencia (M5, malla 112×84, red 308 px).

**Tracking de cámara (hecho)**: `CameraTracking` estima la pose 6DoF por
odometría RGB-D — esquinas arrastradas con flujo óptico Lucas-Kanade,
verificación inversa y `solvePnPRansac` contra los puntos 3D del frame
anterior — con el giroscopio del móvil como semilla de rotación y como respaldo
cuando la escena se queda sin textura. Cuesta **~2,4 ms por frame** y el panel
del visor dibuja el recorrido en planta.

Medido con `tools.check_tracking`, que recorre una habitación sintética con la
pose conocida de antemano: sobre 72 cm de recorrido y 17° de giro, el error
final es de **5,9 cm (8,3%)** y **0,6°**, con la cámara parada la deriva es de
0,0 cm en 12 frames, y sobre una pared lisa el estado nunca sube a GOOD.

Es odometría, no SLAM: no hay cierre de bucle, así que el error se acumula con
el recorrido. Y la escala la hereda de la calibración de profundidad
(`near_m`/`far_m`), de modo que es consistente pero no métrica absoluta.

**Pendiente (fase 2)**: `SpatialMemory` sigue siendo un esqueleto con la
interfaz cerrada y el diseño (TSDF con hash de vóxeles) documentado dentro del
fichero. Hasta que se rellene, la geometría es **del frame actual**, no un mapa
acumulado: la cámara ya sabe dónde está, pero lo que quedó atrás no se conserva.

## Herramientas

```bash
# Comprobación del pipeline sin red ni navegador
.venv/bin/python -m tools.check_pipeline

# Odometría contra un recorrido de pose conocida (error medido, no impresión)
.venv/bin/python -m tools.check_tracking

# Ver los tres modos sobre una foto, sin móvil
.venv/bin/python -m tools.render_preview foto.jpg salida.png [densidad]

# Móvil simulado: alimenta el servidor desde una foto o una escena sintética
.venv/bin/python -m tools.fake_phone --image foto.jpg --fps 15
```

## Decisiones técnicas

**Depth Anything V2 Small por ONNX Runtime.** Mejor relación calidad/latencia
hoy para tiempo real. Detalle que importa mucho: el grafo exportado trae alto y
ancho dinámicos, y eso impide casi toda la optimización — 338 ms por frame.
Fijando las dimensiones libres antes de crear la sesión baja a 43 ms. Medido en
este M5, con las formas ya fijas el EP de CPU gana a CoreML (el grafo ViT no se
compila entero), así que el defecto es CPU; `ODVERSA_DEPTH_PROVIDER=coreml` lo
cambia.

**Calibración de profundidad por ajuste afín en disparidad**, anclando los
percentiles 10 y 90 (no los extremos) y suavizándolos con EMA. Con los extremos,
un reflejo reescalaba la habitación entera.

**Entrada de red con el aspecto de la cámara.** Meter una imagen 4:3 en una
entrada cuadrada la aplasta horizontalmente antes de estimar la profundidad y la
estira después: deforma la geometría y se lleva por delante el detalle fino. Se
eligen alto y ancho múltiplos de 14 (el tamaño de parche del modelo) que
conservan el aspecto y el área equivalente, así que no cuesta más.

**Filtro guiado por la imagen** (He et al., implementado con filtros de caja, sin
dependencias). La red trabaja a menos resolución que la cámara y entrega bordes
redondeados; la imagen sí tiene el contorno nítido. Ajustando en cada ventana
`profundidad ≈ a·imagen + b`, los saltos de profundidad se enganchan al borde
real del objeto. Toda la cadena de refinado (guiado + bilateral + realce) cuesta
**1 ms**, así que es prácticamente gratis.

**Límite honesto del detalle**: esto mejora la silueta y la forma general de los
objetos, no su microrrelieve. Las teclas de un teclado están a milímetros de la
superficie y ningún modelo monocular de este tamaño las resuelve. Un teclado se
reconocerá por su contorno y su bulto sobre la mesa, no por sus teclas.

**El corte de triángulos usa la torsión de la celda**, no la diferencia de
profundidad. Un suelo visto en escorzo tiene un gradiente enorme por píxel y sin
embargo es la superficie más fiable de la escena; con el criterio de diferencia
se borraba el suelo entero. La torsión vale cero en cualquier plano, inclinado o
no: medido sobre escenas reales, su mediana es 0,0005 y los bordes auténticos
pasan de 0,6.

**Malla adaptativa por Delaunay** (`adaptive_mesh.py`). La rejilla uniforme
gastaba los mismos triángulos en una pared lisa que en una planta. Ahora se
eligen **puntos** con densidad variable —pocos donde la superficie se parece a la
interpolación de sus esquinas, muchos alrededor de un canto o una planta— y se
triangulan con Delaunay.

Delaunay es lo que hace que "cuadre matemáticamente": produce por construcción
una triangulación válida del conjunto de puntos, sin huecos, sin solapes y sin
vértices colgando en mitad de una arista. Eso elimina de golpe el problema de las
grietas entre zonas de distinta densidad.

Medido sobre la misma escena, a densidad 224:

| | Triángulos | kB/frame | Coste |
|---|---|---|---|
| Uniforme | 46.206 | 823 | 1,0 ms |
| Cuadtree | 8.241 | 150 | 2,9 ms |
| **Delaunay** | **1.200** | **23** | 12,2 ms |

Delaunay cuesta 11 ms más de CPU y ahorra el **97%** de los triángulos. Sale a
cuenta de sobra: esos milisegundos van en el hilo de geometría, que se solapa
con la inferencia, mientras que los triángulos se pagan en red y en GPU en cada
frame.

**Estabilidad temporal: los vértices se conservan entre frames.** Recalculando
los puntos desde cero en cada frame, dos imágenes casi idénticas producían
conjuntos distintos y la malla hervía, lo que hace ilegible la forma. Ahora se
parte de los puntos del frame anterior que siguen siendo válidos y se corrige
solo lo que cambió. Medido sobre una secuencia con la cámara moviéndose:
**87% de los vértices sobreviven de un frame al siguiente**.

De paso sale más barato, porque partiendo de un conjunto ya bueno basta una
ronda en vez de cinco: el mallado bajó de 12,2 a **5,1 ms**.

El tope de vértices se ata a la densidad (`points_per_column`), así que el
deslizador de la interfaz sigue controlando el detalle. Cuando se supera, se
podan los de menor curvatura: es un criterio determinista, así que dos frames
parecidos podan los mismos puntos y no reaparece el parpadeo.

Dos cosas que costó afinar:

- **La semilla tiene que ser escasa.** Sembrar una retícula gruesa para llegar
  antes a la densidad final parecía buena idea, pero los puntos de retícula que
  nadie necesita no se eliminan luego y el suelo volvía a salir cuadriculado.
  Partiendo de las esquinas del encuadre y los contornos de la escena, cada
  punto que aparece está ahí porque el error lo pidió.
- **Los candidatos se submuestrean** (uno de cada dos píxeles). Localizar el
  triángulo que contiene cada candidato es la parte cara; sin esto, una escena
  muy ruidosa disparaba el mallado a 57 ms y se comía el margen del hilo.
- **El interior del triángulo se decide por mayoría.** Exigir que TODOS los
  puntos de muestra cayeran en zona válida era demasiado severo con triángulos
  grandes: uno que cubre medio suelo se caía entero por rozar un hueco de dos
  píxeles, y de ahí venían los claros en la malla.

Y lo que eso permite es lo importante: la densidad de malla puede subir mucho.
224 adaptativa sale mucho más barata que 128 uniforme, así que el detalle sube y
el coste baja a la vez.

> Queda también `AdaptiveMesher` (cuadtree restringido, con regla 2:1 y puntos
> medios en las aristas) seleccionable con `mesh_style = "quadtree"`. Da
> triángulos alineados a cuadrícula: se le nota el origen, pero es 2,4× más
> rápido de construir.

**Pipeline en dos hilos encadenados.** La red de profundidad se lleva el 86% del
tiempo (58,8 ms de 68) y todo lo demás suma 9,3 ms. En un solo hilo esos tiempos
se suman; separando profundidad y geometría en dos etapas, mientras una trabaja
con el frame N la otra ya va por el N+1, y el ritmo pasa a ser el del cuello de
botella en vez de la suma. Medido: de 11,3 a 14,9 fps, un 32% más.

**Ritmo por acuse de recibo.** El servidor confirma cada frame consumido y el
móvil no manda más de dos sin confirmar. Sin eso el teléfono emite a su ritmo,
los frames se apilan en el buffer del socket, y el sistema mantiene los fps pero
cada imagen llega con varios frames de retraso — que al mover el móvil se nota
mucho más que un fps menos. Medido: la latencia bajó de 197 a ~80 ms.

Ojo con un detalle que costó encontrar: hay que acusar también los frames que se
**descartan** en el hueco de entrada. Si solo se acusan los consumidos, el
contador del móvil se atasca y deja de enviar (daba 15 fps donde tocaban 20).

**Optimizaciones medidas y descartadas** (por si vuelve la tentación):

- *Modelos cuantizados*: int8 tarda 129 ms frente a los 57 del fp32 en este
  ARM — más del doble. `model_quantized` va igual que el original. No compensa.
- *Ajustar los hilos de ONNX Runtime a mano*: cualquier valor fijo (2, 4, 6, 8,
  10) es peor que dejarlo en automático.
- *Varias inferencias en paralelo*: 20,8 fps agregados frente a 17,3, pero la
  latencia sube de 58 a 192 ms. Para algo que sigue el movimiento de la mano,
  ese cambio empeora la sensación aunque el contador diga otra cosa.
- *Limitar los hilos de OpenCV* para que no le quiten núcleos a la inferencia:
  da igual (60,2 ms con 10 hilos, 60,8 con 1). No hay tal competencia.

**IA de superficies universales** (`surface_detection.py`). Detecta lo que toda
escena tiene —suelo, paredes, mesas, techo— sin otra red neuronal: esas
superficies comparten estructura geométrica universal (son planos y su
orientación los delata), y eso se detecta con RANSAC en ~1,5 ms. El visor
muestra lo detectado como etiquetas en el panel INFORMACIÓN.

La detección hace doble servicio: además de etiquetar, **aquieta**. La
profundidad de los píxeles de un plano se proyecta sobre el plano (mezcla
continua, no umbral: con umbral el borde parpadeaba y metía temblor en vez de
quitarlo), y los parámetros del plano se suavizan entre frames.

**Los planos detectados vacían la malla.** La detección de superficies no solo
etiqueta: alimenta al mallador. El interior de un plano confirmado (suelo,
pared, mesa) casi no recibe vértices —la geometría ya la cuenta el plano, no
hace falta triangulación fina— y su **contorno** entra como vértices propios,
así que las aristas de la malla se alinean con la junta suelo-pared o el canto
de la mesa en vez de cruzarlos. Es lo que da el aspecto de la referencia:
triángulos enormes en el suelo, nítidos en los bordes.

Medido sobre una escena de geometría conocida (suelo + pared + mesa):
**5.455 → 368 triángulos (−93%)** y el mallado de 27,6 → 3,7 ms. La poda es
determinista (máscara de planos + retícula fija), así que no mete parpadeo.

**Quietud, medida con pulso de mano simulado** (desplazamientos de 1-3 px):

| | Temblor global | Zona lejana |
|---|---|---|
| Antes | 11,1 cm/frame | 61,5 cm/frame |
| Ahora | **5,6 cm/frame** | **22,7 cm/frame** |

Dos hallazgos que importan más que los números:

- El temblor NO estaba donde parecía: el suelo ya estaba quieto (0,8 cm). El
  temblor vivía en la **zona lejana**, donde la red amplifica cualquier
  variación (allí 1/z es minúsculo y un pelo de disparidad son 60 cm). El
  filtro temporal ahora refuerza la memoria con la distancia: una pared a 15 m
  no puede cambiar rápido en la realidad.
- El refinado guiado por imagen iba DESPUÉS del filtro temporal, así que la
  imagen temblando re-metía el temblor ya filtrado. La quietud final (`settle`)
  se aplica al final de la cadena, sobre lo que de verdad se malla.

**La malla es continua: `far_m` no es un cuchillo.** Durante un tiempo la malla
tenía un agujero permanente donde caía el fondo de la escena, y la causa era una
contradicción interna: la calibración ancla el percentil 10 de la disparidad a
`far_m`, y la máscara de validez descartaba todo lo que llegaba a `far_m` — o
sea que por construcción se tiraba el 10% más lejano de *cualquier* escena, la
pared del fondo de todas las habitaciones. Ahora la profundidad puede llegar
hasta 2,5× `far_m` antes de darse por perdida, y la malla envuelve la escena
entera como en un escaneo de verdad.

**Oclusión por doble pasada.** La misma geometría se dibuja dos veces: una opaca
que solo escribe profundidad (o negro, en SOLO MALLA) y encima el alambre. Sin
eso se verían las líneas de la pared del fondo a través de los objetos.

**Sombreado por normales en el alambre.** Desde el punto de vista de la propia
cámara, una malla sacada de un mapa de profundidad se proyecta como una rejilla
regular y la forma no se aprecia. Modulando el brillo con la inclinación de cada
polígono, suelo, paredes y objetos se separan solos. Es información geométrica:
ni texturas, ni colores de objeto, ni iluminación simulada.
