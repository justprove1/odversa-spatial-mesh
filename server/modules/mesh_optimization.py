"""MeshOptimization - limpieza y simplificacion de la malla.

Todo son operaciones vectorizadas con numpy, sin dependencias pesadas, para que
puedan correr dentro del bucle de tiempo real:

  * `laplacian_smooth`  -> quita el temblor de alta frecuencia sin mover planos.
  * `cluster_decimate`  -> agrupacion de vertices por celda: baja poligonos donde
    sobran (zonas lejanas o superficies planas) manteniendo la forma.
  * `drop_degenerate`   -> elimina triangulos de area casi nula y aristas rotas.
  * `enforce_budget`    -> techo duro de triangulos, simplificando hasta cumplirlo.
"""

from __future__ import annotations

import numpy as np


def drop_degenerate(vertices: np.ndarray, faces: np.ndarray, min_area: float = 1e-8
                    ) -> tuple[np.ndarray, np.ndarray]:
    if faces.size == 0:
        return vertices, faces
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep = area2 > min_area
    keep &= (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return vertices, faces[keep]


def laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 1,
                     strength: float = 0.5) -> np.ndarray:
    """Promedia cada vertice con sus vecinos de arista."""
    if iterations <= 0 or faces.size == 0:
        return vertices
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    src, dst = edges[:, 0], edges[:, 1]
    n = vertices.shape[0]
    counts = np.bincount(src, minlength=n).astype(np.float32)
    counts[counts == 0] = 1.0

    out = vertices.astype(np.float32)
    for _ in range(iterations):
        acc = np.zeros_like(out)
        for axis in range(3):
            acc[:, axis] = np.bincount(src, weights=out[dst, axis], minlength=n)
        avg = acc / counts[:, None]
        out = (1.0 - strength) * out + strength * avg
    return out


def cluster_decimate(vertices: np.ndarray, faces: np.ndarray, cell: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Simplificacion por agrupacion espacial.

    Cada celda de lado `cell` colapsa en un unico vertice (su centroide) y los
    triangulos que quedan con vertices repetidos desaparecen. Es O(n), no da la
    calidad de una decimacion por metrica cuadrica, pero cuesta microsegundos y
    en una malla de escaneo la diferencia visible es minima.
    """
    if faces.size == 0 or cell <= 0:
        return vertices, faces
    keys = np.floor(vertices / cell).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.astype(np.int64).ravel()
    n_clusters = counts.shape[0]

    acc = np.zeros((n_clusters, 3), np.float64)
    for axis in range(3):
        acc[:, axis] = np.bincount(inverse, weights=vertices[:, axis], minlength=n_clusters)
    new_vertices = (acc / counts[:, None]).astype(np.float32)

    new_faces = inverse[faces]
    ok = ((new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2])
          & (new_faces[:, 0] != new_faces[:, 2]))
    new_faces = new_faces[ok].astype(np.uint32)

    # Reindexar solo los vertices supervivientes.
    if new_faces.size:
        used = np.zeros(n_clusters, bool)
        used[new_faces.ravel()] = True
        remap = np.full(n_clusters, -1, np.int64)
        remap[used] = np.arange(int(used.sum()))
        new_faces = remap[new_faces].astype(np.uint32)
        new_vertices = new_vertices[used]
    else:
        new_vertices = np.zeros((0, 3), np.float32)
    return new_vertices, new_faces


def enforce_budget(vertices: np.ndarray, faces: np.ndarray, max_tris: int,
                   base_cell: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Simplifica en pasos hasta bajar del presupuesto. Devuelve el nivel usado."""
    lod = 0
    cell = base_cell
    while faces.shape[0] > max_tris and lod < 4:
        cell *= 1.8
        vertices, faces = cluster_decimate(vertices, faces, cell)
        lod += 1
    return vertices, faces, lod
