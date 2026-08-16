"""Modelo de camara e intrinsecos.

El movil no expone sus intrinsecos por WebAPI, asi que se estiman a partir del
campo de vision declarado por el navegador (o de un FOV por defecto razonable
para camaras traseras de telefono: ~65 grados en horizontal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_HFOV_DEG = 65.0


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = DEFAULT_HFOV_DEG) -> "Intrinsics":
        f = (width * 0.5) / np.tan(np.radians(hfov_deg) * 0.5)
        return cls(width, height, f, f, width * 0.5, height * 0.5)

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """Intrinsecos equivalentes al redimensionar la imagen."""
        sx = width / self.width
        sy = height / self.height
        return Intrinsics(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.height * 0.5 / self.fy)))

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "vfov_deg": self.vfov_deg,
        }


def pixel_rays(intr: Intrinsics) -> np.ndarray:
    """Rayos unitarios-en-z para cada pixel: HxWx3 con z == 1.

    Multiplicar por la profundidad da directamente el punto en el sistema de camara.
    """
    xs = (np.arange(intr.width, dtype=np.float32) - intr.cx) / intr.fx
    ys = (np.arange(intr.height, dtype=np.float32) - intr.cy) / intr.fy
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy, np.ones_like(gx)], axis=-1)
