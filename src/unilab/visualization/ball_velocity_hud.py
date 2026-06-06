"""Motrix overlay HUD for live ball linear velocity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from motrixsim.render import Layout
from PIL import Image, ImageDraw

HUD_WIDTH = 420
HUD_HEIGHT = 72


def render_ball_velocity_hud_pixels(vel: np.ndarray) -> np.ndarray:
    vel = np.asarray(vel, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(vel))
    speed_xy = float(np.linalg.norm(vel[:2]))
    text = (
        "Ball velocity (m/s)\n"
        f"vx {vel[0]:+.3f}   vy {vel[1]:+.3f}   vz {vel[2]:+.3f}\n"
        f"|v| {speed:.3f}     |v_xy| {speed_xy:.3f}"
    )
    image = Image.new("RGB", (HUD_WIDTH, HUD_HEIGHT), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), text, fill=(235, 235, 235))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


@dataclass
class BallVelocityHud:
    render: Any
    _widget: object | None = None

    def update(self, vel: np.ndarray) -> None:
        pixels = render_ball_velocity_hud_pixels(vel)
        image = self.render.create_image(pixels)
        if self._widget is None:
            self._widget = self.render.widgets.create_image_widget(
                image,
                layout=Layout(left=10, top=10, width=HUD_WIDTH, height=HUD_HEIGHT),
            )
            return
        self._widget.update(image=image)
