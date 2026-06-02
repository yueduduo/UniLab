from __future__ import annotations

import numpy as np

from unilab.visualization.ball_velocity_hud import render_ball_velocity_hud_pixels


def test_render_ball_velocity_hud_pixels_shape() -> None:
    pixels = render_ball_velocity_hud_pixels(np.array([1.0, -0.5, 0.0]))
    assert pixels.shape == (72, 420, 3)
    assert pixels.dtype == np.uint8
