"""Unit tests for GSD calibration (Module 3)."""
import math

import pytest

from src.calibration.gsd import (
    CAMERA_PRESETS,
    CameraSpec,
    gsd_cm_per_px,
    gsd_from_camera,
    pixel_area_to_cm2,
    pixel_area_to_cm2_from_flight,
)


def test_gsd_matches_known_phantom4_value():
    # DJI Phantom 4 Pro at 30 m -> ~0.82 cm/px (well-known reference value).
    gsd = gsd_cm_per_px(sensor_width_mm=13.2, altitude_m=30,
                        focal_length_mm=8.8, image_width_px=5472)
    assert math.isclose(gsd, 0.8224, rel_tol=1e-3)


def test_gsd_scales_linearly_with_altitude():
    cam = CAMERA_PRESETS["dji_phantom4_pro"]
    g30 = gsd_from_camera(cam, 30)
    g60 = gsd_from_camera(cam, 60)
    assert math.isclose(g60, 2 * g30, rel_tol=1e-9)
    # zero altitude -> zero GSD
    assert gsd_from_camera(cam, 0) == 0.0


def test_pixel_area_to_cm2_scales_with_gsd_squared():
    assert pixel_area_to_cm2(1000, 0.5) == 250.0        # 1000 * 0.25
    assert pixel_area_to_cm2(1000, 1.0) == 1000.0
    # end-to-end from flight params equals the two-step computation
    gsd = gsd_cm_per_px(13.2, 30, 8.8, 5472)
    direct = pixel_area_to_cm2_from_flight(5000, 13.2, 30, 8.8, 5472)
    assert math.isclose(direct, 5000 * gsd ** 2, rel_tol=1e-12)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        gsd_cm_per_px(13.2, 30, 0, 5472)          # zero focal length
    with pytest.raises(ValueError):
        gsd_cm_per_px(13.2, 30, 8.8, 0)           # zero image width
    with pytest.raises(ValueError):
        pixel_area_to_cm2(100, -1)                # negative gsd


def test_presets_are_wellformed():
    for key, cam in CAMERA_PRESETS.items():
        assert isinstance(cam, CameraSpec)
        assert cam.sensor_width_mm > 0 and cam.focal_length_mm > 0
        assert cam.image_width_px > 0
