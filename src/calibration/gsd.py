"""Ground Sampling Distance (GSD) calibration for drone imagery.

The GSD tells us how many centimetres on the ground one image pixel covers. It
is the bridge between pixel-space segmentation areas and physical cm², which the
weight-regression model needs.

Formula (horizontal / nadir view):

    GSD [cm/px] = (sensor_width_mm * altitude_m * 100) / (focal_length_mm * image_width_px)

Derivation: the sensor images a ground swath of width
``altitude_m * sensor_width_mm / focal_length_mm`` metres (similar triangles,
``sensor_width_mm`` and ``focal_length_mm`` in the same unit so they form a pure
ratio). Dividing by ``image_width_px`` gives **metres per pixel**; ×100 → cm/px.

NOTE ON THE SPEC FORMULA: the project brief wrote this as
``(sensor_width_mm * altitude_m) / (focal_length_mm * image_width_px)`` — i.e.
without the ×100. That version is dimensionally metres-per-pixel, not cm/px, and
under-reports the GSD by 100×. This module uses the physically correct form
(a DJI Phantom 4 Pro at 30 m then gives the well-known ~0.82 cm/px, matching
manufacturer tables), and the test-suite pins that expected value.

Multiple flight altitudes (as in the SOIL-Data campaign) simply yield different
GSDs; higher flights → larger GSD → coarser resolution.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSpec:
    """Physical camera parameters needed for GSD.

    ``sensor_width_mm``   : physical width of the imaging sensor.
    ``focal_length_mm``   : lens focal length.
    ``image_width_px``    : horizontal pixel count of the captured image.
    """
    sensor_width_mm: float
    focal_length_mm: float
    image_width_px: int
    name: str = "generic"


# A few common survey-drone presets (approximate manufacturer specs). These are
# reasonable defaults; always confirm against the actual EXIF/spec sheet.
CAMERA_PRESETS = {
    # DJI Phantom 4 Pro / Mavic-class 1" sensor
    "dji_phantom4_pro": CameraSpec(13.2, 8.8, 5472, "DJI Phantom 4 Pro"),
    # DJI Mavic 3 (4/3 sensor)
    "dji_mavic3": CameraSpec(17.3, 12.29, 5280, "DJI Mavic 3"),
    # Generic 1/2.3" consumer drone
    "generic_1_2_3": CameraSpec(6.17, 4.5, 4000, "Generic 1/2.3\""),
}


def gsd_cm_per_px(
    sensor_width_mm: float,
    altitude_m: float,
    focal_length_mm: float,
    image_width_px: int,
) -> float:
    """Return the ground sampling distance in **cm per pixel**.

    Raises ``ValueError`` on non-positive focal length or image width.
    """
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be > 0")
    if image_width_px <= 0:
        raise ValueError("image_width_px must be > 0")
    if sensor_width_mm <= 0 or altitude_m < 0:
        raise ValueError("sensor_width_mm must be > 0 and altitude_m >= 0")
    # ×100 converts metres-per-pixel to centimetres-per-pixel.
    return (sensor_width_mm * altitude_m * 100.0) / (focal_length_mm * image_width_px)


def gsd_from_camera(camera: CameraSpec, altitude_m: float) -> float:
    """GSD (cm/px) for a :class:`CameraSpec` at a given altitude."""
    return gsd_cm_per_px(
        camera.sensor_width_mm, altitude_m,
        camera.focal_length_mm, camera.image_width_px,
    )


def pixel_area_to_cm2(area_px: float, gsd_cm_per_px_value: float) -> float:
    """Convert a pixel area to cm² given the GSD (area scales with GSD²)."""
    if gsd_cm_per_px_value < 0:
        raise ValueError("gsd must be >= 0")
    return area_px * (gsd_cm_per_px_value ** 2)


def pixel_area_to_cm2_from_flight(
    area_px: float,
    sensor_width_mm: float,
    altitude_m: float,
    focal_length_mm: float,
    image_width_px: int,
) -> float:
    """Convenience: pixel area → cm² directly from flight parameters."""
    gsd = gsd_cm_per_px(sensor_width_mm, altitude_m, focal_length_mm, image_width_px)
    return pixel_area_to_cm2(area_px, gsd)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute drone GSD and pixel->cm2 area.")
    p.add_argument("--preset", choices=sorted(CAMERA_PRESETS), default=None,
                   help="use a built-in camera preset")
    p.add_argument("--sensor-width-mm", type=float, default=13.2)
    p.add_argument("--focal-length-mm", type=float, default=8.8)
    p.add_argument("--image-width-px", type=int, default=5472)
    p.add_argument("--altitude-m", type=float, required=True)
    p.add_argument("--area-px", type=float, default=None,
                   help="optional pixel area to convert to cm2")
    return p


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    if args.preset:
        cam = CAMERA_PRESETS[args.preset]
    else:
        cam = CameraSpec(args.sensor_width_mm, args.focal_length_mm, args.image_width_px)
    gsd = gsd_from_camera(cam, args.altitude_m)
    result = {
        "camera": cam.name,
        "altitude_m": args.altitude_m,
        "gsd_cm_per_px": round(gsd, 5),
    }
    if args.area_px is not None:
        result["area_px"] = args.area_px
        result["area_cm2"] = round(pixel_area_to_cm2(args.area_px, gsd), 3)
    import json
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
