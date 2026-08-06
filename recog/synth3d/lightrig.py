"""
recog.synth3d.lightrig - where an off-axis light goes and which way it faces.

No bpy import, so the one piece of this that is easy to get silently wrong -
the aiming - can be tested outside Blender.

Why it is a separate module
---------------------------
`camera_softbox`, the only rig kind the dataset had, puts the light at the
camera. Under a top-down camera that means the light is directly overhead and
coaxial with the view, so shading is flat and every cast shadow falls exactly
behind the object that cast it, hidden from the camera. The reference factory
photos have the opposite: light arriving from overhead-and-to-one-side, with
visible directional shadows whose direction and length change from photo to
photo. That is a strong cue and the training set had none of it.

Getting an area light to aim at a point is four lines of trigonometry, and a
90-degree error in any of them is close to invisible: the light still lights
*something*, the render is still plausible, and only a careful look at the
shadow direction across a whole sweep would show that the rig is not doing
what its config says. Hence a testable module rather than four lines inlined
into a bpy-only file.

Angles are DEGREES at the boundary (that is what the config carries) and
metres for distance. `elevation` is measured up from the horizontal plane the
parts lie on: 90 is straight overhead, 0 is level with them.
"""

from __future__ import annotations

import math
from typing import Tuple


# A Blender AREA (and SUN, and SPOT) light emits along its own -Z axis, so
# aiming one is a question of finding an Euler rotation that carries -Z onto
# the direction from the lamp to what it should light.
EMIT_AXIS = (0.0, 0.0, -1.0)


def off_axis_placement(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Position and Euler-XYZ rotation for a light aimed at `target`.

    Returns ``(location, rotation_euler)``, both ready to assign to a Blender
    object; the rotation is in RADIANS because that is what `rotation_euler`
    wants, while the arguments are in degrees because that is what the config
    carries.

    The derivation, since it is short and the alternative is trusting it:
    Blender composes ``rotation_euler`` as ``Rz @ Ry @ Rx``. Leaving ``ry`` at
    zero and applying ``Rz(b) @ Rx(a)`` to the emission axis ``(0, 0, -1)``
    gives ``(-sin b sin a, cos b sin a, -cos a)``. The lamp sits at

        (d cos(el) cos(az),  d cos(el) sin(az),  d sin(el))

    relative to the target, so it must emit along the negative of that unit
    vector, ``(-cos el cos az, -cos el sin az, -sin el)``. Matching the two
    term by term gives ``cos a = sin el`` and ``(sin b, cos b) =
    (cos az, -sin az)`` - that is, ``a = 90 - el`` and ``b = az + 90``, which
    is what this function returns. `test_synth3d` re-derives the same result
    from independently written rotation matrices rather than reusing this
    formula, so the two have to agree by being right, not by construction.
    """
    if not 0.0 < elevation_deg <= 90.0:
        raise ValueError(
            f"elevation must be in (0, 90] degrees, got {elevation_deg}. "
            f"At or below 0 the lamp is level with the parts or under the "
            f"backdrop plane, which lights nothing the camera can see.")
    if distance <= 0.0:
        raise ValueError(f"distance must be positive, got {distance}")

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    horiz = distance * math.cos(el)

    location = (target[0] + horiz * math.cos(az),
                target[1] + horiz * math.sin(az),
                target[2] + distance * math.sin(el))
    rotation = (math.radians(90.0 - elevation_deg),
                0.0,
                math.radians(azimuth_deg + 90.0))
    return location, rotation


def shadow_direction(azimuth_deg: float) -> Tuple[float, float]:
    """
    Unit vector, in the ground plane, that a shadow is cast ALONG.

    The lamp is at azimuth `azimuth_deg`, so shadows point the opposite way.
    Used only for reporting and testing - nothing in the render depends on
    it - but it is the property the whole off-axis rig exists to vary, so it
    is worth being able to state rather than eyeball.
    """
    az = math.radians(azimuth_deg)
    return (-math.cos(az), -math.sin(az))


__all__ = ["EMIT_AXIS", "off_axis_placement", "shadow_direction"]
