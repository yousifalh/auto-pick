# Spec #3 (realism) — design decisions taken 2026-08-09

Recorded before the spec is written, so the reasoning is not re-derived.

## Target camera: rig-realistic only

The deployed sensor is a **fixed overhead machine-vision camera**, not a phone.
Near top-down, tilt roughly 0-10 degrees off vertical, no roll, fixed working
distance around 400 mm, machine-vision lens FOV.

The generator models **that**, not the handheld phone geometry of the existing
real photographs.

### The consequence, which is the important part

The original framing of spec #3 was "match the phone: 3024x4032 portrait,
handheld tilt, because the real photos look like that." That is now **wrong**.
The real photos are a convenience sample, not the target domain.

Two things follow:

1. **The new photograph set must be shot near-overhead at ~400 mm**, or the
   50-100 annotated images will measure transfer to a domain the robot will
   never operate in. The annotation protocol has been changed accordingly.
2. The existing handheld obliques in `recog/realtest/images/` stay useful as a
   **stress set** but must be kept in a separate directory and scored
   separately. Mixing them blurs the single measurement this exercise exists to
   make.

This also *reduces* scope: wide tilt and roll sampling is now explicitly out of
scope. The variation budget moves to fill level, cartridge type, lighting and
clutter — things the robot will actually encounter.

## Scene clutter: all of it

The blue jig plate with a proper material; loose 18650 cells on the bench
outside the cartridge; tools and cables; other cartridges partly in frame; and
general incidental background clutter.

Loose cells outside the bay are worth calling out as the highest-value item:
they force the model to distinguish an in-bay cell from an out-of-bay one,
which is exactly the discrimination the placement logic depends on and which
the current generator never exercises.

## Still open

- Whether the orthographic camera is retired outright or kept as a variant.
  A near-top-down perspective camera at 400 mm is close to orthographic, so
  the difference may be small — worth measuring rather than assuming.
- Lighting model for a fixed rig: a real machine-vision setup usually has
  controlled illumination, which argues for *less* lighting variation than
  spec #4 assumes. These two specs may pull against each other and the
  tension should be resolved when #4 is written.
