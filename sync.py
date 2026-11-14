"""Barrier state contract for the HIP source in a16w16.py.

All launches sharing a workspace must be serialized on its bound HIP stream.
Control [arrival_count, generation] is device-owned and never reset in flight.
Each barrier resets arrivals before release-publishing the next generation.
The counter's acquire/release chain collects every CTA's stores. Generation
equality, rather than ordering, remains correct across uint32 wrap-around.

Fault [failed, site, cta, generation] is sticky. A spin-iteration limit provides
a finite wait; it is deliberately not advertised as a wall-clock timeout.
On failure the workspace/controller must be discarded after the worker stops.
"""

CONTROL_WORDS = 2
ERROR_WORDS = 4
SITE_NORM = 1
SITE_MID = 2
