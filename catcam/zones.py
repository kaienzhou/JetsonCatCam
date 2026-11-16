"""Zone geometry and the debounced enter/leave state machine.

Pure standard library so it runs (and is unit-tested) off the Jetson.
Coordinates are normalized to [0, 1] on both axes.
"""


def point_in_polygon(x, y, poly):
    """Ray-casting test. poly is a list of (x, y)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class Zone(object):
    def __init__(self, name, points):
        self.name = name
        self.points = [(float(px), float(py)) for px, py in points]

    def contains(self, x, y):
        return point_in_polygon(x, y, self.points)


class ZoneTracker(object):
    """Turns a noisy per-frame cat position into stable events.

    update(now, point) -> list of event dicts:
      appear  cat came into view
      enter   confirmed in a zone (needs enter_frames consecutive frames)
      leave   left a zone (to another zone, to open floor, or out of view);
              carries the stay length in "duration"
      lost    cat out of view for lost_seconds
    """

    def __init__(self, zones, enter_frames=5, leave_seconds=3.0, lost_seconds=5.0):
        self.zones = zones
        self.enter_frames = enter_frames
        self.leave_seconds = leave_seconds
        self.lost_seconds = lost_seconds
        self.present = False
        self.current = None          # confirmed zone name, or None
        self.entered_at = None
        self.last_seen = None
        self.last_in_current = None
        self._cand = None
        self._cand_count = 0

    def _zone_at(self, point):
        for z in self.zones:
            if z.contains(point[0], point[1]):
                return z.name
        return None

    def _leave(self, now, events):
        if self.current is not None:
            events.append({"t": now, "type": "leave", "zone": self.current,
                           "duration": round(now - self.entered_at, 1)})
            self.current = None
            self.entered_at = None

    def update(self, now, point):
        events = []
        if point is None:
            if self.present and now - self.last_seen >= self.lost_seconds:
                self._leave(now, events)
                self.present = False
                self._cand, self._cand_count = None, 0
                events.append({"t": now, "type": "lost", "zone": None})
            return events

        if not self.present:
            self.present = True
            events.append({"t": now, "type": "appear", "zone": None})
        self.last_seen = now
        raw = self._zone_at(point)

        if raw == self.current:
            self._cand, self._cand_count = None, 0
            if raw is not None:
                self.last_in_current = now
            return events

        if raw is None:
            # open floor: only leave after the grace period
            if self.current is not None and now - self.last_in_current >= self.leave_seconds:
                self._leave(now, events)
            self._cand, self._cand_count = None, 0
            return events

        if raw == self._cand:
            self._cand_count += 1
        else:
            self._cand, self._cand_count = raw, 1
        if self._cand_count >= self.enter_frames:
            self._leave(now, events)
            self.current = raw
            self.entered_at = now
            self.last_in_current = now
            self._cand, self._cand_count = None, 0
            events.append({"t": now, "type": "enter", "zone": raw})
        return events
