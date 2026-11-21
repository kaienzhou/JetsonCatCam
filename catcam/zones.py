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

    def to_json(self):
        return {"name": self.name, "points": [[round(x, 4), round(y, 4)] for x, y in self.points]}


MAX_ZONES = 12
MAX_POINTS = 24
MIN_AREA = 1e-4  # of the whole picture; anything smaller is a slip of the finger


def polygon_area(points):
    """Absolute area (shoelace formula) in normalized coordinates."""
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _segments_meet(p1, p2, p3, p4):
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return ((d1 == 0 and _on_segment(p3, p4, p1)) or (d2 == 0 and _on_segment(p3, p4, p2)) or
            (d3 == 0 and _on_segment(p1, p2, p3)) or (d4 == 0 and _on_segment(p1, p2, p4)))


def self_intersects(points):
    """True if two edges that are not neighbours cross or touch (a bow-tie shape)."""
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # neighbours share a corner
            if _segments_meet(a, b, points[j], points[(j + 1) % n]):
                return True
    return False


def parse_zones(raw):
    """Check zones that came from the web page. Returns [Zone]; raises ValueError.

    The message is shown to the person editing, so it says what to fix.
    """
    if not isinstance(raw, list):
        raise ValueError("区域列表的格式不对")
    if len(raw) > MAX_ZONES:
        raise ValueError("最多 %d 个区域" % MAX_ZONES)
    zones, names = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("区域的格式不对")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("每个区域都要有名字")
        name = name.strip()
        if len(name) > 24:
            raise ValueError("区域名字最多 24 个字：" + name[:8] + "…")
        if name in names:
            raise ValueError("区域名字重复了：" + name)
        names.add(name)
        pts = item.get("points")
        if not isinstance(pts, list):
            raise ValueError("区域「%s」缺少顶点" % name)
        clean = []
        for p in pts:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError("区域「%s」的顶点格式不对" % name)
            for v in p:
                # NaN and infinity fail the range test too
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                    raise ValueError("区域「%s」的顶点超出了画面" % name)
            q = (round(float(p[0]), 4), round(float(p[1]), 4))
            if not clean or clean[-1] != q:  # drop a doubled tap
                clean.append(q)
        if len(clean) > 1 and clean[0] == clean[-1]:
            clean.pop()
        if not 3 <= len(clean) <= MAX_POINTS:
            raise ValueError("区域「%s」需要 3 到 %d 个顶点" % (name, MAX_POINTS))
        if polygon_area(clean) < MIN_AREA:
            raise ValueError("区域「%s」太小了" % name)
        if self_intersects(clean):
            raise ValueError("区域「%s」的边交叉了，请把顶点理顺" % name)
        zones.append(Zone(name, clean))
    return zones


def watch_window(zones, aspect=1.0, margin=0.05, min_size=0.3, max_area=0.72):
    """The part of the picture the detector has to look at, so that it can look at it bigger.

    Returns (left, top, right, bottom) normalized, or None when the zones need (nearly) the
    whole picture. The window is the bounding box of all zones plus `margin`, at least
    `min_size` of the picture wide (zooming in further than that only shows less), grown to
    `aspect` and moved inside the picture. aspect is width / height of the window in
    normalized units: the network input's aspect divided by the picture's (1.0 when both
    are 4:3). A window covering more than `max_area` of the picture is not worth it.
    """
    pts = [p for z in zones for p in z.points]
    if not pts or aspect <= 0:
        return None
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w = max(x1 - x0 + 2 * margin, min_size)
    h = max(y1 - y0 + 2 * margin, min_size / aspect)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w > 1.0:
        w, h = 1.0, 1.0 / aspect
    if h > 1.0:
        w, h = aspect, 1.0
    if w * h > max_area:
        return None
    left = min(max(cx - w / 2.0, 0.0), 1.0 - w)
    top = min(max(cy - h / 2.0, 0.0), 1.0 - h)
    return (left, top, left + w, top + h)


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

    def set_zones(self, now, zones):
        """Swap the zones while running. A stay in progress ends first. Returns the events."""
        events = []
        self._leave(now, events)
        self.zones = zones
        self._cand, self._cand_count = None, 0
        return events

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
