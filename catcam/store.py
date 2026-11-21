"""History store: every event goes into SQLite; appear/enter events also keep a JPEG snapshot.

Why SQLite: a write is atomic, so a power cut never leaves half a record (the board loses power
now and then). Standard library only, Python 3.6 compatible (SQLite 3.22: no upsert, no window
functions).

An event is {"t": epoch seconds, "type": appear|enter|leave|lost, "zone": name or None,
"duration": seconds (leave only)}. A "stay" is rebuilt from a leave event: it ended at t and
began at t - duration.
"""
import bisect
import datetime
import json
import os
import re
import sqlite3
import threading
import time

SNAP_TYPES = ("appear", "enter")  # the moments worth a picture
_SNAP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/[0-9A-Za-z_.-]+\.jpg$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL NOT NULL,
    type TEXT NOT NULL,
    zone TEXT,
    duration REAL,
    snap TEXT
);
CREATE INDEX IF NOT EXISTS events_t ON events (t);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def day_range(date):
    """(start, end) epoch seconds of one local calendar day."""
    start = time.mktime(date.timetuple())
    end = time.mktime((date + datetime.timedelta(days=1)).timetuple())
    return start, end


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _event(row):
    e = {"id": row[0], "t": row[1], "type": row[2], "zone": row[3]}
    if row[4] is not None:
        e["duration"] = row[4]
    if row[5]:
        e["snap"] = True
    return e


class Store(object):
    def __init__(self, path, snap_dir=None):
        self.path = path
        self.snap_dir = snap_dir
        self.lock = threading.RLock()
        self._warned = False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if snap_dir:
            os.makedirs(snap_dir, exist_ok=True)
        self.db = self._open()

    # ---- opening (and surviving a damaged file)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10, check_same_thread=False, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")  # events are rare, so a sync per write is affordable
        db.executescript(SCHEMA)
        return db

    def _open(self):
        try:
            db = self._connect()
            row = db.execute("PRAGMA quick_check").fetchone()
            if not row or row[0] != "ok":
                db.close()
                raise sqlite3.DatabaseError("quick_check: %s" % (row[0] if row else "no answer"))
            return db
        except sqlite3.DatabaseError as exc:
            backup = "%s.damaged-%d" % (self.path, int(time.time()))
            print("history database is damaged (%s); keeping it as %s and starting a new one"
                  % (exc, backup), flush=True)
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(self.path + suffix):
                    os.replace(self.path + suffix, backup + suffix)
            return self._connect()

    def close(self):
        with self.lock:
            try:
                self.db.close()
            except sqlite3.Error:
                pass

    # ---- writing

    def add(self, event, jpeg=None):
        """Store an event; returns its id. jpeg (bytes) is kept for appear/enter events."""
        snap = None
        if jpeg and self.snap_dir and event["type"] in SNAP_TYPES:
            snap = self._write_snapshot(event["t"], event["type"], jpeg)
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO events (t, type, zone, duration, snap) VALUES (?, ?, ?, ?, ?)",
                (event["t"], event["type"], event.get("zone"), event.get("duration"), snap))
            return cur.lastrowid

    def _write_snapshot(self, t, kind, jpeg):
        """Write the picture first, the row second: a row never points at a missing file."""
        try:
            lt = time.localtime(t)
            day = time.strftime("%Y-%m-%d", lt)
            stem = "%s-%s" % (time.strftime("%H%M%S", lt), kind)
            folder = os.path.join(self.snap_dir, day)
            os.makedirs(folder, exist_ok=True)
            name, n = stem + ".jpg", 1
            while os.path.exists(os.path.join(folder, name)):
                n += 1
                name = "%s-%d.jpg" % (stem, n)
            full = os.path.join(folder, name)
            tmp = full + ".tmp"
            with open(tmp, "wb") as f:
                f.write(jpeg)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, full)
            return day + "/" + name
        except OSError as exc:
            if not self._warned:  # a full disk must not stop event logging
                self._warned = True
                print("could not save a snapshot:", exc, flush=True)
            return None

    # ---- reading

    def recent(self, n=40):
        """The last n events, oldest first."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id, t, type, zone, duration, snap FROM events ORDER BY id DESC LIMIT ?",
                (n,)).fetchall()
        return [_event(r) for r in reversed(rows)]

    def events_between(self, t0, t1, limit=500):
        with self.lock:
            rows = self.db.execute(
                "SELECT id, t, type, zone, duration, snap FROM events "
                "WHERE t >= ? AND t < ? ORDER BY t ASC, id ASC LIMIT ?", (t0, t1, limit)).fetchall()
        return [_event(r) for r in rows]

    def snapshot_path(self, event_id):
        """Full path of an event's picture, or None. The stored name is checked before use."""
        if not self.snap_dir:
            return None
        with self.lock:
            row = self.db.execute("SELECT snap FROM events WHERE id = ?", (event_id,)).fetchone()
        if not row or not row[0] or not _SNAP_RE.match(row[0]):
            return None
        full = os.path.join(self.snap_dir, *row[0].split("/"))
        return full if os.path.isfile(full) else None

    def stays_between(self, t0, t1):
        """[(zone, start, end)] of stays that overlap [t0, t1]."""
        with self.lock:
            rows = self.db.execute(
                "SELECT zone, t - duration, t FROM events WHERE type = 'leave' AND zone IS NOT NULL "
                "AND duration IS NOT NULL AND t >= ? AND t - duration <= ? ORDER BY t - duration",
                (t0, t1)).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def history(self, days, now=None, open_stay=None):
        """Per local day, oldest first: {"date", "t0", "zones": {name: {"seconds", "visits"}}}.

        open_stay=(zone, since) adds the stay in progress, which is not in the database yet
        (a stay is written when the cat leaves).
        """
        now = time.time() if now is None else now
        today = datetime.date.fromtimestamp(now)
        dates = [today - datetime.timedelta(days=i) for i in range(days - 1, -1, -1)]
        bounds = [day_range(d) for d in dates]
        starts = [b[0] for b in bounds]
        stays = self.stays_between(bounds[0][0], bounds[-1][1])
        if open_stay and open_stay[0] and open_stay[1] is not None:
            stays.append((open_stay[0], open_stay[1], now))
        out = [{"date": d.isoformat(), "t0": b[0], "zones": {}} for d, b in zip(dates, bounds)]
        for zone, s, e in stays:
            i = max(0, bisect.bisect_right(starts, s) - 1)
            first = True
            while i < len(bounds) and bounds[i][0] < e:
                b0, b1 = bounds[i]
                z = out[i]["zones"].setdefault(zone, {"seconds": 0.0, "visits": 0})
                z["seconds"] += _overlap(s, e, b0, b1)
                if first and b0 <= s < b1:
                    z["visits"] += 1  # a visit belongs to the day it began
                first = False
                i += 1
        for day in out:
            for z in day["zones"].values():
                z["seconds"] = round(z["seconds"], 1)
        return out

    def day_detail(self, date, now=None, open_stay=None):
        """One day: seconds per zone per hour, plus the day's events."""
        now = time.time() if now is None else now
        b0, b1 = day_range(date)
        stays = self.stays_between(b0, b1)
        if open_stay and open_stay[0] and open_stay[1] is not None:
            stays.append((open_stay[0], open_stay[1], now))
        hours = {}
        for zone, s, e in stays:
            row = hours.setdefault(zone, [0.0] * 24)
            for h in range(24):
                row[h] += _overlap(s, e, b0 + h * 3600, b0 + (h + 1) * 3600)
        for row in hours.values():
            for h in range(24):
                row[h] = round(row[h], 1)
        return {"date": date.isoformat(), "t0": b0, "hours": hours,
                "events": self.events_between(b0, b1, limit=1000)}

    # ---- housekeeping

    def stats(self):
        with self.lock:
            n_events = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            snaps = [r[0] for r in self.db.execute("SELECT snap FROM events WHERE snap IS NOT NULL")]
        size = 0
        for rel in snaps:
            try:
                size += os.path.getsize(os.path.join(self.snap_dir, *rel.split("/")))
            except (OSError, TypeError, AttributeError):
                pass
        try:
            db_bytes = os.path.getsize(self.path)
        except OSError:
            db_bytes = 0
        return {"events": n_events, "snapshots": len(snaps), "snapshot_bytes": size, "db_bytes": db_bytes}

    def purge(self, keep_days=30, max_bytes=500 * 1024 * 1024, now=None):
        """Delete old snapshot files. Event rows stay (they are tiny); only the picture link is cleared.

        Returns how many files were removed.
        """
        if not self.snap_dir:
            return 0
        now = time.time() if now is None else now
        cutoff = now - keep_days * 86400
        with self.lock:
            rows = self.db.execute(
                "SELECT id, snap, t FROM events WHERE snap IS NOT NULL ORDER BY t ASC").fetchall()
        drop, keep, kept_bytes = [], [], 0
        for event_id, rel, t in rows:
            try:
                size = os.path.getsize(os.path.join(self.snap_dir, *rel.split("/")))
            except OSError:
                size = 0
            if t < cutoff:
                drop.append((event_id, rel))
            else:
                keep.append((event_id, rel, size))
                kept_bytes += size
        i = 0
        while kept_bytes > max_bytes and i < len(keep):  # still too big: oldest first
            drop.append((keep[i][0], keep[i][1]))
            kept_bytes -= keep[i][2]
            i += 1
        if drop:
            with self.lock:
                self.db.execute("BEGIN")
                self.db.executemany("UPDATE events SET snap = NULL WHERE id = ?", [(d[0],) for d in drop])
                self.db.execute("COMMIT")
            for _, rel in drop:
                self._remove(rel)
        return len(drop) + self._sweep_orphans(now)

    def _remove(self, rel):
        try:
            os.remove(os.path.join(self.snap_dir, *rel.split("/")))
        except OSError:
            pass

    def _sweep_orphans(self, now):
        """Remove pictures no row points at (a crash between the two writes) and empty day folders."""
        with self.lock:
            known = set(r[0] for r in self.db.execute("SELECT snap FROM events WHERE snap IS NOT NULL"))
        removed = 0
        for day in sorted(os.listdir(self.snap_dir)):
            folder = os.path.join(self.snap_dir, day)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                full = os.path.join(folder, name)
                try:
                    old = now - os.path.getmtime(full) > 3600  # a fresh file may still get its row
                except OSError:
                    continue
                if old and (day + "/" + name) not in known:
                    try:
                        os.remove(full)
                        removed += 1
                    except OSError:
                        pass
            try:
                os.rmdir(folder)  # only succeeds when empty
            except OSError:
                pass
        return removed

    def import_jsonl(self, path):
        """One-time import of the old events.jsonl (corrupt lines are skipped). Returns rows added."""
        if not path or not os.path.isfile(path):
            return 0
        key = "jsonl_imported:" + os.path.abspath(path)
        with self.lock:
            if self.db.execute("SELECT 1 FROM meta WHERE key = ?", (key,)).fetchone():
                return 0
            rows = []
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        rows.append((float(e["t"]), str(e["type"]), e.get("zone"),
                                     None if e.get("duration") is None else float(e["duration"])))
                    except (ValueError, KeyError, TypeError, AttributeError):
                        continue
            self.db.execute("BEGIN")
            try:
                self.db.executemany(
                    "INSERT INTO events (t, type, zone, duration) VALUES (?, ?, ?, ?)", rows)
                self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(len(rows))))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
            return len(rows)
