import json
import os
import time
from collections import deque


class EventLog(object):
    """Append-only JSONL file plus an in-memory tail for the web UI."""

    def __init__(self, path, keep=200):
        self.path = path
        self.recent = deque(maxlen=keep)
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f.readlines()[-keep:]:
                    try:
                        self.recent.append(json.loads(line))
                    except ValueError:
                        pass

    def add(self, event):
        self.recent.append(event)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def today_totals(self):
        """Seconds spent per zone since local midnight, from leave events."""
        lt = time.localtime()
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        totals = {}
        for e in self.recent:
            if e["type"] == "leave" and e["t"] >= midnight:
                totals[e["zone"]] = totals.get(e["zone"], 0) + e["duration"]
        return totals
