"""Pictures saved on purpose, for later: "the cat was right there and was not recognised".

Each sample is the camera picture exactly as the detector got it (no drawings, full size) plus a
small JSON file with what the detector said at that moment. They are what you need to find out why
a cat was missed, or to train a model on your own cat and your own room.

Files are called 20261121-181530.jpg and 20261121-181530.json in data/samples/. The oldest ones
are deleted when there are more than `keep`.
"""
import json
import os
import re
import time

NAME = re.compile(r"^(\d{8}-\d{6}(?:-\d{1,3})?)\.(jpg|json)$")


class Samples(object):
    def __init__(self, directory, keep=300):
        self.dir = directory
        self.keep = keep

    def _names(self):
        try:
            found = [m.group(1) for m in map(NAME.match, os.listdir(self.dir)) if m and m.group(2) == "jpg"]
        except OSError:
            return []
        return sorted(found)

    def count(self):
        return len(self._names())

    def latest(self, n=20):
        """Names of the newest n samples, newest first."""
        return self._names()[::-1][:n]

    def path(self, filename):
        """Full path of a sample file by its name (name.jpg or name.json), or None."""
        if not NAME.match(filename or ""):
            return None
        path = os.path.join(self.dir, filename)
        return path if os.path.isfile(path) else None

    def save(self, jpeg, info, now=None):
        """Store a picture (JPEG bytes) and what the detector said. Returns the name."""
        os.makedirs(self.dir, exist_ok=True)
        now = time.time() if now is None else now
        base = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        name, n = base, 0
        while os.path.exists(os.path.join(self.dir, name + ".jpg")):
            n += 1
            name = "%s-%d" % (base, n)
        info = dict(info, t=now, name=name)
        # the json first: a picture without one is still a sample, a json without a picture is not
        self._write(os.path.join(self.dir, name + ".json"), json.dumps(info, ensure_ascii=False, indent=1).encode("utf-8"))
        self._write(os.path.join(self.dir, name + ".jpg"), jpeg)
        self._prune()
        return name

    @staticmethod
    def _write(path, data):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _prune(self):
        names = self._names()
        for name in names[:max(0, len(names) - self.keep)]:
            for ext in (".jpg", ".json"):
                try:
                    os.remove(os.path.join(self.dir, name + ext))
                except OSError:
                    pass
