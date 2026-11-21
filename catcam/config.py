"""config.json: read it, and write it back safely (the web page edits the zones)."""
import json
import os
import shutil
import tempfile
import threading


def dumps(cfg):
    """Readable enough to edit by hand: one top-level key per line, one zone per line."""
    lines = []
    for key, value in cfg.items():
        if key == "zones" and isinstance(value, list) and value:
            inner = ",\n".join("    " + json.dumps(z, ensure_ascii=False) for z in value)
            lines.append('  "zones": [\n%s\n  ]' % inner)
        else:
            lines.append("  %s: %s" % (json.dumps(key, ensure_ascii=False),
                                       json.dumps(value, ensure_ascii=False)))
    return "{\n" + ",\n".join(lines) + "\n}\n"


class Config(object):
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def load(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def save_zones(self, zones):
        """Replace only "zones"; everything else in the file is kept as it was."""
        with self.lock:
            cfg = self.load()
            cfg["zones"] = zones
            self._write(cfg)

    def _write(self, cfg):
        """Write to a temp file, then rename: a power cut leaves the old file or the new one."""
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(dumps(cfg))
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(self.path):
                shutil.copyfile(self.path, self.path + ".bak")  # one step back, just in case
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
