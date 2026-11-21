"""Access code + signed session cookie. Standard library only (Python 3.6).

The web page and the camera stream are private: you log in once with an access code and
the browser keeps a cookie. The code is random and lives in data/access_code (mode 600),
or you set your own in config.json ("auth": {"code": "..."}).

The cookie is "<random nonce>.<HMAC(nonce)>". The HMAC key is derived from a secret file and the
access code, so changing the code logs everybody out. There are no timestamps in it on
purpose: the board has no battery clock and its time can be wrong for a minute after boot.
"""
import hashlib
import hmac
import os
import secrets
import threading
import time

# no look-alike characters (i, l, o, 0, 1): the code gets typed on a phone
ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def new_code(length=10):
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except (IOError, OSError):
        return ""


def _write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class Auth(object):
    COOKIE = "catcam_session"
    MAX_AGE = 180 * 24 * 3600  # how long the browser keeps the cookie

    def __init__(self, data_dir, code="", enabled=True, max_fail=5, window=60.0, lockout=60.0):
        self.enabled = enabled
        self.fixed_code = code or ""
        self.max_fail = max_fail
        self.window = window
        self.lockout = lockout
        self.lock = threading.Lock()
        self._fails = {}      # ip -> [monotonic times of recent failures]
        self._locked = {}     # ip -> monotonic time the lock ends
        self.created = False  # True when this start generated a new code

        os.makedirs(data_dir, exist_ok=True)
        self.code_path = os.path.join(data_dir, "access_code")
        self.secret_path = os.path.join(data_dir, "secret")

        self.secret = _read(self.secret_path)
        if not self.secret:
            self.secret = secrets.token_hex(32)
            _write_private(self.secret_path, self.secret)
        self._file_code = _read(self.code_path)
        self._mtime = self._stat_mtime()
        if not self.fixed_code and not self._file_code and enabled:
            self._file_code = new_code()
            _write_private(self.code_path, self._file_code)
            self._mtime = self._stat_mtime()
            self.created = True

    def _stat_mtime(self):
        try:
            return os.stat(self.code_path).st_mtime_ns
        except OSError:
            return None

    @property
    def code(self):
        """The current code. `python3 -m catcam new-code` changes the file while the service is
        running, so the file is read again whenever it changes."""
        if self.fixed_code:
            return self.fixed_code
        mtime = self._stat_mtime()
        if mtime != self._mtime:
            self._file_code = _read(self.code_path)
            self._mtime = mtime
        return self._file_code

    def rotate(self):
        """Make a new random code (everybody is logged out). Not possible when config.json sets one."""
        if self.fixed_code:
            raise ValueError("the access code is set in config.json; change it there")
        self._file_code = new_code()
        _write_private(self.code_path, self._file_code)
        self._mtime = self._stat_mtime()
        return self._file_code

    def check_code(self, code):
        if not self.enabled:
            return True
        mine = self.code
        if not isinstance(code, str) or not mine:
            return False
        return hmac.compare_digest(code.encode("utf-8"), mine.encode("utf-8"))

    def _key(self, code):
        return hmac.new(self.secret.encode("utf-8"), code.encode("utf-8"), hashlib.sha256).digest()

    def new_cookie(self):
        nonce = secrets.token_hex(16)
        sig = hmac.new(self._key(self.code), nonce.encode("ascii"), hashlib.sha256).hexdigest()
        return nonce + "." + sig

    def check_cookie(self, value):
        if not self.enabled:
            return True
        mine = self.code
        if not value or value.count(".") != 1 or not mine:
            return False
        nonce, sig = value.split(".")
        if len(nonce) != 32 or len(sig) != 64:
            return False
        want = hmac.new(self._key(mine), nonce.encode("ascii", "replace"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(want.encode("ascii"), sig.encode("ascii", "replace"))

    # --- brute-force brake: a few wrong codes from one address and it is locked for a while

    def login_allowed(self, ip):
        """(True, 0) or (False, seconds to wait)."""
        now = time.monotonic()
        with self.lock:
            until = self._locked.get(ip, 0.0)
            if until > now:
                return False, int(until - now) + 1
            self._locked.pop(ip, None)
            return True, 0

    def record_failure(self, ip):
        now = time.monotonic()
        with self.lock:
            times = [t for t in self._fails.get(ip, []) if now - t < self.window]
            times.append(now)
            if len(times) >= self.max_fail:
                self._locked[ip] = now + self.lockout
                times = []
            self._fails[ip] = times
            if len(self._fails) > 512:  # bounded memory on a long-running device
                for key in [k for k, v in self._fails.items() if not v or now - v[-1] >= self.window]:
                    self._fails.pop(key, None)
                for key in [k for k, v in self._locked.items() if v <= now]:
                    self._locked.pop(key, None)

    def record_success(self, ip):
        with self.lock:
            self._fails.pop(ip, None)
            self._locked.pop(ip, None)
