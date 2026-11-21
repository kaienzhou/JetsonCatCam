"""Access code, session cookie and the brake on guessing."""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from catcam.auth import ALPHABET, Auth


class AuthTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_code_is_made_once_and_kept(self):
        a = Auth(self.dir)
        self.assertTrue(a.created)
        self.assertEqual(len(a.code), 10)
        self.assertTrue(set(a.code) <= set(ALPHABET))
        b = Auth(self.dir)
        self.assertFalse(b.created)
        self.assertEqual(a.code, b.code)

    @unittest.skipUnless(os.name == "posix", "file modes")
    def test_the_files_are_private(self):
        Auth(self.dir)
        for name in ("access_code", "secret"):
            self.assertEqual(os.stat(os.path.join(self.dir, name)).st_mode & 0o077, 0)

    def test_a_code_in_the_config_wins(self):
        a = Auth(self.dir, code="my own code")
        self.assertFalse(a.created)
        self.assertEqual(a.code, "my own code")
        self.assertTrue(a.check_code("my own code"))
        self.assertFalse(a.check_code("my own cod"))
        with self.assertRaises(ValueError):
            a.rotate()

    def test_check_code_rejects_junk(self):
        a = Auth(self.dir)
        for junk in ("", None, 12345, a.code + "x", a.code[:-1]):
            self.assertFalse(a.check_code(junk))
        self.assertTrue(a.check_code(a.code))

    def test_cookie_roundtrip_and_tampering(self):
        a = Auth(self.dir)
        c = a.new_cookie()
        self.assertTrue(a.check_cookie(c))
        self.assertNotEqual(a.new_cookie(), c)
        nonce, sig = c.split(".")
        for bad in ("", "abc", c + ".x", nonce, "." + sig, nonce + ".", nonce + "." + "0" * 64,
                    "0" * 32 + "." + sig):
            self.assertFalse(a.check_cookie(bad), bad)

    def test_a_cookie_from_another_device_is_refused(self):
        other = tempfile.mkdtemp()
        try:
            self.assertFalse(Auth(self.dir).check_cookie(Auth(other).new_cookie()))
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_a_new_code_logs_everybody_out(self):
        a = Auth(self.dir)
        cookie, old = a.new_cookie(), a.code
        new = a.rotate()
        self.assertNotEqual(old, new)
        self.assertFalse(a.check_cookie(cookie))
        self.assertTrue(a.check_cookie(a.new_cookie()))
        self.assertEqual(Auth(self.dir).code, new)

    def test_a_code_changed_by_another_process_is_noticed(self):
        # `python3 -m catcam new-code` runs next to the service, which must follow without a restart
        running = Auth(self.dir)
        cookie = running.new_cookie()
        time.sleep(0.02)  # file times are coarse
        new = Auth(self.dir).rotate()
        self.assertEqual(running.code, new)
        self.assertTrue(running.check_code(new))
        self.assertFalse(running.check_cookie(cookie))

    def test_wrong_codes_lock_an_address_for_a_while(self):
        a = Auth(self.dir, max_fail=3, window=60, lockout=0.3)
        for _ in range(2):
            a.record_failure("1.2.3.4")
            self.assertTrue(a.login_allowed("1.2.3.4")[0])
        a.record_failure("1.2.3.4")
        ok, wait = a.login_allowed("1.2.3.4")
        self.assertFalse(ok)
        self.assertGreaterEqual(wait, 1)
        self.assertTrue(a.login_allowed("5.6.7.8")[0])  # other addresses are not affected
        time.sleep(0.4)
        self.assertTrue(a.login_allowed("1.2.3.4")[0])

    def test_a_good_login_clears_the_failures(self):
        a = Auth(self.dir, max_fail=3, window=60, lockout=60)
        a.record_failure("1.2.3.4")
        a.record_failure("1.2.3.4")
        a.record_success("1.2.3.4")
        a.record_failure("1.2.3.4")
        self.assertTrue(a.login_allowed("1.2.3.4")[0])

    def test_switched_off_lets_everybody_in(self):
        a = Auth(self.dir, enabled=False)
        self.assertFalse(a.created)
        self.assertTrue(a.check_cookie(""))
        self.assertTrue(a.check_code("anything"))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "access_code")))


if __name__ == "__main__":
    unittest.main()
