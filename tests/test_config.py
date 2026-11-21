"""Saving zones back into config.json without touching anything else."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from catcam.config import Config

ZONES = [{"name": "沙发", "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]},
         {"name": "食盆", "points": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]]}]


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "config.json")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"port": 8080, "camera": {"source": "csi://0"}, "note": "手写的", "zones": []}, f)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_save_zones_keeps_everything_else_and_the_key_order(self):
        cfg = Config(self.path)
        cfg.save_zones(ZONES)
        data = cfg.load()
        self.assertEqual(data["port"], 8080)
        self.assertEqual(data["camera"], {"source": "csi://0"})
        self.assertEqual(data["note"], "手写的")
        self.assertEqual(data["zones"], ZONES)
        self.assertEqual(list(data.keys()), ["port", "camera", "note", "zones"])

    def test_the_file_stays_readable_by_humans(self):
        Config(self.path).save_zones(ZONES)
        with open(self.path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("沙发", text)  # not \u-escaped
        zone_lines = [l for l in text.splitlines() if '"name"' in l]
        self.assertEqual(len(zone_lines), 2)  # one zone per line
        self.assertTrue(text.endswith("}\n"))

    def test_the_previous_version_is_kept_as_bak(self):
        Config(self.path).save_zones(ZONES)
        with open(self.path + ".bak", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["zones"], [])

    def test_a_broken_file_is_not_overwritten(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        with self.assertRaises(ValueError):
            Config(self.path).save_zones(ZONES)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "{ this is not json")

    def test_no_temp_files_are_left_behind(self):
        Config(self.path).save_zones(ZONES)
        self.assertEqual(sorted(os.listdir(self.dir)), ["config.json", "config.json.bak"])


if __name__ == "__main__":
    unittest.main()
