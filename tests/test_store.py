"""The SQLite history store: events, pictures, day totals, clean-up, damaged files."""
import datetime
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from catcam.store import Store


def ts(y, mo, d, h=0, mi=0, s=0):
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


def leave(t, zone, duration):
    return {"t": t, "type": "leave", "zone": zone, "duration": duration}


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.snaps = os.path.join(self.dir, "snaps")
        self.store = Store(os.path.join(self.dir, "catcam.db"), self.snaps)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class EventsTest(StoreTestCase):
    def test_add_and_recent_keep_order_and_fields(self):
        a = self.store.add({"t": 1.0, "type": "appear", "zone": None})
        b = self.store.add({"t": 2.0, "type": "leave", "zone": "sofa", "duration": 1.5})
        self.assertLess(a, b)
        got = self.store.recent(10)
        self.assertEqual([e["type"] for e in got], ["appear", "leave"])
        self.assertNotIn("duration", got[0])
        self.assertEqual(got[1]["duration"], 1.5)
        self.assertEqual([e["id"] for e in self.store.recent(1)], [b])

    def test_survives_reopen(self):
        self.store.add({"t": 1.0, "type": "enter", "zone": "sofa"})
        self.store.close()
        self.store = Store(os.path.join(self.dir, "catcam.db"), self.snaps)
        self.assertEqual(len(self.store.recent()), 1)

    def test_several_threads_at_once(self):
        def writer(n):
            for i in range(50):
                self.store.add({"t": float(n * 100 + i), "type": "appear", "zone": None})

        def reader():
            for _ in range(50):
                self.store.recent(10)
                self.store.history(1)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(self.store.recent(1000)), 200)


class SnapshotTest(StoreTestCase):
    def test_pictures_only_for_appear_and_enter(self):
        t = ts(2026, 11, 16, 9, 30)
        entered = self.store.add({"t": t, "type": "enter", "zone": "sofa"}, b"\xff\xd8jpeg")
        left = self.store.add(leave(t + 5, "sofa", 5), b"\xff\xd8jpeg")
        path = self.store.snapshot_path(entered)
        self.assertTrue(path and os.path.isfile(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"\xff\xd8jpeg")
        self.assertIsNone(self.store.snapshot_path(left))
        self.assertTrue(self.store.recent()[0]["snap"])
        self.assertNotIn("snap", self.store.recent()[1])

    def test_same_second_does_not_overwrite(self):
        t = ts(2026, 11, 16, 9, 30)
        a = self.store.add({"t": t, "type": "enter", "zone": "a"}, b"one")
        b = self.store.add({"t": t, "type": "enter", "zone": "b"}, b"two")
        self.assertNotEqual(self.store.snapshot_path(a), self.store.snapshot_path(b))

    def test_a_bad_stored_name_is_never_served(self):
        i = self.store.add({"t": 1.0, "type": "enter", "zone": "a"}, b"x")
        self.store.db.execute("UPDATE events SET snap = ? WHERE id = ?", ("../../etc/passwd", i))
        self.assertIsNone(self.store.snapshot_path(i))

    def test_a_failing_disk_does_not_lose_the_event(self):
        shutil.rmtree(self.snaps)
        open(self.snaps, "w").close()  # a file where the folder should be: every write fails
        i = self.store.add({"t": ts(2026, 11, 16, 9), "type": "enter", "zone": "a"}, b"x")
        self.assertIsNone(self.store.snapshot_path(i))
        self.assertEqual(len(self.store.recent()), 1)

    def test_stats(self):
        self.store.add({"t": ts(2026, 11, 16, 9), "type": "enter", "zone": "a"}, b"x" * 10)
        s = self.store.stats()
        self.assertEqual((s["events"], s["snapshots"], s["snapshot_bytes"]), (1, 1, 10))


class HistoryTest(StoreTestCase):
    def test_a_stay_over_midnight_is_split_and_counted_once(self):
        self.store.add(leave(ts(2026, 11, 16, 0, 30), "sofa", 3600))  # there since 23:30
        days = self.store.history(2, now=ts(2026, 11, 16, 12))
        self.assertEqual([d["date"] for d in days], ["2026-11-15", "2026-11-16"])
        self.assertEqual(days[0]["zones"]["sofa"], {"seconds": 1800.0, "visits": 1})
        self.assertEqual(days[1]["zones"]["sofa"], {"seconds": 1800.0, "visits": 0})

    def test_the_stay_in_progress_counts_and_empty_days_are_empty(self):
        now = ts(2026, 11, 16, 12)
        days = self.store.history(3, now=now, open_stay=("bowl", now - 600))
        self.assertEqual(days[0]["zones"], {})
        self.assertEqual(days[2]["zones"]["bowl"], {"seconds": 600.0, "visits": 1})

    def test_totals_add_up_over_several_stays(self):
        def at(h, m):
            return ts(2026, 11, 16, h, m)
        self.store.add(leave(at(8, 10), "sofa", 600))
        self.store.add(leave(at(9, 0), "sofa", 300))
        self.store.add(leave(at(9, 30), "bowl", 60))
        self.store.add(leave(ts(2026, 11, 10, 9, 30), "bowl", 60))  # another day: not in today's numbers
        day = self.store.history(1, now=at(12, 0))[0]
        self.assertEqual(day["zones"]["sofa"], {"seconds": 900.0, "visits": 2})
        self.assertEqual(day["zones"]["bowl"], {"seconds": 60.0, "visits": 1})

    def test_a_long_stay_covers_every_day_it_touches(self):
        self.store.add(leave(ts(2026, 11, 16, 6, 0), "sofa", 3 * 86400))  # since the 13th, 06:00
        days = self.store.history(5, now=ts(2026, 11, 16, 12))  # the 12th to the 16th
        seconds = [d["zones"].get("sofa", {}).get("seconds", 0) for d in days]
        self.assertEqual(seconds, [0, 18 * 3600.0, 86400.0, 86400.0, 6 * 3600.0])
        visits = [d["zones"].get("sofa", {}).get("visits", 0) for d in days]
        self.assertEqual(visits, [0, 1, 0, 0, 0])  # the visit belongs to the day it began

    def test_day_detail_has_hours_and_events(self):
        end = ts(2026, 11, 16, 10, 30)
        self.store.add({"t": end - 5400, "type": "enter", "zone": "sofa"})
        self.store.add(leave(end, "sofa", 5400))
        detail = self.store.day_detail(datetime.date(2026, 11, 16), now=ts(2026, 11, 16, 12))
        hours = detail["hours"]["sofa"]
        self.assertEqual((hours[8], hours[9], hours[10], sum(hours)), (0, 3600.0, 1800.0, 5400.0))
        self.assertEqual([e["type"] for e in detail["events"]], ["enter", "leave"])


class PurgeTest(StoreTestCase):
    def snap(self, t, data=b"x" * 1000):
        return self.store.add({"t": t, "type": "enter", "zone": "a"}, data)

    def test_old_pictures_go_but_the_events_stay(self):
        now = ts(2026, 11, 16, 12)
        old, new = self.snap(now - 40 * 86400), self.snap(now - 86400)
        old_path = self.store.snapshot_path(old)
        self.assertEqual(self.store.purge(keep_days=30, now=now), 1)
        self.assertFalse(os.path.exists(old_path))
        self.assertIsNone(self.store.snapshot_path(old))
        self.assertIsNotNone(self.store.snapshot_path(new))
        self.assertEqual(len(self.store.recent()), 2)

    def test_size_cap_drops_the_oldest_first(self):
        now = ts(2026, 11, 16, 12)
        ids = [self.snap(now - (3 - i) * 3600) for i in range(3)]  # oldest first
        self.store.purge(keep_days=30, max_bytes=2500, now=now)
        self.assertIsNone(self.store.snapshot_path(ids[0]))
        self.assertIsNotNone(self.store.snapshot_path(ids[1]))
        self.assertIsNotNone(self.store.snapshot_path(ids[2]))

    def test_pictures_without_a_row_are_swept_but_fresh_ones_wait(self):
        folder = os.path.join(self.snaps, "2026-11-01")
        os.makedirs(folder)
        stray = os.path.join(folder, "stray.jpg")
        fresh = os.path.join(folder, "fresh.jpg")
        for p in (stray, fresh):
            with open(p, "wb") as f:
                f.write(b"x")
        past = time.time() - 7200
        os.utime(stray, (past, past))
        self.store.purge(now=time.time())
        self.assertFalse(os.path.exists(stray))
        self.assertTrue(os.path.exists(fresh))  # may still be waiting for its row


class DamageTest(unittest.TestCase):
    def test_a_garbage_file_is_set_aside_and_a_new_database_starts(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "catcam.db")
        with open(path, "wb") as f:
            f.write(b"this is not a database" * 100)
        store = Store(path, None)
        try:
            self.assertEqual(store.recent(), [])
            store.add({"t": 1.0, "type": "appear", "zone": None})
            self.assertEqual(len(store.recent()), 1)
            self.assertTrue([n for n in os.listdir(d) if ".damaged-" in n])
        finally:
            store.close()
            shutil.rmtree(d, ignore_errors=True)


class ImportTest(StoreTestCase):
    def test_old_jsonl_is_imported_once_and_bad_lines_are_skipped(self):
        path = os.path.join(self.dir, "events.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"t": 1000.5, "type": "enter", "zone": "沙发"}\n')
            f.write('{"t": 1010.5, "type": "leave", "zone": "沙发", "duration": 10.0}\n')
            f.write('{"t": 1020.5, "typ')  # cut short by a power loss
            f.write("\nnot json\n[1, 2]\n")
        self.assertEqual(self.store.import_jsonl(path), 2)
        self.assertEqual(self.store.import_jsonl(path), 0)
        self.assertEqual([e["type"] for e in self.store.recent()], ["enter", "leave"])
        self.assertEqual(self.store.recent()[1]["duration"], 10.0)
        self.assertEqual(self.store.import_jsonl(os.path.join(self.dir, "missing.jsonl")), 0)


if __name__ == "__main__":
    unittest.main()
