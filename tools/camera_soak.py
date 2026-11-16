"""Camera soak test: capture from a CSI camera for a while and report dropouts.

    python3 tools/camera_soak.py --seconds 120            # camera only
    python3 tools/camera_soak.py --seconds 120 --detect   # camera + detection load

Prints a line every 10 s (frames, failures, longest gap between frames) and a summary.
Compare with `dmesg | grep -c cdma_handle_timeout` before and after: ISP timeouts there
mean the sensor stopped delivering frames (loose ribbon, power dip, ...).
"""
import argparse
import time

try:
    import jetson.inference as inference
    import jetson.utils as utils
except ImportError:
    import jetson_inference as inference
    import jetson_utils as utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--width", type=int, default=1640)
    ap.add_argument("--height", type=int, default=1232)
    ap.add_argument("--rate", type=int, default=30)
    ap.add_argument("--detect", action="store_true", help="also run ssd-mobilenet-v2 on every frame")
    args = ap.parse_args()

    net = inference.detectNet("ssd-mobilenet-v2", threshold=0.4) if args.detect else None
    cam = utils.videoSource("csi://0", argv=[
        "--input-width=%d" % args.width, "--input-height=%d" % args.height,
        "--input-rate=%d" % args.rate, "--input-flip=rotate-180"])

    start = time.time()
    last = start
    window = start
    frames = fails = win_frames = win_fails = 0
    max_gap = win_gap = 0.0
    while True:
        now = time.time()
        if now - start >= args.seconds:
            break
        if now - window >= 10:  # report on wall-clock time, so a dead camera still shows up
            print("t=%3ds frames=%4d (%.1f fps) fails=%d max_gap=%.2fs" % (
                now - start, win_frames, win_frames / (now - window), win_fails, win_gap), flush=True)
            window, win_frames, win_fails, win_gap = now, 0, 0, 0.0
        try:
            img = cam.Capture(timeout=1000)
        except Exception:
            fails += 1
            win_fails += 1
            continue
        now = time.time()
        gap = now - last
        last = now
        max_gap = max(max_gap, gap)
        win_gap = max(win_gap, gap)
        frames += 1
        win_frames += 1
        if net is not None:
            net.Detect(img, overlay="none")
    total = time.time() - start
    print("SUMMARY %s: %d frames in %.0fs (%.1f fps), %d failed captures, longest gap %.2fs" % (
        "camera+detect" if net else "camera only", frames, total, frames / total, fails, max_gap), flush=True)


if __name__ == "__main__":
    main()
