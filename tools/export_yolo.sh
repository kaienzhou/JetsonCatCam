#!/usr/bin/env bash
# Export YOLOv8n to ONNX with a static input shape that TensorRT 8.0 (JetPack 4.6) accepts.
# Run this on a normal PC (Linux / WSL) with Python 3.9+; it uses its own virtualenv.
#
#     bash tools/export_yolo.sh                  # 384x512 480x640 576x768 768x1024
#     SIZES="768x1024" bash tools/export_yolo.sh # only the sizes you name
#
# Then copy the .onnx files to the Jetson (models/ in this project) and build the engines
# there with tools/build_engine.sh (an engine only works on the GPU and TensorRT version
# it was built with).
#
# Input size is height x width, a multiple of 32; 3:4 sizes match the 4:3 frames of the
# camera (no padding). A bigger input finds smaller / more distant cats but costs time:
# the network takes about as long as the number of pixels, 24 ms at 384x512 on a Nano 2GB.
set -e
SIZES="${SIZES:-384x512 480x640 576x768 768x1024}"
WORK="${WORK:-$HOME/yolo-export}"
mkdir -p "$WORK" && cd "$WORK"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
# CPU PyTorch is enough for exporting (about 200 MB instead of several GB)
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --quiet ultralytics onnx onnxslim onnxruntime
python - $SIZES <<'PY'
import shutil
import sys
from ultralytics import YOLO
for size in sys.argv[1:]:
    h, w = (int(v) for v in size.split("x"))
    out = YOLO("yolov8n.pt").export(format="onnx", imgsz=(h, w), opset=12, simplify=True,
                                    dynamic=False, batch=1)
    shutil.copy(out, "yolov8n_%dx%d.onnx" % (h, w))
    print("wrote yolov8n_%dx%d.onnx" % (h, w))
PY
ls -la "$WORK"/*.onnx
