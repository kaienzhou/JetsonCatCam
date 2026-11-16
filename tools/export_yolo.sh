#!/usr/bin/env bash
# Export YOLOv8n to ONNX with a static input shape that TensorRT 8.0 (JetPack 4.6) accepts.
# Run this on a normal PC (Linux / WSL) with Python 3.9+; it uses its own virtualenv.
#
#     bash tools/export_yolo.sh            # -> yolov8n_384x512.onnx  yolov8n_480x640.onnx
#
# Then copy the .onnx files to the Jetson (models/ in this project) and build the engines
# there (an engine only works on the GPU and TensorRT version it was built with):
#
#     /usr/src/tensorrt/bin/trtexec --onnx=models/yolov8n_384x512.onnx --fp16 --workspace=512 \
#         --saveEngine=models/yolov8n_384x512.fp16.engine
#
# Input size is height x width; 384x512 matches the 4:3 frames of the camera (no padding).
set -e
WORK="${WORK:-$HOME/yolo-export}"
mkdir -p "$WORK" && cd "$WORK"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
# CPU PyTorch is enough for exporting (about 200 MB instead of several GB)
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --quiet ultralytics onnx onnxslim onnxruntime
python - <<'PY'
import shutil
from ultralytics import YOLO
for h, w in ((384, 512), (480, 640)):
    out = YOLO("yolov8n.pt").export(format="onnx", imgsz=(h, w), opset=12, simplify=True,
                                    dynamic=False, batch=1)
    shutil.copy(out, "yolov8n_%dx%d.onnx" % (h, w))
    print("wrote yolov8n_%dx%d.onnx" % (h, w))
PY
ls -la "$WORK"/*.onnx
