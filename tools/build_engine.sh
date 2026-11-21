#!/usr/bin/env bash
# Build a TensorRT engine on the Jetson from an ONNX file in models/ and print how fast it is.
#
#     bash tools/build_engine.sh 768x1024    # models/yolov8n_768x1024.onnx -> ...fp16.engine
#
# The ONNX files come from tools/export_yolo.sh (run on a PC, copy them to models/).
# This takes a long time on a Nano 2GB (about 5 minutes at 384x512, 15 at 576x768, 25 or more
# at 768x1024) and needs most of the memory: stop the camera service first so the build has the
# board to itself and picks the fastest kernels:
#
#     sudo systemctl stop catcam    ...build...    sudo systemctl start catcam
set -e
cd "$(dirname "$0")/../models"
SIZE="${1:?usage: build_engine.sh HxW   (e.g. 768x1024)}"
ONNX="yolov8n_${SIZE}.onnx"
ENGINE="yolov8n_${SIZE}.fp16.engine"
[ -f "$ONNX" ] || { echo "models/$ONNX is missing (make it with tools/export_yolo.sh, copy it here)"; exit 1; }
if systemctl is-active --quiet catcam 2>/dev/null; then
    echo "note: the catcam service is running; the build is slower and may pick slower kernels"
fi
/usr/src/tensorrt/bin/trtexec --onnx="$ONNX" --fp16 --workspace=256 --saveEngine="$ENGINE" \
    --warmUp=500 --duration=5 2>&1 | grep -E "Engine built|GPU Compute Time|Throughput|PASSED|FAILED|rror"
ls -la "$ENGINE"
echo "use it with:  \"detector\": {\"type\": \"yolo\", \"engine\": \"models/$ENGINE\"}"
