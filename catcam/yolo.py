"""YOLOv8 on TensorRT for the Jetson Nano (JetPack 4.6: TensorRT 8.0, Python 3.6).

The engine is built from an ONNX export with static input shape, e.g.
    /usr/src/tensorrt/bin/trtexec --onnx=yolov8n_384x512.onnx --fp16 \
        --saveEngine=yolov8n_384x512.fp16.engine
(see tools/export_yolo.sh). Input is 1x3xHxW float32 RGB in 0..1, output is
1x(4+80)xN: rows 0-3 are cx, cy, w, h in input pixels, rows 4.. are the COCO class
scores (sigmoid already applied, no objectness).

Input and output live in CUDA mapped (zero-copy) memory allocated through jetson-utils,
so no pycuda is needed.
"""
import numpy as np

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"]
CAT = COCO.index("cat")


def nms(boxes, scores, iou_thr):
    """Greedy non-maximum suppression. boxes: (n, 4) x1, y1, x2, y2. Returns kept indices."""
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if not rest.size:
            break
        w = np.maximum(0.0, np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]))
        h = np.maximum(0.0, np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]))
        inter = w * h
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(out, conf=0.4, iou=0.5, class_ids=None):
    """Candidate boxes of one YOLOv8 output (4 + classes, N) -> [(class_id, score, (x1, y1, x2, y2))].

    Boxes are in input pixels, best score first. class_ids limits the work to those
    classes (reading only their score rows); None looks at all of them.
    """
    if class_ids is None:
        scores = out[4:]
        best = scores.max(axis=0)
        idx = np.nonzero(best >= conf)[0]
        if not idx.size:
            return []
        ids = scores[:, idx].argmax(axis=0)  # only for the few candidate anchors
    else:
        rows = out[[4 + c for c in class_ids]]
        best = rows.max(axis=0)
        idx = np.nonzero(best >= conf)[0]
        if not idx.size:
            return []
        ids = np.asarray(class_ids)[rows[:, idx].argmax(axis=0)]
    cx, cy, w, h = out[0, idx], out[1, idx], out[2, idx], out[3, idx]
    boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
    sc, cls = best[idx], ids
    found = []
    for c in np.unique(cls):  # suppress overlaps within each class only
        m = np.nonzero(cls == c)[0]
        for k in nms(boxes[m], sc[m], iou):
            found.append((int(c), float(sc[m[k]]), tuple(float(v) for v in boxes[m[k]])))
    found.sort(key=lambda d: -d[1])
    return found


def letterbox_params(fw, fh, w, h):
    """Fit a (fw, fh) frame into a (w, h) network input keeping the aspect ratio.

    Returns (scale, new width, new height, x offset, y offset) of the resized frame.
    """
    scale = min(w / float(fw), h / float(fh))
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    return scale, nw, nh, (w - nw) // 2, (h - nh) // 2


def to_frame(box, fw, fh, scale, dx, dy):
    """Box in network-input pixels -> (l, t, r, b) normalized to the frame, clipped to 0..1."""
    x1, y1, x2, y2 = box
    out = ((x1 - dx) / (fw * scale), (y1 - dy) / (fh * scale),
           (x2 - dx) / (fw * scale), (y2 - dy) / (fh * scale))
    return tuple(min(1.0, max(0.0, v)) for v in out)


class YoloTrt(object):
    """Runs a YOLOv8 TensorRT engine on BGR frames (OpenCV) and returns normalized boxes."""

    def __init__(self, engine_path, utils, conf=0.4, iou=0.5, class_ids=(CAT,)):
        import cv2
        import tensorrt as trt
        self.cv2 = cv2
        self.conf = conf
        self.iou = iou
        self.class_ids = list(class_ids) if class_ids else None
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("cannot load TensorRT engine %s (built with another TensorRT "
                               "version or for another GPU?)" % engine_path)
        self.ctx = self.engine.create_execution_context()
        shape_in = tuple(self.engine.get_binding_shape(0))    # (1, 3, H, W)
        shape_out = tuple(self.engine.get_binding_shape(1))   # (1, 4 + classes, N)
        if len(shape_in) != 4 or shape_in[0] != 1 or -1 in shape_in + shape_out:
            raise RuntimeError("engine needs static shapes, got %s -> %s" % (shape_in, shape_out))
        self.h, self.w = shape_in[2], shape_in[3]
        self.rows, self.anchors = shape_out[1], shape_out[2]
        self.mem_in = utils.cudaAllocMapped(size=3 * self.h * self.w * 4)
        self.mem_out = utils.cudaAllocMapped(size=self.rows * self.anchors * 4)
        # numpy views of the mapped memory (shape (rows*cols, width, 1) -> reshape is a view)
        self.view_in = utils.cudaToNumpy(self.mem_in, width=self.w, height=3 * self.h, depth=1)
        self.view_in = self.view_in.reshape(3, self.h, self.w)
        self.view_out = utils.cudaToNumpy(self.mem_out, width=self.anchors, height=self.rows, depth=1)
        self.view_out = self.view_out.reshape(self.rows, self.anchors)
        self.bindings = [int(self.mem_in.ptr), int(self.mem_out.ptr)]
        self.names = COCO

    def detect(self, frame):
        """frame: BGR uint8 (h, w, 3) -> [(class name, score, (l, t, r, b) normalized to the frame)]."""
        cv2 = self.cv2
        fh, fw = frame.shape[:2]
        scale, nw, nh, dx, dy = letterbox_params(fw, fh, self.w, self.h)
        img = frame if (nw, nh) == (fw, fh) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if (nw, nh) != (self.w, self.h):  # letterbox: keep the aspect ratio, pad with grey
            canvas = np.full((self.h, self.w, 3), 114, np.uint8)
            canvas[dy:dy + nh, dx:dx + nw] = img
            img = canvas
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (self.w, self.h), swapRB=True, crop=False)
        np.copyto(self.view_in, blob.reshape(3, self.h, self.w))
        if not self.ctx.execute_v2(self.bindings):
            raise RuntimeError("TensorRT execution failed")
        results = []
        # all classes: one bulk copy out of the mapped memory is much faster than letting
        # numpy walk over it; cat only: reading a single row straight from it is cheapest
        out = np.array(self.view_out) if self.class_ids is None else self.view_out
        for cid, score, box in decode(out, self.conf, self.iou, self.class_ids):
            results.append((self.names[cid], score, to_frame(box, fw, fh, scale, dx, dy)))
        return results
