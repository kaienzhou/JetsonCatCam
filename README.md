# JetsonCatCam

固定机位的本地 AI 猫咪相机，跑在 Jetson Nano 2GB 上。不上云，手机浏览器直接看。

- 实时画面里画出你划定的区域（沙发、食盆、猫爬架……）
- 猫进入或离开区域时记录事件，可推送到手机
- 统计猫今天在各区域待了多久
- 检测用 COCO 预训练模型（含 `cat` 类），**不需要自己训练**：默认的 jetson-inference `ssd-mobilenet-v2`，或精度更好的 YOLOv8n（TensorRT）

## 状态

| 部分 | 状态 |
|---|---|
| 区域状态机（防抖、离开宽限、丢失） | 已完成，单元测试通过 |
| Web 界面、事件日志 | 已完成，真机上跑通 |
| 推送（`notify.url`） | 已实现（HTTP GET 模板，Bark 风格），尚未在真实推送服务上验证 |
| CSI 摄像头采集（OpenCV + GStreamer） | 真机验证：30 帧/秒，连续运行稳定 |
| 检测：ssd-mobilenet-v2 | 真机验证：约 21 帧/秒；猫要占画面宽度约四分之一以上才可靠 |
| 检测：YOLOv8n（TensorRT） | 真机验证：512×384 约 36 ms/帧，限制 15 帧/秒连续 8 分钟稳定；640×480 更准，但满载约 3 分钟后板子曾掉电复位（疑似供电，未完全确认），见下文"检测模型"和"供电" |
| 开机自启（systemd）+ 关桌面 | 真机验证：断电重启后自己起来，见"开机自启" |
| 外壳、产品化镜像 | 未开始 |

## 在 Jetson 上运行

前提：JetPack 4.6（L4T R32.6/32.7，Python 3.6），已装好 [jetson-inference](https://github.com/dusty-nv/jetson-inference) 的 `L4T-R32.7.1` 分支（带 Python 绑定），摄像头接好。

```bash
git clone <本仓库> && cd JetsonCatCam
cp config.example.json config.json    # 按你的房间改 camera 和 zones
python3 -m catcam --backend jetson
```

首次启动会为 `ssd-mobilenet-v2` 构建 TensorRT 引擎，**要等几分钟**（板子上实测约 280 秒）。之后手机浏览器打开 `http://<jetson-ip>:8080`。页面上会显示每秒处理帧数，以及采集 / 检测 / 转换各花多少毫秒。

没有硬件时可以用模拟猫测界面和逻辑：

```bash
python3 -m catcam --backend fake
python3 -m unittest discover -s tests
```

## 配置要点

`config.json` 里的坐标全部是 0 到 1 的归一化坐标，左上角为原点。

- `zones`：多边形顶点列表，按顺序排列。
- `tracker.enter_frames`：连续多少帧在区域内才算进入（防误检抖动）。
- `tracker.leave_seconds`：离开区域多久才算离开（猫短暂探头不算）。
- `notify.url`：推送地址模板，`{title}` 和 `{body}` 会被替换，例如 Bark：`https://api.day.app/<key>/{title}/{body}`。`notify.zones` 指定哪些区域进入时推送。
- `debug`：设为 `true` 时画面里画出模型认出的**所有**物体（黄框和置信度），用来判断"为什么没认出来"。YOLO 要同时设 `detector.all_classes: true` 才会报告猫以外的物体。
- `max_fps`：处理帧率上限，0 为不限制。限制后 GPU 有空闲，发热和功耗都更低；盯猫 10～15 帧/秒足够。
- `detector`：见下文。

## 检测模型

`config.json` 里的 `detector` 选择检测器：

```json
"detector": {"type": "ssd", "min_conf": 0.4}
"detector": {"type": "yolo", "engine": "models/yolov8n_384x512.fp16.engine", "min_conf": 0.4}
```

### 为什么要换 YOLO

用 `tools/size_test.py` 把猫图缩小贴进真实房间画面测试：`ssd-mobilenet-v2` 要求猫宽度占画面的 25%～45% 以上才可靠（约 1～1.3 米内），再小就基本检不到。偏色（NoIR 白天偏紫）对它影响不大，瓶颈是目标大小。

### 准备 YOLO 引擎

1. 在普通电脑（Linux / WSL）上导出 ONNX：`bash tools/export_yolo.sh`（用独立 venv，装 CPU 版 PyTorch 和 ultralytics，权重来自官方发布，约 6 MB）。
2. 把 `yolov8n_384x512.onnx` 复制到 Jetson 的 `models/`，在 Jetson 上编译引擎（引擎只能用于编译它的那块 GPU 和 TensorRT 版本）：

   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=models/yolov8n_384x512.onnx --fp16 --workspace=512 \
       --saveEngine=models/yolov8n_384x512.fp16.engine
   ```
3. 冒烟测试：`python3 tools/smoke_detect.py --engine models/yolov8n_384x512.fp16.engine <图片>`

输入尺寸 384×512（高×宽）和摄像头 4:3 的画面一致，不需要补边。

## 使用 CSI 摄像头（Raspberry Pi Camera V2 / NoIR V2）

两者都是 IMX219，JetPack 4.6 自带驱动。`camera` 里写：

```json
"camera": {"source": "csi://0", "width": 640, "height": 480, "rate": 30, "rotate": 0,
           "sensor_mode": 3, "sensor_width": 1640, "sensor_height": 1232}
```

- `width`/`height` 是送去检测和显示的画面大小；传感器模式 3 是 1640×1232、30 帧/秒（全视野，2×2 合并像素，暗处更干净）。
- `rotate`：画面方向，0 / 90 / 180 / 270（顺时针度数）。0 是传感器原方向。
- 2GB 版只有一个 CSI 口。排线的银色触点一面朝向散热片，蓝色加强片朝外（这是 Jetson Nano 4GB 资料里的插法，2GB 接口规格相同但没找到官方确认；务必断电后再插拔）。
- **CSI 采集不是用 jetson-utils 自带的 `csi://`，而是 OpenCV + GStreamer。** 原因：jetson-utils 在这套环境下，只要不转 180° 就会报 `nvbuf_utils: dmabuf_fd ... mapped entry NOT found` 然后进程直接退出，还会把摄像头服务（nvargus-daemon）带崩。
- NoIR 没有红外滤光片：白天画面偏紫红。实测对检测影响很小（置信度基本不变，自动白平衡也无益）。夜里配 850nm 红外补光灯可以看到猫，但画面是灰度，效果需要实测。
- 补光灯放在镜头旁边斜着打，别正对镜头，否则反光。
- V2 水平视角约 62°。

USB 摄像头：`"source": "/dev/video0"`，走 jetson-utils，`width`/`height` 是采集大小。

## 开机自启

```bash
sudo bash deploy/install-service.sh        # 装成 systemd 服务，开机自启，退出后自动重启
systemctl status catcam                     # 状态
journalctl -u catcam -f                     # 看日志
sudo bash deploy/install-service.sh --remove
```

程序在摄像头 20 秒没有新画面时会主动退出，服务会先重启摄像头服务（nvargus-daemon）再把它拉起来。改了 `config.json` 后，`sudo systemctl restart catcam` 生效。

### 关掉桌面（推荐）

2GB 板子的内存很紧，不接显示器用的话把图形界面关掉：

```bash
sudo systemctl set-default multi-user.target   # 以后开机不起桌面
sudo systemctl isolate multi-user.target       # 立刻关掉当前桌面
# 想恢复：sudo systemctl set-default graphical.target && sudo reboot
```

实测关桌面后可用内存约增加 70 MB。内存仍然紧张时，还可以停掉用不到的服务（例如 docker）。

## 供电

Jetson Nano 2GB 只有一个 USB-C 供电口，要求 **5V/3A**。供电不够时，板子在 GPU 满载时会直接**掉电复位**（内核启动日志里是 `TEGRA_POWER_ON_RESET`，而且没有正常关机记录；手动拔插电源也会留下同样的记录，先排除人为断电），摄像头也可能随之报错。

- 用标称 15W 以上的 USB-C 充电头，不要用电脑 USB 口。
- 线要短而粗（1 米以内、能跑 3A）。有 USB 功率计的话看满载时的电压，低于 4.75V 就有问题。
- 板子上别再接耗电的 USB 设备。
- 供电条件一般时：用 512×384 的引擎，并设置 `max_fps`（10～15）。

## 排错工具（`tools/`）

| 脚本 | 用途 |
|---|---|
| `smoke_detect.py` | 对图片跑检测，看结果和速度 |
| `camera_soak.py` | 摄像头长时间采集，统计掉帧 |
| `tint_test.py` | 模拟偏色，看对猫置信度的影响 |
| `size_test.py` | 猫图缩放贴进画面，测多小还能检到 |
| `export_yolo.sh` | 导出 YOLOv8n 的 ONNX |

## 已知限制

- 默认只追踪画面里置信度最高的一只猫。
- 区域判定用检测框中心点，俯拍效果最好；平视时猫在远处会和近处的区域重叠。
- JetPack 4.6 是 Nano 的最后一个系统版本（Python 3.6），不会再更新。
- 2GB 板子内存很紧：应用约占 1 GB，带桌面系统时只剩约 100 MB 余量，产品化时建议关掉桌面（`sudo systemctl set-default multi-user.target`）。

## 路线图

1. 画区域的网页工具（现在要手改 JSON）
2. 只读根文件系统，防 SD 卡损坏
3. 一键刷写的 SD 卡镜像
4. 3D 打印外壳（散热、摄像头支架、壁挂）
