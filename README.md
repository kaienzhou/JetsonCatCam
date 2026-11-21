# JetsonCatCam

固定机位的本地 AI 猫咪相机，跑在 Jetson Nano 2GB 上。不上云，手机浏览器直接看。

- 手机或电脑浏览器里看实时画面，用手指直接在画面上**画区域**（沙发、食盆、猫爬架……），保存后马上生效
- 猫进入或离开区域时记录事件，进入时还会留一张照片，可推送到手机
- **历史记录**：每天各区域停留多久、一天里什么时候在哪、每次出现的照片，可导出 CSV
- 要**访问口令**才能看；网页可以添加到手机主屏幕，当 App 用
- 检测用 COCO 预训练模型（含 `cat` 类），**不需要自己训练**：默认的 jetson-inference `ssd-mobilenet-v2`，或精度更好的 YOLOv8n（TensorRT）

## 状态

| 部分 | 状态 |
|---|---|
| 区域状态机（防抖、离开宽限、丢失） | 已完成，单元测试通过 |
| 网页 App：实时 / 历史 / 区域编辑 / 设置 | 已完成：电脑上用模拟摄像头逐页点过；板子上用真实摄像头测过登录、实时画面、区域编辑和历史接口 |
| 访问口令、历史库（SQLite，扛断电）、事件照片 | 已完成：117 项单元测试在 Python 3.12 和板子的 Python 3.6 上都通过；事件照片还没用真猫在板子上录过 |
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

首次启动会为 `ssd-mobilenet-v2` 构建 TensorRT 引擎，**要等几分钟**（板子上实测约 280 秒）。

**第一次启动时会生成访问口令**，并打印在日志里（用 systemd 的话看 `journalctl -u catcam`）。随时在项目目录里查看：

```bash
python3 -m catcam code        # 显示口令和一个自动登录的链接
python3 -m catcam new-code    # 换新口令，所有已登录的手机和电脑都要重新登录
```

然后手机或电脑浏览器打开 `http://<jetson-ip>:8080`，输入口令（或直接打开 `code` 命令给出的链接）。登录一次后浏览器会记住 180 天。

没有硬件时可以用模拟猫（一间小房间里走来走去的猫）测界面和逻辑，需要 OpenCV 才有画面：

```bash
python3 -m catcam --backend fake      # 配置里加 "fake": {"speed": 0.3} 让猫走快一点
python3 -m unittest discover -s tests
```

## 网页 App

- **实时**：视频、猫现在在哪个区域、今天各区域停留时间、最近事件（带照片）。没人看的时候板子不画图也不压 JPEG，更省电。
- **历史**：最近 7 / 14 / 30 天每天的停留时间（按区域叠起来的柱状图）；点某一天看它按小时的分布和当天所有事件；可导出 CSV。
- **区域**：在一张没有叠加图形的实时画面上画区域。点「＋ 新区域」依次点出几个角；点区域选中它，拖圆点改形状，拖区域里面整体移动，点边线加顶点，长按圆点删顶点。保存后立刻生效，不用重启。
- **设置**：设备信息、存储用量、导出全部记录、退出登录。

手机上把它加到主屏幕：iPhone 用 Safari（分享 → 添加到主屏幕）；安卓用 Chrome（菜单 → 添加到主屏幕）。通过普通 HTTP 访问时，安卓只能做成"快捷方式"，要完整的"安装"需要 HTTPS。

### 数据放在哪

都在 `data/`（可用 `data_dir` 改）：

| 文件 | 内容 |
|---|---|
| `catcam.db` | 所有事件（SQLite，写入是原子的，断电不会留下写了一半的记录；文件损坏时会改名保留并重新开始） |
| `snapshots/` | 出现和进入区域时的照片，默认保留 30 天、总共不超过 500 MB |
| `access_code`、`secret` | 访问口令和签名密钥（只有自己能读） |

旧版的 `events.jsonl` 会在第一次启动时自动导入一次。

### 接口

登录后（cookie，或请求头 `Authorization: Bearer <口令>`）可以用，方便写脚本：

| 路径 | 内容 |
|---|---|
| `GET /api/hello` | 不用登录：`{"app": "catcam", "api": 1, "version", "name", "login"}`，客户端用它认出相机 |
| `GET /api/state` | 现在的状态、今天各区域停留、最近事件 |
| `GET /api/history?days=14` | 每天各区域的停留秒数和次数 |
| `GET /api/day?date=2026-11-16` | 一天里按小时的分布和事件 |
| `GET /snap/<事件 id>.jpg` | 事件照片 |
| `GET /api/export.csv?days=365` | 导出事件 |
| `GET/POST /api/zones` | 读取 / 保存区域 |
| `GET /stream.mjpg` | 实时视频（最多同时 6 路） |

### 安全

口令连错 5 次会把那个地址锁 1 分钟。页面和视频没有加密（普通 HTTP），所以**只在信得过的局域网里用**；要从外面看，请走 VPN（例如 Tailscale），不要把端口直接映射到公网。

## 配置要点

`config.json` 里的坐标全部是 0 到 1 的归一化坐标，左上角为原点。

- `zones`：多边形顶点列表，按顺序排列。平时在网页的「区域」页里画就行，保存时程序会改写这一项（其余内容不动，并留一份 `config.json.bak`）。
- `data_dir`：数据目录，默认 `data`，放历史库、事件照片和口令文件，见"数据放在哪"。
- `auth`：`{"enabled": true, "code": ""}`。`code` 留空就用自动生成的随机口令，想用自己的口令就填在这里；`enabled` 设成 `false` 会关掉口令，只在完全信得过的网络里这么做。
- `snapshots`：`{"enabled": true, "keep_days": 30, "max_mb": 500}`，事件照片的开关、保留天数和总大小上限。
- `fake`：只对 `--backend fake` 有效，例如 `{"speed": 0.3}`。
- `log`：旧版事件日志的路径，只在第一次启动时导入一次。
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

程序在摄像头 20 秒没有新画面时会主动退出，服务会先重启摄像头服务（nvargus-daemon）再把它拉起来。在网页上改区域不用重启；手改了 `config.json` 的其他项后，`sudo systemctl restart catcam` 生效。

给板子起个好记的名字，局域网里就能用 `http://catcam.local:8080` 访问（电脑或手机开着代理时，`.local` 名字有时会被代理劫持，打不开就还是用 IP）：

```bash
sudo bash deploy/set-hostname.sh              # 改成 catcam
sudo bash deploy/set-hostname.sh living-room  # 或者别的名字（小写英文字母、数字和 -）
```

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
| `make_icons.py` | 重画网页 App 的图标（需要 OpenCV，平时不用） |

## 已知限制

- 默认只追踪画面里置信度最高的一只猫。
- 板子没有电池时钟：断电重启后时间要靠联网校时。刚开机的几十秒里，或者网络没通时，时间可能不对，这期间记下的事件时间也会不对。
- 只有一个共用的访问口令，不分用户；页面和视频不加密（普通 HTTP）。
- 区域判定用检测框中心点，俯拍效果最好；平视时猫在远处会和近处的区域重叠。
- JetPack 4.6 是 Nano 的最后一个系统版本（Python 3.6），不会再更新。
- 2GB 板子内存很紧：应用约占 1 GB，带桌面系统时只剩约 100 MB 余量，产品化时建议关掉桌面（`sudo systemctl set-default multi-user.target`）。

## 路线图

1. 无线网络：USB 网卡、存多个 Wi-Fi、手机热点兜底，没有屏幕也能配网（现在要接网线）
2. 从外面看：走 VPN（如 Tailscale）；上 HTTPS 以后网页才能完整"安装"、才能收网页推送
3. 推送带上事件照片
4. 只读根文件系统，防 SD 卡损坏（数据放单独的可写分区）
5. 一键刷写的 SD 卡镜像
6. 3D 打印外壳（散热、摄像头支架、壁挂）
