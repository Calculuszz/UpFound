# edge_cctv — Process 1 (Edge AI): CCTV → Event Contract

Live Hikvision RTSP → detect abandoned objects → crop + CLIP embed → emit the
**same Event Contract** as the replay producer (`clip_to_events.py`). Only
`source="cctv"` differs, so the backend cannot tell replay and cctv apart.

See `../PROCESS_1_cctv_edge_spec.md` for the full spec.

## ⚡ ใช้ง่ายสุด — ตัวช่วย `./cctv`

ไม่ต้องจำคำสั่งยาว ใช้ wrapper `./cctv` (อยู่ที่ root ของ repo) — activate venv ให้เอง,
เซฟ preview ให้, แล้ว**บอกคำสั่ง scp ที่พร้อม copy** ตอนจบ:

```bash
./cctv clip                      # ทดสอบคลิป default (detector: yolo)
DET=yoloe ./cctv clip            # สลับเป็น YOLOE (จับ tablet/ของนอก COCO ได้)
DET=yoloe ./cctv clip 804956840.350391.mp4   # เลือกคลิปเอง
DET=yoloe ./cctv cam             # กล้องจริง (RTSP)
./cctv events 10                 # ดู 10 event ล่าสุด
./cctv help                      # วิธีใช้ทั้งหมด
```

- สลับ detector = ตัวแปร `DET` (`yolo` = default COCO / `yoloe` = open-vocab)
- preview เซฟแยกไฟล์ตาม detector → `out/preview_yolo.mp4`, `out/preview_yoloe.mp4`
- flag อื่นๆ ส่งต่อได้เลย เช่น `./cctv clip --max-frames 100`

> เบื้องหลังมันเรียก `python -m edge_cctv.run ...` ให้ ถ้าต้องคุมละเอียดใช้คำสั่งเต็มได้ (ด้านล่าง)

## Setup

```bash
pip install -r requirements.txt
```

> ⚠️ บน DGX Spark: **ห้าม** `pip install -r requirements.txt` ถ้ามันจะดึง torch ทับ —
> torch build `cu130` (เห็น GPU) ลงไว้แล้ว ลงทับจะได้ build ที่มองไม่เห็น GPU

## Run — 2 modes

### Mode 1: RTSP (กล้องจริง)

```powershell
# ตั้ง env
$env:EDGE_RTSP_PASSWORD = "รหัสผ่านกล้อง"
$env:EDGE_CAMERA_IP = "192.168.1.64"

# production (main stream ch 101, full pipeline)
python -m edge_cctv.run

# dev (sub stream ch 102, ไม่โหลด CLIP, เปิด preview)
python -m edge_cctv.run --dev --no-embed --preview

# smoke test 60 วินาที
python -m edge_cctv.run --dev --no-embed --preview --duration 60
```

### Mode 2: Video file (integration test — ไม่ต้องมีกล้อง)

ใช้ไฟล์วิดีโอที่บันทึกไว้ใน `vedio_test/` แทน RTSP. ผลลัพธ์ reproducible 100%.

```powershell
# รันจากวิดีโอ + เปิดหน้าต่าง preview ดูผล
python -m edge_cctv.run --source vedio_test/804956840.304276.mp4 --no-embed --preview

# รันจากวิดีโออีกคลิป
python -m edge_cctv.run --source vedio_test/804956840.350391.mp4 --no-embed --preview

# รันแบบ headless (ไม่มีหน้าต่าง) ดูแค่ log + events.jsonl
python -m edge_cctv.run --source vedio_test/804956840.304276.mp4 --no-embed
```

### Preview แบบ headless → เซฟเป็นไฟล์วิดีโอ (`--save-preview`)

บนเครื่อง remote/SSH ที่ไม่มีจอ (`cv2.imshow` เปิดไม่ได้) ใช้ `--save-preview PATH`
เพื่อเขียนเฟรมที่วาดกรอบ/สถานะ dwell/เส้น owner ลงไฟล์ `.mp4` แทนการเปิดหน้าต่าง
แล้ว `scp` กลับมาดูบน PC เอง โหมดนี้ **ไม่เรียก GUI เลย** ปลอดภัยกับ headless.

```bash
python -m edge_cctv.run --source vedio_test/804956840.304276.mp4 --no-embed --save-preview out/preview.mp4
```

- ใช้ logic การวาดชุดเดียวกับ `--preview` (กรอบ item, `ABANDONED`/นับถอยหลัง,
  gate `static`/`owner`, เส้น owner ระยะ px, กล่อง person สีน้ำเงิน)
- fps เอาจาก source (VideoSource) ถ้าไม่ได้ default 25; ขนาดเฟรมตามเฟรมจริง
- ปิดไฟล์ให้เรียบร้อยเสมอ (แม้โดน Ctrl-C)
- **codec fallback (ARM64):** ลอง `mp4v` → `avc1` → สุดท้าย `.avi`/`XVID`
  ถ้า codec ไม่ครบ จะ log path จริงที่เขียนได้ (ไม่ fail เงียบ)
- ใช้ร่วมกับ flag อื่นได้ เช่น `--no-embed --no-person --save-preview ...`

**ข้อแตกต่าง video mode vs RTSP:**
- timestamp ใน event ใช้ frame_idx/fps (เริ่มจาก 2026-01-01 UTC) แทน wall-clock → ผลลัพธ์เหมือนกันทุกครั้งที่รัน
- จบอัตโนมัติเมื่อวิดีโอหมด
- ไม่ต้องตั้ง `EDGE_RTSP_PASSWORD`
- reader thread ไม่ใช้ (อ่านจากไฟล์ตามลำดับปกติ)

## Output

- `./out/events.jsonl` — event ทุกตัวที่ยิง (1 บรรทัด = 1 JSON event)
- `./out/crops/` — รูป crop ของวัตถุที่ถูกทิ้ง
- `./out/persons/` — รูปเฟรมคนวาง (สำหรับเทียบใบหน้า Process 3)

## Flags ทั้งหมด

| Flag | ทำอะไร |
|------|--------|
| `--dev` | ใช้ sub stream (ch 102) เบากว่า |
| `--no-embed` | ข้าม CLIP embedding (dev/test) |
| `--no-person` | ปิด person capture (เร็วขึ้น) |
| `--preview` | เปิดหน้าต่างภาพสด + กรอบ + เส้น owner (ต้องมีจอ) |
| `--save-preview PATH` | เขียนวิดีโอที่วาดกรอบ/สถานะ ลงไฟล์ (headless-safe) |
| `--source PATH` | อ่านจากไฟล์วิดีโอแทน RTSP |
| `--duration N` | หยุดหลัง N วินาที |
| `--max-frames N` | หยุดหลังประมวล N เฟรม |

## Test

```powershell
# unit tests (ไม่ต้องมีกล้อง/GPU)
python -m pytest edge_cctv/tests -q

# integration test (ต้องมี YOLO weights + วิดีโอ)
python -m edge_cctv.run --source vedio_test/804956840.304276.mp4 --no-embed --preview
```

## Detector: yolo (default) vs yoloe (open-vocab)

สลับด้วย env `EDGE_DETECTOR` (หรือ `DET=` ผ่าน `./cctv`) — **ไม่ต้องแก้ dwell/owner logic เลย**
config จะสลับ `ITEM_CLASSES` / thresholds / weights / `model_version` ให้อัตโนมัติ:

| | `EDGE_DETECTOR=yolo` (default) | `EDGE_DETECTOR=yoloe` |
|---|---|---|
| โมเดล | yolo26x (COCO 80 คลาส) | YOLOE-11L open-vocab |
| คลาสที่จับ | fixed COCO (`ITEM_CLASSES`) | **text prompt ปรับได้** (`EDGE_YOLOE_PROMPTS`) |
| จับ tablet/iPad | ❌ COCO ไม่มีคลาส tablet | ✅ prompt `"tablet"` ได้เลย |
| `model_version` | `yolo26x_clip-vitb32` | `yoloe11l_clip-vitb32` |
| speed @1920 FP16 | ~12 fps | ~21 fps |

**YOLOE prompts** (default: `backpack,handbag,suitcase,laptop,tablet,umbrella,book`) —
ปรับผ่าน `EDGE_YOLOE_PROMPTS` (คั่นด้วย comma). เพิ่มอะไรก็ได้ที่นึกออก เช่น
`"tablet,wallet,water bottle,headphones"` โดยไม่ต้องเทรนใหม่

> ⚠️ **อย่าใส่ prompt ที่ความหมายทับกัน** (เช่น `tablet` + `wallet` พร้อมกัน) — คลาสจะ
> flicker ราย เฟรม (majority-vote ยังกลบ event class ให้ถูก แต่ noisy ขึ้น)

> ⚠️ **Event Contract:** `model_version` เปลี่ยนตาม detector → **Process 2 (matching) +
> replay producer (`clip_to_events.py`) ต้องใช้สตริงเดียวกัน** ไม่งั้น event mismatch

## Model & resolution (default)

- **โมเดล default = `yolo26x.pt`** (YOLO26 extra-large, mAP 56.8) — ความแม่นยำสูงสุด
  สำหรับกล้อง CCTV มุมสูง. ultralytics จะดาวน์โหลดให้อัตโนมัติครั้งแรกที่เรียก (~113MB).
  เปลี่ยนได้ผ่าน env `EDGE_YOLO_WEIGHTS` (เช่น `yolo26m.pt` ถ้าอยากเบา/เร็วขึ้น).
- **`IMGSZ` default = 1920** (ต้องหารด้วย 32 ลงตัว). สูง = แม่นกับของเล็ก/ไกลขึ้น
  แต่ช้าลงและกิน memory มากขึ้น. ปรับผ่าน env `EDGE_IMGSZ` ได้ (เช่น `1280`, `1536`).
- **FP16 inference เปิดเป็น default** (`EDGE_HALF=1`) — เร็วขึ้น ~1.4x บน tensor cores
  ของ GB10, memory ลดครึ่ง, ความแม่นต่างแทบวัดไม่ได้. ปิดได้ด้วย `EDGE_HALF=0`
  (บนเครื่องไม่มี CUDA จะ fallback เป็น FP32 อัตโนมัติ).
  ผลวัดจริงบน Spark: 1920+FP16 เร็วเท่า 1536+FP32 (~12.5 เฟรม/วิ) แต่ resolution สูงกว่า.
- ⚠️ **`ACTIVE_MODEL_VERSION` = `yolo26x_clip-vitb32`** เป็นส่วนหนึ่งของ Event Contract —
  ถ้าเปลี่ยนโมเดล ต้อง sync ค่านี้กับ **Process 2 (matching)** และ **replay producer
  (`clip_to_events.py`)** ให้ตรงกัน ไม่งั้น backend จะ mismatch/ทิ้ง event.

### ⚠️ Unified memory (DGX Spark / GB10)

บนเครื่อง unified-memory (CPU+GPU ใช้ RAM ก้อนเดียว 128GB, ไม่มี VRAM แยก)
**"GPU OOM" = "System OOM ทั้งเครื่อง"** — ถ้า memory เกิน เครื่องอาจค้างทั้งระบบ
(SSH หลุด ต้อง hard reboot) ไม่ใช่แค่ job ตาย. เวลาเพิ่ม `EDGE_IMGSZ` ให้**ไต่ทีละขั้น
แล้ววัด memory** อย่ากระโดดไปสูงสุดทันที.

`nvidia-smi` **อ่าน memory ไม่ได้** บน unified system — ให้ดูจาก:

```bash
grep MemAvailable /proc/meminfo      # หรือ
python -c "import psutil; print(psutil.virtual_memory())"
```

> หมายเหตุ: **ห้าม** `pip install torch` ทับ — torch build `cu130` (เห็น GPU) ลงไว้ถูกแล้ว.
> warning ว่า GB10 cuda capability 12.1 เกินช่วง torch → ข้ามได้ ปลอดภัย ไม่ใช่ error.

## Detection logic overview

1. YOLO26 detect + ByteTrack track → ได้ bbox + track_id ต่อเฟรม
2. กรอง: per-class confidence + min bbox size (กัน false positive)
3. DwellTracker: วัดความนิ่งแบบ anchor + scale-aware tolerance
4. Track adoption: สืบทอดสถานะเมื่อ track_id เปลี่ยน (IoU)
5. Owner-left gating: ยิงเฉพาะเมื่อคนวาง → เดินจากไป > 3 วิ
6. Majority-vote class: คลาสสุดท้ายใน event = โหวตตลอดอายุ track
7. Fire once → emit Event Contract + crop + person frame

## Env vars (ปรับ tuning)

| Var | Default | ความหมาย |
|-----|---------|----------|
| `EDGE_DWELL_SECONDS` | 8.0 | นิ่งกี่วินาทีถึงยิง |
| `EDGE_MOVE_TOL` | 25 | px floor สำหรับตัดสินว่า "ขยับ" |
| `EDGE_MOVE_TOL_FRAC` | 0.15 | สัดส่วนของ box side ที่ใช้เป็น tolerance |
| `EDGE_DETECTOR` | yolo | `yolo` (COCO) หรือ `yoloe` (open-vocab จับ tablet ได้) |
| `EDGE_YOLOE_PROMPTS` | backpack,handbag,suitcase,laptop,tablet,umbrella,book | prompt สำหรับ yoloe (คั่น comma) |
| `EDGE_YOLOE_CONF` | 0.25 | conf floor ของ yoloe (open-vocab conf ต่ำกว่า COCO) |
| `EDGE_YOLOE_WEIGHTS` | yoloe-11l-seg.pt | weights ของ yoloe |
| `EDGE_YOLO_WEIGHTS` | yolo26x.pt | ไฟล์ weights (yolo26m.pt = เบา/เร็วกว่า) |
| `EDGE_IMGSZ` | 1920 | YOLO input resolution (สูง=แม่นกับของเล็ก/ไกล แต่ช้า+กิน memory; หาร 32 ลงตัว) |
| `EDGE_HALF` | 1 | FP16 inference บน GPU (0=FP32; ไม่มี CUDA จะ FP32 อัตโนมัติ) |
| `EDGE_CONF_MIN` | 0.40 | confidence ขั้นต่ำ (default) |
| `EDGE_MIN_BBOX_SIDE` | 40 | กรอง bbox ด้านสั้น < X px |
| `EDGE_OWNER_RADIUS` | 150 | รัศมี px ที่นับว่า "คนอยู่ใกล้ของ" |
| `EDGE_OWNER_LEFT_SECONDS` | 3.0 | คนต้องห่างกี่วินาทีถึงนับว่า "ทิ้ง" |
| `EDGE_ADOPT_IOU` | 0.3 | IoU ขั้นต่ำสำหรับ track adoption |
| `EDGE_SAMPLE_EVERY` | 3 | ประมวลทุกกี่เฟรม |
| `EDGE_REQUIRE_MOVEMENT` | 1 | ต้องเคยขยับ/มีคนอยู่ใกล้ ก่อนยิง |
| `EDGE_REQUIRE_OWNER_LEFT` | 1 | ต้องรอให้คนเดินจากไปก่อนยิง |
| `EDGE_READER_THREAD` | 1 | background reader (กัน latency สะสม) |
