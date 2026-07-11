# edge_cctv — Process 1 (Edge AI): CCTV → Event Contract

Live Hikvision RTSP → detect abandoned objects → crop + CLIP embed → emit the
**same Event Contract** as the replay producer (`clip_to_events.py`). Only
`source="cctv"` differs, so the backend cannot tell replay and cctv apart.

See `../PROCESS_1_cctv_edge_spec.md` for the full spec.

## Setup

```bash
pip install -r requirements.txt
```

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

## Detection logic overview

1. YOLO26s detect + ByteTrack track → ได้ bbox + track_id ต่อเฟรม
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
| `EDGE_IMGSZ` | 1280 | YOLO input resolution (1280=ดีกับของเล็ก/ไกล, 640=เร็วกว่า) |
| `EDGE_CONF_MIN` | 0.40 | confidence ขั้นต่ำ (default) |
| `EDGE_MIN_BBOX_SIDE` | 40 | กรอง bbox ด้านสั้น < X px |
| `EDGE_OWNER_RADIUS` | 150 | รัศมี px ที่นับว่า "คนอยู่ใกล้ของ" |
| `EDGE_OWNER_LEFT_SECONDS` | 3.0 | คนต้องห่างกี่วินาทีถึงนับว่า "ทิ้ง" |
| `EDGE_ADOPT_IOU` | 0.3 | IoU ขั้นต่ำสำหรับ track adoption |
| `EDGE_SAMPLE_EVERY` | 3 | ประมวลทุกกี่เฟรม |
| `EDGE_REQUIRE_MOVEMENT` | 1 | ต้องเคยขยับ/มีคนอยู่ใกล้ ก่อนยิง |
| `EDGE_REQUIRE_OWNER_LEFT` | 1 | ต้องรอให้คนเดินจากไปก่อนยิง |
| `EDGE_READER_THREAD` | 1 | background reader (กัน latency สะสม) |
