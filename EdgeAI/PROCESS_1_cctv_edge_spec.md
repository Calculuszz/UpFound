# UpFound — Process 1 (Edge AI): CCTV Stream → Event Contract — Implementation Spec

> สำหรับ **AI coding agent** ใช้ implement โดยตรง
> เป็น "ขั้นที่ 2 ในflow" (Edge AI ตรวจจับ) เวอร์ชัน source = CCTV (Hikvision RTSP)
> **Output ต้องเหมือน `clip_to_events.py` ทุกประการ** ต่างแค่ source="cctv" — ปลายทาง (backend) ต้องแยกไม่ออกว่ามาจาก replay หรือ cctv

---

## 1. Goal & Scope

**Goal:** ดึงภาพสดจากกล้อง Hikvision ผ่าน RTSP → ตรวจจับของถูกทิ้ง → crop + embed → ยิง Event Contract เข้าคิว/ไฟล์ เหมือน producer ตัว replay

**In scope:**
- เชื่อม RTSP แบบทนหลุด (auto-reconnect)
- อ่านเฟรมแบบ drop-frame (กัน latency สะสม)
- Detect + track + dwell logic (ของนิ่งเกินกำหนด = abandoned)
- Crop + CLIP embed + ประกอบ Event + emit
- **เก็บเฟรมของ "คนวางของ" ด้วย** (ตาม design v3 — ใช้เทียบใบหน้า process 3)

**Out of scope:**
- การจับคู่/ค้นหา (Process 2)
- การพิสูจน์ตัวตน (Process 3)
- วิเคราะห์ลักษณะบุคคล (สีเสื้อ ฯลฯ) — **ห้ามทำ** ตาม design decision

---

## 2. Camera / RTSP Config

```python
# Hikvision channel: main=1xx01, sub=1xx02
# 101 = main stream (คมชัด, ใช้ตอน detect/embed จริง)
# 102 = sub stream (เบา, ใช้ตอน dev/preview)
CAMERA = {
    "camera_id": "cam-01",
    "zone":      "fl2-zoneA",       # ผูกกล้อง 1 ตัว = 1 โซน (mapping ภายนอก)
    "ip":        "192.168.1.64",
    "username":  "admin",
    "password":  "<< จาก env/secret ไม่ hardcode >>",
    "channel":   "101",             # main stream สำหรับ production
    "rtsp_tmpl": "rtsp://{u}:{p}@{ip}:554/Streaming/Channels/{ch}",
}
```

> **ความปลอดภัย:** อย่า hardcode password ในโค้ด — อ่านจาก env var / secret manager
> **Latency vs คุณภาพ:** ให้เลือก channel ได้ผ่าน config; default production = 101

---

## 3. RTSP Capture — ต้องทน + ไม่สะสม latency

โค้ด preview เดิม (`while cap.read()`) ใช้ production ไม่ได้ ต้องแก้ 2 จุด:

```
CLASS RtspSource:
    __init__(url, use_tcp=True):
        # บังคับ TCP transport กัน packet loss:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.url = url
        self._open()

    _open():
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # กัน buffer สะสมเฟรมเก่า

    read_latest() -> frame | None:
        # drop-frame: เคลียร์เฟรมค้างใน buffer เอาเฉพาะล่าสุด
        ok, frame = self.cap.read()
        IF NOT ok:
            self.reconnect()          # ← ห้าม break; reconnect แทน
            RETURN None
        RETURN frame

    reconnect(max_backoff=30s):
        # exponential backoff: 1,2,4,...,30s วนจนต่อได้
        self.cap.release()
        sleep(backoff); self._open()
```

**Requirements:**
- `rtsp_transport=tcp` (Hikvision บน UDP หลุดง่าย)
- `BUFFERSIZE=1` + อ่านทิ้งเฟรมเก่า → ประมวลเฉพาะเฟรมล่าสุดเสมอ
- อ่านไม่ได้ = reconnect แบบ backoff ไม่ใช่ break
- timestamp ของเฟรม CCTV ใช้ **wall-clock ปัจจุบัน** (`datetime.now(timezone.utc)`) ต่างจาก replay ที่คำนวณจาก frame_idx/fps

---

## 4. Detect + Track + Dwell (เหมือน replay logic)

reuse logic เดียวกับ `clip_to_events.py` — ต่างแค่ที่มาของเฟรมกับ timestamp

```
FUNCTION run(source: RtspSource):
    tracks = {}   # track_id -> {still_since, last_centroid, fired, person_frame}
    LOOP:
        frame = source.read_latest()
        IF frame is None: CONTINUE          # กำลัง reconnect
        now = datetime.now(timezone.utc)

        # sample: ประมวลทุก N เฟรม (ปรับตาม fps จริงของ stream)
        IF should_skip(): CONTINUE

        results = yolo.track(frame, persist=True,
                             classes=ITEM_CLASSES, tracker="bytetrack.yaml")

        # (ตัวเลือก) จับเฟรม "คนวางของ": detect person ใกล้ track ที่ยังเคลื่อน
        #   เก็บ frame ล่าสุดที่มีคนอยู่ใกล้ object ก่อนมันนิ่ง → track.person_frame
        #   ** เก็บภาพไว้เทียบใบหน้าเท่านั้น ห้ามวิเคราะห์ลักษณะ **

        FOR each tracked object (bbox, track_id, cls):
            update_dwell(tracks[track_id], centroid, now)
            IF dwell(track_id) >= DWELL_SECONDS AND NOT fired:
                emit_event(frame, bbox, track_id, cls, now,
                           person_frame=tracks[track_id].person_frame)
                tracks[track_id].fired = True
```

Config เดียวกับ replay: `SAMPLE_EVERY`, `DWELL_SECONDS`, `MOVE_TOL`, `ITEM_CLASSES = {24:backpack, 25:umbrella, 26:handbag, 28:suitcase, 63:laptop, 67:cell phone, 73:book}`

---

## 5. Emit Event Contract (output — ต้องตรง schema เดิม)

```
FUNCTION emit_event(frame, bbox, track_id, cls, ts, person_frame):
    x1,y1,x2,y2 = bbox
    crop = frame[y1:y2, x1:x2]
    crop_ref = save(crop)                       # ที่เก็บ object crop
    person_ref = save(person_frame) IF person_frame ELSE None  # เฟรมคนวาง

    event = {
      "schema_version": "1.0",
      "model_version":  "yolo26s_clip-vitb32",  # ต้องตรงกับ Process 2
      "event_id":       hash(camera_id, track_id, ts),
      "source":         "cctv",                 # ← ต่างจาก replay ตรงนี้เท่านั้น
      "camera_id":      CAMERA.camera_id,
      "zone":           CAMERA.zone,
      "capture_ts":     ts.isoformat(),         # wall-clock จริง
      "detect_type":    "abandoned_object",
      "object_class":   ITEM_CLASSES[cls],
      "track_id":       track_id,
      "bbox":           [x1,y1,x2-x1,y2-y1],
      "crop_ref":       crop_ref,
      "person_ref":     person_ref,             # เพิ่มสำหรับเทียบหน้า (nullable)
      "embedding":      clip_embed(crop),       # 512-d normalized
    }
    publish(event)   # Redis Stream / Kafka; dev = เขียน events.jsonl
```

> **Schema note:** ถ้า Event เดิมยังไม่มี `person_ref` ให้เพิ่มเป็น optional field ใน contract (nullable) — ตรงกับ design v3 ที่เก็บเฟรมคนวางไว้เทียบหน้า และ backend ต้อง handle เคส null (ไม่เห็นหน้า → ไป backup คำถามบริบท)

---

## 6. Config

```python
ACTIVE_MODEL_VERSION = "yolo26s_clip-vitb32"
CHANNEL_DEV, CHANNEL_PROD = "102", "101"
SAMPLE_EVERY  = 3          # CCTV fps มักต่ำกว่าคลิป ปรับตามจริง
DWELL_SECONDS = 8.0
MOVE_TOL      = 25
RECONNECT_MAX_BACKOFF = 30
RTSP_TRANSPORT = "tcp"
```

---

## 7. Acceptance Tests

```
T1  ดึง stream 101 ได้ frame ไม่ null ต่อเนื่อง > 60 วิ (สภาพเน็ตปกติ)
T2  ถอดสาย/บล็อกกล้อง → source ไม่ crash, reconnect กลับมาได้เอง
T3  วางวัตถุนิ่งเกิน DWELL_SECONDS → ยิง event 1 ครั้ง (ไม่ซ้ำต่อเฟรม)
T4  event ที่ยิงออก parse เป็น Event schema ผ่าน + source=="cctv" + model_version ตรง
T5  latency: เวลาระหว่าง capture_ts กับ publish < 2 วิ (ยืนยันว่าไม่สะสม buffer)
T6  ยิง event จาก cctv เข้า Process 2 (matching) แล้วทำงานได้เหมือน event จาก replay
    → ยืนยันว่า contract เดียวกันจริง (จุดขายของ Event Contract)
T7  person_ref = null เมื่อจับเฟรมคนวางไม่ได้ → backend ไม่พัง (ไป backup)
```

---

## 8. Module Layout

```
edge_cctv/
  rtsp_source.py    # §3 RtspSource (reconnect + drop-frame)  ← เขียน+ทดสอบก่อน
  detector.py       # §4 YOLO track + dwell
  person_capture.py # §4 จับเฟรมคนวาง (optional, nullable)
  emitter.py        # §5 ประกอบ + publish Event
  config.py         # §6
  run.py            # ต่อทั้งหมด เป็น entrypoint
  tests/
    test_rtsp.py    # T1-T2, T5
    test_pipeline.py# T3-T4, T6-T7
```

**ลำดับสร้าง:** `rtsp_source.py` (แก้ latency/reconnect ให้เสถียรก่อน) → `detector.py` → `emitter.py` → ต่อ `run.py` → person_capture ทีหลัง (optional)

> เริ่มที่ `rtsp_source.py` เพราะถ้า stream ไม่เสถียร ทุกอย่างข้างบนไม่มีความหมาย — และมันคือจุดที่โค้ด preview เดิมยังขาด
