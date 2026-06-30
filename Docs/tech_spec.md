# UpFound — Technical Spec

> สถาปัตยกรรม, data contract, pipeline และ tech stack ของระบบ Lost & Found AI
> หลักออกแบบ: decouple ทุกอย่างผ่าน **Event Contract** เพื่อให้ source (คลิป / Edge / CCTV) สลับกันได้โดย downstream ไม่ต้องแก้

---

## 1. Architecture Overview

ระบบแบ่งเป็น 3 ส่วน ตามผู้รับผิดชอบ โดยมี Event Contract เป็นเส้นแบ่ง:

| ส่วน            | ผู้รับผิดชอบ | ขอบเขต                                                                            |
| --------------- | ------------ | --------------------------------------------------------------------------------- |
| **Edge AI**     | คน A         | source → detect → encode → emit event (replay / CCTV adapter / fallback)          |
| **Workflow AI** | คน B         | ingest → cascade match → journey → confidence/abstain → verification → claim pass |
| **Web App**     | คน C         | ฟอร์มแจ้ง, แสดง candidate + หลักฐาน, พิสูจน์ตัวตน, รับ QR                         |

```
[คลิป/Edge/CCTV] --produce--> | Event Contract | --consume--> [Workflow AI] <--> [Web App]
        (คน A)                  (เส้นแบ่ง)                   (คน B)         (คน C)
```

## 2. Data Flow (สองเส้น)

**เส้น A — ตรวจจับ (เขียน):** source → detect abandoned object → crop + embed → `Event` → message queue → log + vector DB

**เส้น B — ตามหา (อ่าน):** user query → cascade match กับ vector DB/log → journey + candidates → confidence gate → verification → claim pass

ทั้งสองบรรจบที่ขั้น match: เส้น A เติม gallery, เส้น B query gallery

## 3. Event Contract

data contract กลางที่ทุกฝ่ายผูกไว้ (ใช้ Pydantic validate + `schema_version`)

```json
{
  "schema_version": "1.0",
  "model_version": "yolo11n_clip-vitb32",
  "event_id": "a1b2c3d4e5f6",
  "source": "replay | edge | cctv",
  "camera_id": "cam-03",
  "zone": "fl2-zoneA",
  "capture_ts": "2026-07-16T14:32:05.120Z",
  "detect_type": "abandoned_object",
  "object_class": "backpack",
  "track_id": 7,
  "bbox": [320, 180, 90, 140],
  "crop_ref": "s3://.../cam03_1432_obj7.jpg",
  "embedding": [0.12, -0.88, "... 512-d, L2-normalized"]
}
```

**กฎเหล็ก 2 ข้อ (ห้ามพลาด):**

1. **`model_version`** — embedding คนละรุ่นเทียบกันไม่ได้ เปลี่ยนโมเดล = ต้อง re-embed ทั้ง gallery
2. **`capture_ts` = เวลาถ่ายจริง** (ไม่ใช่เวลา ingest) — ไม่งั้น spatio-temporal filter เพี้ยน โดยเฉพาะ replay

## 4. Pipeline: คลิป → Event (source แรก)

deterministic (frame index → timestamp ตายตัว) เหมาะเป็น replay simulator + ชุดทดสอบ

**Logic 6 ขั้น:**

1. **Decode + sample** — `capture_ts = CLIP_START + frame_idx / fps`, ประมวลทุก N เฟรม
2. **Detect + track** — YOLO หาคลาสกระเป๋า/สัมภาระ + แปะ `track_id` (ByteTrack)
3. **Dwell logic** — centroid ขยับเกิน tol = reset timer; นิ่งเกิน `DWELL_SECONDS` = abandoned
4. **Fire once / track** — flag กันยิงซ้ำ (1 ของ = 1 event ไม่ใช่ 1 เฟรม = 1 event)
5. **Crop + embed** — ตัดตาม bbox → CLIP encode → normalize
6. **Emit** — ประกอบ `Event` → เขียน queue

> เวอร์ชันเดโมใช้ dwell logic เรียบง่าย (ของนิ่งนานเกินกำหนด) เพราะ Edge alert เป็นแค่ advisory อยู่แล้ว เวอร์ชันแกร่งขึ้นค่อยเพิ่ม person-proximity (abandoned เมื่อไม่มีคนในรัศมีเกิน X วิ)

## 5. Matching Cascade (consumer side)

จับคู่แบบไล่ระดับ — ตัด candidate ทิ้งจากถูกที่สุดไปแพงที่สุด

1. **Spatio-temporal filter** — กรองด้วยโซน + ช่วงเวลา (ตัดทิ้ง ~90% ก่อน, ถูกสุด)
2. **Attribute match** — ประเภท/สี/ยี่ห้อ จาก structured form
3. **CLIP rerank** — image→image (ถ้ามีรูป) หรือ **text→image** (ถ้ามีแค่คำบรรยาย) เป็นตัว rerank ไม่ใช่ตัวตัดสินหลัก

## 6. Confidence & Abstain

confidence ต้อง **calibrated** และระบบต้อง **abstain ได้**:

| ระดับ | การกระทำ                                   |
| ----- | ------------------------------------------ |
| สูง   | เสนอ candidate top-3 พร้อมหลักฐาน          |
| กลาง  | ส่ง staff review dashboard (HITL)          |
| ต่ำ   | ตอบ "ยังไม่เจอ" + แจ้งเตือนเมื่อเจอภายหลัง |

ออกแบบให้กล้าพูดว่าไม่รู้ → ลด false positive ของวัตถุหน้าตาเหมือนกัน (เช่น กระเป๋าดำ)

## 7. Ownership Verification

ระบบมองไม่เห็นของข้างใน จึงถาม **เฉพาะสิ่งที่ฟุตเทจยืนยันได้** แล้วเทียบกับเฟรมที่บันทึกตัวเจ้าของ:

- นั่ง/อยู่โซนไหน
- มากับกี่คน
- เสื้อสีอะไร
- เดินเข้ามาจากทางไหน
- วางของไว้ประมาณกี่โมง

ผ่าน → QR claim pass (provisional) · ไม่ผ่าน → คิวเจ้าหน้าที่
**ของข้างในจริง** ให้เจ้าหน้าที่เปิดยืนยันตอนส่งมอบ (HITL ซื่อสัตย์)

## 8. Tech Stack

| ส่วน                 | เลือกใช้                                   | เหตุผล                                                    |
| -------------------- | ------------------------------------------ | --------------------------------------------------------- |
| ภาษา                 | Python                                     | —                                                         |
| Decode               | OpenCV (`VideoCapture`)                    | คลิปไฟล์เดียวพอ; ขยับไป PyAV/decord ถ้าต้องการ perf/codec |
| Detection + Tracking | Ultralytics YOLO v11 (ByteTrack)           | COCO มี backpack/handbag/suitcase                         |
| Embedding            | OpenCLIP ViT-B/32 (512-d)                  | รองรับ image→image และ text→image                         |
| Schema               | Pydantic                                   | validate + versioning                                     |
| Transport            | JSONL (เดโม) → Redis Stream / Kafka (จริง) | decouple + กัน backpressure                               |
| Vector DB            | Qdrant / pgvector                          | ค้น similarity                                            |
| Serialization (จริง) | protobuf / avro                            | embedding เป็น float vector ใหญ่ JSON เปลือง              |

## 9. Privacy & Security (by design)

- ประมวลผลบน **edge** — วิดีโอดิบไม่ออกจากกล่อง ส่งขึ้นแค่ metadata + crop + embedding
- **เบลอใบหน้า** เป็น default
- **retention สั้น** — ลบ raw อัตโนมัติ เก็บเฉพาะเคส active
- access control + audit log
- **ตัด face match / biometric ออกจาก core**

## 10. Demo Setup & Fallback

- source หลักวันงาน: เลือกได้ระหว่าง CCTV สด หรือ replay — Event Contract ทำให้ downstream ไม่รู้ความต่าง
- **fallback switch** CCTV ↔ replay ต้องสลับได้ < 5 วินาที เผื่อกล้องสดหลุด (เน็ต/RTSP/แสง)
- ถ่ายคลิป scenario ผ่านมุมเดียวกับ CCTV เก็บไว้ล่วงหน้า
- รันทุกอย่าง local ตอน demo (ไม่พึ่ง cloud), พก 4G router สำรอง

## 11. คำถามด้านเทคนิคที่ยังเปิดอยู่

- [ ] Edge AI ที่จะใช้ เก็บ embedding ของ "ตัววัตถุ" ตอน detect เลยไหม หรือเก็บแค่เฟรม? (ชี้ขาดว่าต้องมี reverse video search หรือไม่)
- [ ] เลือก embedding model สำหรับ cross-domain (มือถือชัด vs CCTV เบลอ) — CLIP พอไหม หรือต้อง fine-tune / ใช้ ReID
- [ ] CCTV ที่งานดึง RTSP/ONVIF ได้ไหม + เน็ตหน้างาน (ต้องเคลียร์ก่อน Gate 1)
- [ ] schema migration strategy เมื่อ `schema_version` / `model_version` เปลี่ยน
