# Workshop 4 — Real-Time Face Detection with MediaPipe & OpenCV

โปรเจกต์นี้เป็นการตรวจจับใบหน้า (Face Detection) แบบ real-time ผ่านกล้องเว็บแคม โดยใช้ MediaPipe Tasks API และ OpenCV

---

## สารบัญ

- [ภาพรวม](#ภาพรวม)
- [ความต้องการของระบบ](#ความต้องการของระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [วิธีใช้งาน](#วิธีใช้งาน)
- [อธิบายโค้ด](#อธิบายโค้ด)
- [พารามิเตอร์ที่ปรับได้](#พารามิเตอร์ที่ปรับได้)
- [ปัญหาที่พบบ่อย](#ปัญหาที่พบบ่อย)

---

## ภาพรวม

โปรแกรมจะเปิดกล้องเว็บแคม แล้วตรวจจับใบหน้าในแต่ละเฟรมแบบ real-time โดย:

- วาด **bounding box** สีเขียวรอบใบหน้าที่ตรวจพบ
- แสดง **confidence score** (0.00–1.00) เหนือกรอบใบหน้า
- กด `q` เพื่อปิดโปรแกรม

### Library ที่ใช้

| Library | บทบาท |
|---|---|
| `mediapipe` (0.10+) | โมเดล AI สำหรับตรวจจับใบหน้า |
| `opencv-python` | อ่านภาพจากกล้อง, วาดกราฟิก, แสดงผล |

---

## ความต้องการของระบบ

- Python 3.8–3.11
- เว็บแคม (built-in หรือ USB)
- macOS / Windows / Linux

---

## การติดตั้ง

### 1. สร้าง Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 2. ติดตั้ง Dependencies

```bash
pip install mediapipe opencv-python
```

### 3. ดาวน์โหลด Model File

โมเดล `blaze_face_short_range.tflite` จำเป็นต้องอยู่ในโฟลเดอร์เดียวกับไฟล์ `.py`

```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
    'blaze_face_short_range.tflite'
)
print('Downloaded successfully')
"
```

---

## โครงสร้างโปรเจกต์

```
workshop-4/
├── face_detection.py               # โค้ดหลัก
├── blaze_face_short_range.tflite   # โมเดล AI (ต้องดาวน์โหลด)
└── README.md                       # ไฟล์นี้
```

---

## วิธีใช้งาน

```bash
python face_detection.py
```

หน้าต่างกล้องจะเปิดขึ้นมา กด **`q`** เพื่อปิดโปรแกรม

---

## อธิบายโค้ด

### 1. โหลดโมเดลและตั้งค่า Detector

```python
base_options = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
)
```

- `model_asset_path` — ชี้ไปยังไฟล์โมเดล `.tflite`
- `min_detection_confidence` — ค่า threshold ต่ำสุดที่จะนับว่าตรวจพบใบหน้า (0.0–1.0)

### 2. เปิดกล้องและวน Loop

```python
cap = cv2.VideoCapture(0)
```

- `0` คือกล้องตัวแรกของระบบ (built-in webcam) เปลี่ยนเป็น `1`, `2`, ... สำหรับกล้องตัวอื่น

### 3. แปลงสีก่อนส่งให้ MediaPipe

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
```

OpenCV อ่านภาพในรูปแบบ **BGR** แต่ MediaPipe ต้องการ **RGB** จึงต้องแปลงก่อนทุกครั้ง

### 4. ตรวจจับและวาด Bounding Box

```python
results = detector.detect(mp_image)
if results.detections:
    for detection in results.detections:
        bbox = detection.bounding_box
        x, y = bbox.origin_x, bbox.origin_y
        bw, bh = bbox.width, bbox.height
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
```

- `bounding_box` ใน MediaPipe Tasks API ให้พิกเซลจริง (ไม่ใช่ relative coordinates แบบ API เก่า)
- `(0, 255, 0)` คือสีเขียวในรูปแบบ BGR

### 5. แสดง Confidence Score

```python
score = detection.categories[0].score
cv2.putText(frame, f"{score:.2f}", (x, y - 8), ...)
```

- `score` มีค่า 0.0–1.0 ยิ่งใกล้ 1.0 ยิ่งมั่นใจว่าเป็นใบหน้า

---

## พารามิเตอร์ที่ปรับได้

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `min_detection_confidence` | `0.5` | ลดลงเพื่อให้ตรวจจับได้ง่ายขึ้น, เพิ่มขึ้นเพื่อลด false positive |
| `cv2.VideoCapture(0)` | `0` | เปลี่ยนเลขเพื่อเลือกกล้องอื่น |
| สีกรอบ `(0, 255, 0)` | สีเขียว | เปลี่ยนได้ตามต้องการ เช่น `(0, 0, 255)` สีแดง |

---

## ปัญหาที่พบบ่อย

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`

MediaPipe 0.10+ ยกเลิก `solutions` API แล้ว โค้ดในโปรเจกต์นี้ใช้ `mediapipe.tasks` API ซึ่งรองรับเวอร์ชันใหม่แล้ว

### ชื่อไฟล์ชนกับ Library

ห้ามตั้งชื่อไฟล์ว่า `mediapipe.py` เพราะจะ shadow library ทำให้ import ไม่ได้

### กล้องไม่เปิด

- ตรวจสอบว่าไม่มีแอปอื่นใช้กล้องอยู่
- ลองเปลี่ยน `VideoCapture(0)` เป็น `VideoCapture(1)`

### ไม่พบไฟล์ `.tflite`

ให้รันคำสั่งดาวน์โหลดในขั้นตอนการติดตั้งอีกครั้ง และตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกับ `face_detection.py`
