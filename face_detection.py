import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.8,
)

cap = cv2.VideoCapture(0)

with vision.FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        if results.detections:
            for detection in results.detections:
                bbox = detection.bounding_box
                x, y = bbox.origin_x, bbox.origin_y
                bw, bh = bbox.width, bbox.height
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                score = detection.categories[0].score
                cv2.putText(
                    frame,
                    f"{score:.2f}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("MediaPipe Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
