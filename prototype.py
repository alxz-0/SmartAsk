
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Membuka kamera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Kamera tidak dapat dibuka.")

# 2. Memuat model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# 3. Konfigurasi deteksi
threshold = 0.2
confidence_threshold = 0.5

try:
    while True:
        # 4. Membaca frame kamera
        ret, frame = camera.read()

        if not ret:
            print("Gagal membaca frame kamera.")
            break

        # 5. Mendeteksi pose semua orang
        results = model(frame, verbose=False)
        result = results[0]

        # Menggambar kerangka pose dari YOLO
        annotated_frame = result.plot()

        people_count = 0

        # 6. Memeriksa apakah ada keypoint terdeteksi
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()

            people_count = len(keypoints_xy)

            # 7. Memproses setiap orang
            for person_index, person in enumerate(keypoints_xy):
                confidence = keypoints_conf[person_index]

                # Indeks keypoint COCO
                left_shoulder = person[5]
                right_shoulder = person[6]
                left_wrist = person[9]
                right_wrist = person[10]
                nose = person[0]
                left_eye = person[1]
                right_eye = person[2]

                # Periksa confidence titik yang diperlukan
                required_points = [5, 6, 9, 10]

                if not all(
                    confidence[i] >= confidence_threshold
                    for i in required_points
                ):
                    continue

                # 8. Menghitung jarak antarbahu
                shoulder_width = np.linalg.norm(
                    left_shoulder - right_shoulder
                )

                if shoulder_width < 1:
                    continue

                # 9. Menghitung rasio ketinggian tangan
                left_ratio = (
                    left_eye[1] - left_wrist[1]
                ) / shoulder_width

                right_ratio = (
                    right_eye[1] - right_wrist[1]
                ) / shoulder_width

                # 10. Menentukan status tangan
                left_raised = left_ratio >= threshold
                right_raised = right_ratio >= threshold

                if left_raised or right_raised:
                    status = "TANGAN TERANGKAT"
                    color = (0, 255, 0)
                else:
                    status = "TANGAN NORMAL"
                    color = (0, 0, 255)

                # 11. Menampilkan status di dekat kepala
                x = int(nose[0])
                y = int(nose[1])

                cv2.putText(
                    annotated_frame,
                    status,
                    (x, max(y - 15, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # 12. Menampilkan jumlah orang
        cv2.putText(
            annotated_frame,
            f"People detected: {people_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # 13. Menampilkan video
        cv2.imshow(
            "HearMe - Hand Raising Detection",
            annotated_frame
        )

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # 14. Membersihkan resource
    camera.release()
    cv2.destroyAllWindows()
