import cv2 
# import mediapipe as mp
from ultralytics import YOLO
import numpy as np 
import time 
import sounddevice as sd
import soundfile as sf

# default kamera (testing)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Kamera tidak dapat dibuka.")

#  YOLO pose model (dengan model YOLO8n)
model = YOLO('yolov8n-pose.pt')

# default kamera secara lebar & tinggi
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# menetapkan codec video dan membuat objek VideoWriter untuk menyimpan video

# codec video & audio/suara
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# nama file output, codec, fps, ukuran frame
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))  # nama file output, codec, fps, ukuran frame

try:
    while True:
        # membaca frame dari kamera
        ret, frame = camera.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]
       
        # Menggambar kerangka pose dari YOLO
        annotated_frame = result.plot()
       
        people_count = 0

        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            people_count = len(keypoints_xy)

        for r in results:
            if r.keypoints is None:
                points = r.keypoints.xy.cpu().numpy()

                for person_index, person in enumerate(points):
                    confidence = keypoints_conf[person_index]
                    # COCO keypoint index
                    nose = person[0]

                    # mata 
                    right_eye = person[1]
                    left_eye = person[2]

                    # telinga
                    right_ear = person[3]
                    left_ear = person[4]

                    # bahu 
                    right_shoulder = person[5]
                    left_shoulder = person[6]
                    # pergelangan tangan 

                    right_wrist = person[9]
                    left_wrist = person[10]

                    raise_hand = False

                    required_points = [5, 6, 9, 10]

                    if not all(confidence[i]>= 0.5 for i in required_points):
                        continue

                    # menghitung jarak antar bahu 
                    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                    if shoulder_width < 1:
                        continue
                    
                    left_raise = (left_wrist[1] < left_shoulder[1]) and (left_wrist[0] > left_shoulder[0])
                    right_raise = (right_wrist[1] < right_shoulder[1]) and (right_wrist[0] < right_shoulder[0])

                    left_raised = left_raise >= 0.2
                    right_raised = right_raise >= 0.2

                    if left_raised or right_raised:
                        raise_hand = True
                        print(f"Orang ke-{person_index + 1} mengangkat tangan.")
                        color = (0, 255, 0)  # hijau
                    else:
                        print(f"Tangan normal.")
                        color = (0, 0, 255)  # merah

        # menampilkan jumlah orang yang terdeteksi
        cv2.putText(annotated_frame, f"Jumlah orang: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # menampilkan frame yang dianotasi
        cv2.imshow("Pose Detection", annotated_frame)
        
finally:
    # melepaskan kamera dan menutup jendela
    camera.release()
#out.release()
cv2.destroyAllWindows()