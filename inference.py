import cv2
from ultralytics import YOLO
import time

model = YOLO(r"C:\Users\Rose\Desktop\internship\PPE DETECTION\models\best.pt")


cap=cv2.VideoCapture(r"PUSH/sample_video.mp4")

while cap.isOpened():
    success, frame=cap.read()
    if not success:
        break

    results=model(frame)

    annotated_frame=results[0].plot()

    cv2.imshow("PPE DETECTION",annotated_frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()