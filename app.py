import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile

model = YOLO(r"C:\Users\Rose\Desktop\internship\PPE DETECTION\models\best.pt")
#import torch

#model = torch.load(r"C:\Users\Rose\Desktop\internship\PPE DETECTION\models\best.pt", weights_only=False)


def detect_ppe(video):
    cap = cv2.VideoCapture(video)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    return temp_file.name

demo = gr.Interface(fn=detect_ppe,
                    inputs=gr.Video(label="Upload Video"),
                    outputs=gr.Video(label="Output with PPE Detection"))

demo.launch()