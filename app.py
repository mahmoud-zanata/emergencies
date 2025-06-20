import gradio as gr
from transformers import VivitForVideoClassification, VivitImageProcessor
import torch
import cv2
import os
import tempfile

model = VivitForVideoClassification.from_pretrained("mahmoud0125651561/emergency")
processor = VivitImageProcessor.from_pretrained("mahmoud0125651561/emergency")

def predict(video_file):
    # احفظ الملف مؤقتًا
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(video_file.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)
    frames = []
    while len(frames) < model.config.num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (model.config.image_size, model.config.image_size))
        frames.append(frame)
    cap.release()
    os.remove(temp.name)

    if len(frames) < model.config.num_frames:
        return "Not enough frames in video."

    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(-1).item()
        pred_label = model.config.id2label[pred_id]
    
    return pred_label

demo = gr.Interface(fn=predict, inputs="video", outputs="text")
demo.launch()
