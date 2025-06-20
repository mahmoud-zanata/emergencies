import gradio as gr
from transformers import VivitForVideoClassification, VivitImageProcessor
import torch
import cv2
import os
import tempfile

# تحميل الموديل والمعالج من Hugging Face
model = VivitForVideoClassification.from_pretrained("mahmoud0125651561/emergency")
processor = VivitImageProcessor.from_pretrained("mahmoud0125651561/emergency")

def predict(video_file):
    # حفظ الفيديو مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    # استخراج الفريمات
    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    while len(frames) < model.config.num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (model.config.image_size, model.config.image_size))
        frames.append(frame)

    cap.release()
    os.remove(temp_video_path)

    if len(frames) < model.config.num_frames:
        return "❌ Not enough frames in video."

    # تجهيز الإدخال
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = logits.argmax(-1).item()
        pred_label = model.config.id2label[pred_id]

    return f"✅ Prediction: {pred_label}"

# واجهة Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Prediction"),
    title="🚨 Emergency Detection",
    description="Upload a video to detect emergency cases using Vivit model."
)

demo.launch()
