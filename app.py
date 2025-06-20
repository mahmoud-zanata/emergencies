from flask import Flask, request, jsonify
from transformers import VivitForVideoClassification, VivitImageProcessor
import torch
import tempfile
import cv2
import numpy as np
import os

app = Flask(__name__)

# تحميل الموديل والمعالج من Hugging Face
model = VivitForVideoClassification.from_pretrained("mahmoud0125651561/emergency")
processor = VivitImageProcessor.from_pretrained("mahmoud0125651561/emergency")

@app.route('/')
def home():
    return '🚀 Emergency Detection API is Running!'

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No video uploaded."}), 400

    # احفظ الفيديو مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        file.save(temp_video.name)
        video_path = temp_video.name

    # استخراج الفريمات
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < model.config.num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (model.config.image_size, model.config.image_size))
        frames.append(frame)

    cap.release()
    os.remove(video_path)

    if len(frames) < model.config.num_frames:
        return jsonify({"error": "Not enough frames in video."})

    # تجهيز البيانات
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = logits.argmax(-1).item()
        pred_label = model.config.id2label[pred_id]

    return jsonify({"prediction": pred_label})
