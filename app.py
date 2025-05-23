# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12icbNPqYit84S0aDJbvxuHQTJtQ8hS9x
"""

from flask import Flask, request, jsonify
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

def send_email(subject, body, to_email):
    EMAIL_ADDRESS = 'zanatamahmoud7@gmail.com'
    EMAIL_PASSWORD = 'sbgy sveh ysjh zcni'

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False

@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # هنا المفروض يتم معالجة الفيديو والموديل
    result = "emergency"  # استبدله بنتيجة الموديل لاحقًا

    if result == "emergency":
        email_sent = send_email(
            subject="Emergency Alert 🚨",
            body="An emergency situation has been detected!",
            to_email="target_email@example.com"
        )
        return jsonify({"status": "emergency detected", "email_sent": email_sent})
    else:
        return jsonify({"status": "no emergency detected"})

if __name__ == '__main__':
    app.run(debug=True)