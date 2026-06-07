import torch
from ultralytics.nn.tasks import DetectionModel
import torch.nn as nn

# Add safe globals for PyTorch serialization
torch.serialization.add_safe_globals([DetectionModel, nn.Sequential, nn.Module, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d])

from flask import Flask, request, render_template, redirect, url_for
import os
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load YOLO model
model = YOLO('best.pt')

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image
        result = process_image(filepath)
        
        return render_template('result.html', result=result)

def process_image(image_path):
    # Run YOLO prediction
    results = model.predict(image_path, conf=0.5)
    
    # Load image
    img = cv2.imread(image_path)
    
    # First pass: determine if helmet is present
    helmet_present = False
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls == 0:  # with helmet
                helmet_present = True
                break
        if helmet_present:
            break
    
    # Second pass: draw boxes
    number_plate_boxes = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if cls == 0:  # with helmet - no violation, green
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.putText(img, 'Helmet (No Violation)', (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            elif cls == 1:  # without helmet - violation, red
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(img, 'Without Helmet (Violation)', (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif cls == 2:  # rider - violation, red
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(img, 'Rider', (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif cls == 3 and not helmet_present:  # number plate - only if violation
                number_plate_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                cv2.putText(img, 'Number Plate', (int(x1), int(y1)-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    
    # Extract number plate text only if violation detected
    number_plate_text = ""
    if not helmet_present and number_plate_boxes:
        for x1, y1, x2, y2 in number_plate_boxes:
            cropped = img[int(y1):int(y2), int(x1):int(x2)]
            text_results = reader.readtext(cropped)
            number_plate_text += ' '.join([text[1] for text in text_results]) + ' '
        number_plate_text = number_plate_text.strip()
    
    # Save annotated image
    annotated_filename = 'annotated_' + os.path.basename(image_path)
    annotated_path = os.path.join('static', annotated_filename)
    cv2.imwrite(annotated_path, img)
    
    if helmet_present:
        violation = "No violation detected"
    else:
        violation = "Violation detected"
    
    return {
        'violation': violation,
        'number_plate': number_plate_text,
        'annotated_image': annotated_filename
    }

if __name__ == '__main__':
    app.run(debug=True)