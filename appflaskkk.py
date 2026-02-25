"""
Flask Web Application for Philippine License Plate Detection System
===================================================================
Contains:
- Flask Routes
- Model Loading
- File Upload Handling
- Debug Endpoints
"""

import os
import cv2
import shutil
import time
import json
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory, jsonify

# Import models and pipeline
from models import (
    RealESRGANEnhancer, PhilippinePlateOCR, 
    SVMPlateTypeClassifier, SVMCodingViolationDetector
)
from pipeline import process_image_with_yolo, save_detected_plates_to_txt

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Custom secure_filename
try:
    from werkzeug.utils import secure_filename
except Exception:
    import re
    def secure_filename(filename):
        if not filename:
            return filename
        filename = filename.strip().replace(' ', '_')
        filename = re.sub(r'[^A-Za-z0-9._-]', '_', filename)
        return filename

# ----------------------------
# 🔹 Flask App Setup
# ----------------------------

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/results'
app.config['SVM_FOLDER'] = 'static/svm'
app.config['ENHANCEMENT_FOLDER'] = 'static/enhancements'
app.config['VIOLATION_FOLDER'] = 'static/violations'
app.config['PLATES_TXT_FOLDER'] = 'static/plates_txt'

# Create directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 
               app.config['SVM_FOLDER'], app.config['ENHANCEMENT_FOLDER'], 
               app.config['VIOLATION_FOLDER'], app.config['PLATES_TXT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)


# ----------------------------
# 🔹 Load Models
# ----------------------------

print("\n" + "="*60)
print("LOADING 4-STAGE PIPELINE MODELS")
print("="*60 + "\n")

# Stage 1: YOLO Detection
yolo_model = None
if YOLO_AVAILABLE:
    yolo_paths = [
        r"C:\Users\DELL\Videos\systematic\thesis-project\yolo\models\best.pt",
        r"C:\Users\DELL\Videos\systematic\thesis-project\yolo\best.pt",
    ]
    
    for yolo_path in yolo_paths:
        if os.path.exists(yolo_path):
            try:
                yolo_model = YOLO(yolo_path)
                print(f"✅ YOLO Detection loaded")
                break
            except Exception as e:
                print(f"❌ Error loading YOLO: {e}")
    
    if yolo_model is None:
        try:
            print("🔄 Trying pretrained YOLOv8 model...")
            yolo_model = YOLO('yolov8n.pt')
            print("✅ Loaded pretrained YOLOv8n model")
        except Exception as e:
            print(f"❌ Could not load YOLO model: {e}")

# Stage 2: Real-ESRGAN Enhancement
print(f"\n🔍 Loading Real-ESRGAN Model...")
enhancer = RealESRGANEnhancer()
print(f"✅ Enhancement loaded")

# Stage 3: OCR
print(f"🔍 Loading OCR...")
ph_ocr = PhilippinePlateOCR()
print(f"✅ OCR loaded")

# Stage 4: SVM Models
print(f"\n🔍 Loading SVM Models...")
svm_type_classifier = SVMPlateTypeClassifier()
svm_violation_detector = SVMCodingViolationDetector()
print(f"✅ SVM Classifiers loaded")

print("\n" + "="*60)
print("SYSTEM STATUS:")
print(f"1. 🎯 YOLO Detection: {'✅' if yolo_model else '❌'}")
print(f"2. 🎨 Enhancement: {'✅' if enhancer.enhancement_available else '❌'}")
print(f"3. 🔤 OCR: {'✅' if ph_ocr.easyocr_reader else '❌'}")
print(f"4. 🤖 SVM Type: {'✅' if svm_type_classifier.model else '❌'}")
print(f"5. ⚠️ SVM Violation: {'✅' if svm_violation_detector.model else '❌'}")
print("="*60 + "\n")


# ----------------------------
# 🔹 Flask Routes
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Process single image through pipeline"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        print(f"\n🖼️ Processing: {filename}")
        
        image = cv2.imread(input_path)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        start_time = time.time()
        
        result = process_image_with_yolo(
            image, filename, app.config, yolo_model, 
            enhancer, ph_ocr, svm_type_classifier, svm_violation_detector
        )
        
        processing_time = time.time() - start_time
        
        if result and 'success' in result and result['success']:
            response_data = {
                'success': True,
                'processing_time': f"{processing_time:.1f}s",
                'filename': filename,
                'original_image_url': f'/static/svm/{result.get("original_image")}',
                'annotated_image_url': f'/static/results/{result.get("annotated_image")}',
                'total_plates_detected': result.get('total_plates_detected', 0),
                'detected_plates': result.get('detected_plates', []),
            }
            
            if result.get('detected_plates'):
                first_plate = result['detected_plates'][0]
                response_data.update({
                    'plate_text': first_plate.get('text', ''),
                    'ocr_confidence': first_plate.get('ocr_confidence', 0),
                    'svm_type_classification': first_plate.get('svm_type_classification'),
                    'violation_detection': first_plate.get('violation_detection')
                })
                
                if 'comparison_image_url' in first_plate:
                    response_data['comparison_image_url'] = f'/static/{first_plate["comparison_image_url"]}'
            
            print(f"✅ Processed in {processing_time:.1f}s")
            return jsonify(response_data)
        else:
            error_msg = result.get('error', 'Processing failed')
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        if filename.startswith('results/'):
            actual = filename.replace('results/', '', 1)
            return send_from_directory(app.config['OUTPUT_FOLDER'], actual)
        elif filename.startswith('enhancements/'):
            actual = filename.replace('enhancements/', '', 1)
            return send_from_directory(app.config['ENHANCEMENT_FOLDER'], actual)
        elif filename.startswith('svm/'):
            actual = filename.replace('svm/', '', 1)
            return send_from_directory(app.config['SVM_FOLDER'], actual)
        elif filename.startswith('violations/'):
            actual = filename.replace('violations/', '', 1)
            return send_from_directory(app.config['VIOLATION_FOLDER'], actual)
        else:
            return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': f'File not found: {filename}'}), 404


# ----------------------------
# 🔹 Debug Endpoints
# ----------------------------

@app.route('/debug/status')
def debug_status():
    return jsonify({
        'yolo_loaded': yolo_model is not None,
        'enhancement_available': enhancer.enhancement_available,
        'ocr_available': ph_ocr.easyocr_reader is not None,
        'svm_type_loaded': svm_type_classifier.model is not None,
        'svm_violation_loaded': svm_violation_detector.model is not None
    })

@app.route('/debug/enhancements')
def debug_enhancements():
    import glob
    files = glob.glob(os.path.join(app.config['ENHANCEMENT_FOLDER'], '*comparison*'))
    return jsonify({
        'count': len(files),
        'files': [os.path.basename(f) for f in files if os.path.getsize(f) > 0]
    })


# ----------------------------
# 🔹 Main
# ----------------------------

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 PHILIPPINE LICENSE PLATE DETECTION SYSTEM")
    print("1. 🎯 YOLO Detection")
    print("2. 🎨 Real-ESRGAN Enhancement") 
    print("3. 🔤 Advanced OCR")
    print("4. 🤖 SVM Classification")
    print("="*60 + "\n")
    print("🌐 http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)