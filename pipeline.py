"""
Processing Pipeline for Philippine License Plate Detection System
===============================================================
Contains:
- 4-Stage Pipeline Processing
- Image Processing Functions
- Video Processing Functions
- Text Report Generation
"""

import os
import cv2
import numpy as np
import time
from datetime import datetime

# Import models
from models import RealESRGANEnhancer, PhilippinePlateOCR, SVMPlateTypeClassifier, SVMCodingViolationDetector

# ----------------------------
# 🔹 Save Detected Plates to Text File
# ----------------------------

def save_detected_plates_to_txt(plates_data, filename_prefix, save_dir):
    """Save detected plates information to a text file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f"{filename_prefix}_detected_plates_{timestamp}.txt"
        txt_path = os.path.join(save_dir, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("DETECTED LICENSE PLATES REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Unique Plates Detected: {len(plates_data)}\n\n")
            
            for i, plate in enumerate(plates_data, 1):
                f.write(f"[{i}] PLATE NUMBER: {plate.get('text', 'N/A')}\n")
                f.write(f"    OCR Confidence: {plate.get('ocr_confidence', 0)*100:.1f}%\n")
                f.write(f"    OCR Method: {plate.get('ocr_method', 'N/A')}\n")
                f.write(f"    Raw OCR Text: {plate.get('raw_text', 'N/A')}\n")
                
                if 'svm_type_classification' in plate:
                    f.write(f"    Plate Type: {plate.get('svm_type_classification', 'N/A')}\n")
                    f.write(f"    Type Confidence: {plate.get('svm_type_confidence', 0)*100:.1f}%\n")
                
                if 'violation_detection' in plate:
                    violation = plate['violation_detection']
                    f.write(f"    Violation Detected: {'YES' if violation['has_violation'] else 'NO'}\n")
                    if violation['has_violation']:
                        f.write(f"    Violation Type: {violation['violation_type']}\n")
                        f.write(f"    Last Digit: {violation['last_digit']}\n")
                        f.write(f"    Hour: {violation['hour']:02d}:00\n")
                        f.write(f"    Rush Hour: {'YES' if violation['is_rush_hour'] else 'NO'}\n")
                
                f.write("-" * 40 + "\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 60 + "\n")
            
            total_violations = sum(1 for plate in plates_data if plate.get('violation_detection', {}).get('has_violation', False))
            f.write(f"Total Plates: {len(plates_data)}\n")
            f.write(f"Total Violations: {total_violations}\n")
        
        print(f"✅ Detected plates saved to: {txt_path}")
        return txt_filename
        
    except Exception as e:
        print(f"❌ Error saving plates to text file: {e}")
        return None


# ----------------------------
# 🔹 Process Single Plate Pipeline
# ----------------------------

def process_single_plate_pipeline(plate_crop, detection_idx, timestamp, app_config, enhancer, ph_ocr, svm_type_classifier, svm_violation_detector):
    """Process single plate crop through 4-stage pipeline"""
    try:
        if plate_crop is None or plate_crop.size == 0:
            return None
        
        base_name = f'plate_{detection_idx}_{timestamp}'
        
        # Save original crop
        original_filename = f'{base_name}_original.jpg'
        original_path = os.path.join(app_config['SVM_FOLDER'], original_filename)
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        cv2.imwrite(original_path, plate_crop)
        print(f"    📸 Original saved: {original_filename}")
        
        # STAGE 2: Enhancement
        print(f"    🎨 Stage 2: Enhancement...")
        plate_copy = plate_crop.copy()
        enhanced_plate = enhancer.enhance_image(plate_copy)
        
        if enhanced_plate is None:
            enhanced_plate = plate_crop
        
        # Save enhanced image
        enhanced_filename = f'{base_name}_enhanced.jpg'
        enhanced_path = os.path.join(app_config['ENHANCEMENT_FOLDER'], enhanced_filename)
        os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
        cv2.imwrite(enhanced_path, enhanced_plate)
        print(f"    📸 Enhanced saved: {enhanced_filename}")
        
        # Create comparison image
        comparison_filename = f'{base_name}_comparison.jpg'
        comparison_image = enhancer.create_comparison_image(plate_crop, enhanced_plate)
        comparison_path = os.path.join(app_config['ENHANCEMENT_FOLDER'], comparison_filename)
        cv2.imwrite(comparison_path, comparison_image)
        print(f"    📸 Comparison saved: {comparison_filename}")
        
        # STAGE 3: OCR
        print(f"    🔤 Stage 3: OCR...")
        ocr_result = ph_ocr.recognize_philippine_plate(enhanced_plate)
        
        plate_info = {
            'detection_idx': detection_idx,
            'enhancement_comparison': comparison_filename,
            'original_image': original_filename,
            'enhanced_image': enhanced_filename,
            'original_image_url': f'svm/{original_filename}',
            'enhanced_image_url': f'enhancements/{enhanced_filename}',
            'comparison_image_url': f'enhancements/{comparison_filename}'
        }
        
        if ocr_result and ocr_result.get('text'):
            plate_text = ocr_result['text']
            is_valid_plate = ph_ocr.is_valid_philippine_plate(plate_text)
            
            plate_info.update({
                'text': plate_text,
                'ocr_confidence': ocr_result['confidence'],
                'ocr_method': ocr_result['method'],
                'raw_text': ocr_result.get('raw_text', ''),
                'is_valid_plate': is_valid_plate
            })
            
            print(f"    ✅ OCR Success: '{plate_text}' (conf: {ocr_result['confidence']:.1%})")
            
            # STAGE 4: SVM Classification
            if is_valid_plate:
                print(f"    🤖 Stage 4: SVM Classification...")
                
                type_result = svm_type_classifier.classify(enhanced_plate)
                plate_info['svm_type_classification'] = type_result['classification']
                plate_info['svm_type_confidence'] = type_result['confidence']
                plate_info['svm_is_valid'] = type_result['is_valid']
                
                print(f"      Type: {type_result['classification']} (conf: {type_result['confidence']:.1%})")
                
                violation_result = svm_violation_detector.detect_violation(plate_text)
                plate_info['violation_detection'] = violation_result
                
                if violation_result['has_violation']:
                    print(f"    ⚠️ VIOLATION DETECTED: {violation_result['violation_type']}")
                    violation_filename = f'{base_name}_violation.jpg'
                    violation_path = os.path.join(app_config['VIOLATION_FOLDER'], violation_filename)
                    cv2.imwrite(violation_path, plate_crop)
                    plate_info['violation_image'] = violation_filename
                    plate_info['violation_image_url'] = f'violations/{violation_filename}'
            else:
                plate_info['invalid_plate'] = True
        else:
            plate_info['ocr_failed'] = True
        
        return plate_info
        
    except Exception as e:
        print(f"    ❌ Pipeline error: {e}")
        return None


# ----------------------------
# 🔹 Process Image with YOLO
# ----------------------------

def process_image_with_yolo(image, filename, app_config, yolo_model, enhancer, ph_ocr, svm_type_classifier, svm_violation_detector):
    """Process single image with YOLO detection"""
    try:
        if image is None or image.size == 0:
            return {'error': 'Invalid image'}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        
        print(f"\n  📍 Processing Image: {filename}")
        
        # Save original image
        original_filename = f'{base_name}_{timestamp}_original.jpg'
        original_path = os.path.join(app_config['SVM_FOLDER'], original_filename)
        cv2.imwrite(original_path, image)
        
        annotated_image = image.copy()
        detected_plates = []
        
        # STAGE 1: YOLO Detection
        print(f"  🎯 Stage 1: YOLO Detection...")
        
        if yolo_model is None:
            return {'error': 'YOLO model not available'}
        
        try:
            results = yolo_model.predict(
                image, 
                imgsz=1280,
                conf=0.15,
                iou=0.4,
                augment=True,
                verbose=False
            )
        except Exception as e:
            print(f"  ❌ YOLO prediction error: {e}")
            return {'error': f'YOLO prediction failed: {str(e)}'}
        
        if len(results[0].boxes) == 0:
            print(f"  ⚠️ No plates detected")
            return {'error': 'No license plates detected'}
        
        print(f"  ✅ YOLO detected {len(results[0].boxes)} plates")
        
        for detection_idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf = float(box.conf[0])
            
            print(f"    Detection {detection_idx+1}: (YOLO conf={yolo_conf:.2%})")

            padding = 0.2
            pad_x = int((x2 - x1) * padding)
            pad_y = int((y2 - y1) * padding)
            x1_crop = max(0, x1 - pad_x)
            y1_crop = max(0, y1 - pad_y)
            x2_crop = min(image.shape[1], x2 + pad_x)
            y2_crop = min(image.shape[0], y2 + pad_y)
            
            plate_crop = image[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if plate_crop.size == 0 or plate_crop.shape[0] < 20 or plate_crop.shape[1] < 20:
                continue

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_image, f"YOLO: {yolo_conf:.2f}", 
                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            pipeline_result = process_single_plate_pipeline(
                plate_crop, detection_idx, timestamp, app_config, 
                enhancer, ph_ocr, svm_type_classifier, svm_violation_detector
            )
            
            if pipeline_result:
                pipeline_result.update({
                    'yolo_confidence': yolo_conf,
                    'yolo_bbox': [x1, y1, x2, y2],
                })
                detected_plates.append(pipeline_result)
                
                if 'text' in pipeline_result:
                    text = pipeline_result['text']
                    text_y = max(y1 - 60, 20)
                    cv2.putText(annotated_image, f"PLATE: {text}", 
                              (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Save annotated image
        annotated_filename = f'{base_name}_{timestamp}_annotated.jpg'
        annotated_path = os.path.join(app_config['OUTPUT_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)
        
        result_data = {
            'success': True,
            'filename': filename,
            'annotated_image': annotated_filename,
            'total_plates_detected': len(detected_plates),
            'detected_plates': detected_plates,
            'original_image': original_filename,
            'annotated_image_url': f'results/{annotated_filename}',
            'original_image_url': f'svm/{original_filename}'
        }
        
        if detected_plates:
            result_data['stage3_ocr_success'] = sum(1 for plate in detected_plates if 'text' in plate)
            result_data['stage4_violations_detected'] = sum(1 for plate in detected_plates 
                                                          if plate.get('violation_detection', {}).get('has_violation', False))
        
        return result_data
        
    except Exception as e:
        print(f"  ❌ Image processing error: {e}")
        return {'error': str(e)}