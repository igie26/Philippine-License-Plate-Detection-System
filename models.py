"""
Model Classes for Philippine License Plate Detection System
===========================================================
Contains:
- RRDBNet Architecture for Real-ESRGAN
- SVM Plate Type Classifier
- SVM Coding Violation Detector
- Real-ESRGAN Enhancer
- Philippine Plate OCR
"""

import os
import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
from datetime import datetime

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

# ----------------------------
# 🔹 RRDBNet Architecture for Real-ESRGAN
# ----------------------------

if TORCH_AVAILABLE:
    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super(ResidualDenseBlock, self).__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super(RRDB, self).__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class SimpleRRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
            super(SimpleRRDBNet, self).__init__()
            self.num_feat = num_feat
            self.num_block = num_block
            
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            
            self.body = nn.ModuleList()
            for i in range(num_block):
                self.body.append(RRDB(num_feat, num_grow_ch))
            
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            fea = self.conv_first(x)
            
            for i in range(self.num_block):
                fea = self.body[i](fea)
            
            trunk = self.conv_body(fea)
            fea = fea + trunk
            
            fea = self.lrelu(self.conv_up1(fea))
            fea = self.lrelu(self.conv_up2(fea))
            out = self.conv_last(self.lrelu(self.conv_hr(fea)))
            
            return out
else:
    class SimpleRRDBNet:
        def __init__(self, *args, **kwargs):
            pass


# ----------------------------
# 🔹 SVM Plate Type Classifier
# ----------------------------

class SVMPlateTypeClassifier:
    """SVM for Plate Type Classification"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load trained SVM model and scaler"""
        try:
            model_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\svm model\svm_model_balanced.pkl",
                r"C:\Users\DELL\Videos\systematic\thesis-project\SVM\svm_model_balanced.pkl",
            ]
            
            scaler_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\svm model\scaler.pkl",
                r"C:\Users\DELL\Videos\systematic\thesis-project\SVM\scaler.pkl",
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"✅ SVM Plate Type Classifier loaded")
                    break
            else:
                print("❌ Could not find SVM model")
                self.model = None
            
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"✅ Scaler loaded")
                    break
            else:
                print("❌ Scaler not found")
            
        except Exception as e:
            print(f"❌ Error loading SVM model: {e}")
            self.model = None
    
    def extract_features(self, image):
        """Extract HOG features from plate image"""
        try:
            if image is None or image.size == 0:
                return None
            
            resized = cv2.resize(image, (100, 50))
            
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            hog = cv2.HOGDescriptor((100, 50), (20, 20), (10, 10), (10, 10), 9)
            features = hog.compute(gray)
            
            return features.flatten() if features is not None else None
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def classify(self, image):
        """Classify plate type using SVM"""
        try:
            if image is None or image.size == 0:
                return {'classification': 'unknown', 'is_valid': False, 'confidence': 0.0}
            
            if self.model is not None:
                features = self.extract_features(image)
                if features is not None and self.scaler is not None:
                    try:
                        features_scaled = self.scaler.transform([features])
                        prediction = self.model.predict(features_scaled)[0]
                        probability = np.max(self.model.predict_proba(features_scaled))
                        
                        class_mapping = {
                            0: 'plate_government',
                            1: 'plate_motorcycle', 
                            2: 'plate_old',
                            3: 'plate_white',
                            4: 'plate_yellow'
                        }
                        
                        classification = class_mapping.get(prediction, 'plate_white')
                        is_valid = classification != 'plate_motorcycle'
                        
                        return {
                            'classification': classification,
                            'is_valid': is_valid,
                            'confidence': float(probability)
                        }
                    except Exception as e:
                        print(f"SVM prediction error: {e}")
            
            # FALLBACK: Use basic heuristics
            height, width = image.shape[:2]
            aspect_ratio = width / height if height > 0 else 0
            
            if aspect_ratio > 3.0:
                return {'classification': 'plate_motorcycle', 'is_valid': False, 'confidence': 0.7}
            elif aspect_ratio > 2.0:
                return {'classification': 'plate_white', 'is_valid': True, 'confidence': 0.6}
            else:
                return {'classification': 'plate_white', 'is_valid': True, 'confidence': 0.5}
            
        except Exception as e:
            print(f"SVM classification error: {e}")
            return {'classification': 'plate_white', 'is_valid': True, 'confidence': 0.5}


# ----------------------------
# 🔹 SVM Coding Violation Detector
# ----------------------------

class SVMCodingViolationDetector:
    """SVM for Coding Violation Detection"""
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load coding violation detection model"""
        try:
            model_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\SVM\svm_model_balanced.pkl",
                r"C:\Users\DELL\Videos\systematic\thesis-project\svm model\svm_model_balanced.pkl",
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"✅ SVM Coding Violation Detector loaded")
                    break
            else:
                print("❌ Coding violation model not found")
                self.model = None
            
        except Exception as e:
            print(f"❌ Error loading coding violation model: {e}")
            self.model = None
    
    def detect_violation(self, plate_text, timestamp=None):
        """Detect coding violation using plate text"""
        try:
            if not plate_text:
                return {
                    'has_violation': False,
                    'violation_type': 'none',
                    'confidence': 0.0,
                    'reason': 'No plate text'
                }
            
            last_char = plate_text[-1]
            if last_char.isdigit():
                last_digit = int(last_char)
            else:
                last_digit = 0
            
            current_time = datetime.now()
            hour = current_time.hour
            day_of_week = current_time.weekday()
            is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
            
            has_violation = False
            violation_type = 'none'
            confidence = 0.8
            
            if is_rush_hour:
                if last_digit in [1, 2] and day_of_week == 0:
                    has_violation = True
                    violation_type = 'rush_hour_monday'
                elif last_digit in [3, 4] and day_of_week == 1:
                    has_violation = True
                    violation_type = 'rush_hour_tuesday'
                elif last_digit in [5, 6] and day_of_week == 2:
                    has_violation = True
                    violation_type = 'rush_hour_wednesday'
                elif last_digit in [7, 8] and day_of_week == 3:
                    has_violation = True
                    violation_type = 'rush_hour_thursday'
                elif last_digit in [9, 0] and day_of_week == 4:
                    has_violation = True
                    violation_type = 'rush_hour_friday'
            
            return {
                'has_violation': has_violation,
                'violation_type': violation_type,
                'confidence': confidence,
                'last_digit': last_digit,
                'hour': hour,
                'is_rush_hour': bool(is_rush_hour),
                'day_of_week': day_of_week
            }
                
        except Exception as e:
            print(f"Coding violation detection error: {e}")
            return {
                'has_violation': False,
                'violation_type': 'none',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }


# ----------------------------
# 🔹 Real-ESRGAN Enhancer
# ----------------------------

class RealESRGANEnhancer:
    """Real-ESRGAN enhancer for license plate images"""
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.load_model()
        self.enhancement_available = False
    
    def load_model(self):
        """Load Real-ESRGAN model"""
        try:
            if not TORCH_AVAILABLE:
                print("❌ PyTorch not available - Real-ESRGAN disabled")
                self.enhancement_available = False
                return
                
            model_paths = [
                r"C:\Users\DELL\Videos\systematic\thesis-project\real-cnn-model\RealESRGAN_x4plus.pth",
                r"C:\Users\DELL\Videos\systematic\thesis-project\real-cnn-model\net_g_100000.pth",
            ]
            
            found_model = None
            for mp in model_paths:
                if os.path.exists(mp):
                    found_model = mp
                    print(f"✅ Found Real-ESRGAN model")
                    break
            
            if found_model:
                print(f"🔄 Loading Real-ESRGAN model...")
                
                try:
                    try:
                        from basicsr.archs.rrdbnet_arch import RRDBNet
                        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                    except ImportError:
                        print("🔄 Using custom RRDBNet implementation")
                        self.model = SimpleRRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
                    
                    checkpoint = torch.load(found_model, map_location=self.device)
                    
                    if 'params_ema' in checkpoint:
                        state_dict = checkpoint['params_ema']
                    elif 'params' in checkpoint:
                        state_dict = checkpoint['params']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.eval()
                    self.model = self.model.to(self.device)
                    
                    self.enhancement_available = True
                    print(f"✅ Real-ESRGAN Model loaded successfully")
                    
                except Exception as e:
                    print(f"❌ Error loading Real-ESRGAN model: {e}")
                    self.model = None
                    self.enhancement_available = False
                
            else:
                print(f"❌ Real-ESRGAN model not found")
                self.model = None
                self.enhancement_available = False
                
        except Exception as e:
            print(f"❌ Error loading Real-ESRGAN: {e}")
            self.model = None
            self.enhancement_available = False
    
    def enhance_image(self, image):
        """Enhance image"""
        try:
            enhanced = self.adaptive_enhancement(image)
            
            if self.enhancement_available and self.model is not None:
                try:
                    esrgan_enhanced = self.real_esrgan_enhance(enhanced)
                    if esrgan_enhanced is not None:
                        print("  🎨 Real-ESRGAN Enhancement Applied")
                        return esrgan_enhanced
                except Exception as e:
                    print(f"  ⚠️ Real-ESRGAN error: {e}")
                    return enhanced
            else:
                print("  🎨 Adaptive Enhancement Applied")
                return enhanced
            
        except Exception as e:
            print(f"Enhancement error: {e}")
            return image
    
    def real_esrgan_enhance(self, image):
        """Real-ESRGAN enhancement"""
        try:
            if image is None or image.size == 0:
                return None
            
            if len(image.shape) == 2:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
            
            h, w = img_rgb.shape[:2]
            max_size = 512
            
            if h > max_size or w > max_size:
                scale = min(max_size/h, max_size/w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
            
            output = output.squeeze().permute(1, 2, 0).cpu().numpy()
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            return output_bgr
            
        except Exception as e:
            print(f"Real-ESRGAN processing error: {e}")
            return None
    
    def adaptive_enhancement(self, image):
        """Adaptive enhancement"""
        try:
            if image is None or image.size == 0:
                return image
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
            elif len(image.shape) == 2:
                l = image.copy()
            else:
                return image
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l)
            
            denoised = cv2.fastNlMeansDenoising(enhanced_l, None, 10, 7, 21)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                enhanced_lab = cv2.merge([sharpened, a, b])
                result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:
                result = sharpened
            
            return result
            
        except Exception as e:
            print(f"Adaptive enhancement error: {e}")
            return image

    def create_comparison_image(self, original, enhanced):
        """Create side-by-side comparison image"""
        try:
            if original is None or enhanced is None:
                return original
            
            if len(original.shape) == 2:
                original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            else:
                original_bgr = original.copy()
            
            if len(enhanced.shape) == 2:
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_bgr = enhanced.copy()
            
            h1, w1 = original_bgr.shape[:2]
            h2, w2 = enhanced_bgr.shape[:2]
            target_height = max(h1, h2, 100)
            
            scale1 = target_height / h1
            scale2 = target_height / h2
            new_w1 = int(w1 * scale1)
            new_w2 = int(w2 * scale2)
            
            resized_orig = cv2.resize(original_bgr, (new_w1, target_height))
            resized_enh = cv2.resize(enhanced_bgr, (new_w2, target_height))
            
         