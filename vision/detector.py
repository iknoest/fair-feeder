import cv2
import numpy as np

class CatDetector:
    def __init__(self, model_path='efficientdet_lite2.tflite', min_detection_confidence=0.4):
        """
        Initialize TensorFlow Lite Object Detection.
        Tries ai-edge-litert first (Google's modern TFLite runtime for edge devices).
        Falls back to standard TensorFlow or Haar Cascade if needed.
        """
        self.min_detection_confidence = min_detection_confidence
        self.use_tflite = False
        self.use_haarcascade = False
        self.interpreter_type = None  # Track which interpreter we're using
        self.model_path = model_path
        
        # Try ai-edge-litert first (preferred for Raspberry Pi)
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.interpreter = Interpreter(model_path)
            self.interpreter.allocate_tensors()  # IMPORTANT: Must call this for ai-edge-litert!
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.interpreter_type = 'ai-edge-litert'
            self.use_tflite = True
            print("✅ Using ai-edge-litert (Google's TensorFlow Lite runtime)")
            return
        except ImportError:
            print("⚠️  ai-edge-litert not found, trying tflite-runtime...")
        except Exception as e:
            print(f"⚠️  ai-edge-litert failed ({e}), trying alternatives...")
        
        # Try tflite_runtime or tensorflow as fallback
        try:
            try:
                import tflite_runtime.interpreter as tflite
                Interpreter = tflite.Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.interpreter_type = 'tflite_runtime'
            self.use_tflite = True
            print("✅ Using tflite-runtime (TensorFlow Lite)")
            return
        except Exception as e:
            print(f"⚠️  TFLite not available ({e}), falling back to Haar Cascade")
        
        # Fallback to Haar Cascade (lightweight, built into OpenCV)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcat_extended.xml'
        try:
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self.use_haarcascade = True
            print("✅ Using Haar Cascade cat detector")
        except Exception as e2:
            print(f"⚠️  Haar Cascade also failed ({e2})")
            self.cascade = None

    def detect(self, image, filter_cats=True):
        """
        Detects objects in the image.
        Returns a list of dicts with bounding boxes and confidence scores.
        """
        if self.use_tflite:
            return self._detect_tflite(image, filter_cats)
        elif self.use_haarcascade:
            return self._detect_haarcascade(image)
        else:
            return []

    def _detect_tflite(self, image, filter_cats=True):
        """Detect using TensorFlow Lite."""
        try:
            if self.interpreter_type == 'ai-edge-litert':
                return self._detect_ai_edge_litert(image, filter_cats)
            else:
                return self._detect_tflite_runtime(image, filter_cats)
        except Exception as e:
            print(f"⚠️  TFLite inference error: {e}")
            return []

    def _detect_ai_edge_litert(self, image, filter_cats=True):
        """Detect using Google's ai-edge-litert (newer, faster)."""
        try:
            # Get input details
            input_shape = self.input_details[0]['shape']
            h_in, w_in = input_shape[1], input_shape[2]
            
            # Prepare input
            resized = cv2.resize(image, (w_in, h_in))
            input_data = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            
            # Set input and invoke (no arguments to invoke()!)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()  # No arguments!
            
            # Get outputs
            detections = self._parse_ai_edge_litert_outputs()
            
            # Filter by confidence and class
            filtered = []
            for det in detections:
                if det['score'] < self.min_detection_confidence:
                    continue
                if filter_cats and det['class'] != 'cat':
                    continue
                filtered.append(det)
            
            return filtered
        except Exception as e:
            print(f"⚠️  ai-edge-litert error: {e}")
            return []

    def _detect_tflite_runtime(self, image, filter_cats=True):
        """Detect using tflite_runtime (older, but wider compatibility)."""
        try:
            input_shape = self.input_details[0]['shape']
            h_in, w_in = input_shape[1], input_shape[2]
            
            # Prepare input
            resized = cv2.resize(image, (w_in, h_in))
            input_data = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            
            # Set input and run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get outputs
            detections = self._parse_tflite_runtime_outputs()
            
            # Filter by confidence and class
            filtered = []
            for det in detections:
                if det['score'] < self.min_detection_confidence:
                    continue
                if filter_cats and det['class'] != 'cat':
                    continue
                filtered.append(det)
            
            return filtered
        except Exception as e:
            print(f"⚠️  tflite_runtime error: {e}")
            return []

    def _detect_haarcascade(self, image):
        """
        Detect using OpenCV Haar Cascade (lightweight, no external dependencies).
        Returns simple bounding boxes without confidence scores.
        """
        if not hasattr(self, 'cascade') or self.cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cats = self.cascade.detectMultiScale(gray, 1.1, 4)
            
            detections = []
            for (x, y, w, h) in cats:
                h_img, w_img = image.shape[:2]
                detections.append({
                    'bbox': (x / w_img, y / h_img, w / w_img, h / h_img),  # Normalized
                    'score': 0.95,  # Haar doesn't give confidence, assume high
                    'class': 'cat',
                })
            
            return detections
        except Exception:
            return []

    def _parse_ai_edge_litert_outputs(self):
        """
        Parse ai-edge-litert outputs using get_tensor.
        Same method as tflite_runtime after invoke() is called.
        """
        detections = []
        
        try:
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            classes_ = self.interpreter.get_tensor(self.output_details[1]['index'])
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            num_dets = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
            
            # Class mapping for COCO dataset
            class_names = {16: 'cat', 17: 'dog'}
            
            for i in range(min(num_dets, len(scores[0]))):
                score = float(scores[0][i])
                class_id = int(classes_[0][i])
                class_name = class_names.get(class_id, f'class_{class_id}')
                
                # Normalize box coordinates
                y1, x1, y2, x2 = boxes[0][i]
                width = float(x2 - x1)
                height = float(y2 - y1)
                
                detections.append({
                    'bbox': (float(x1), float(y1), width, height),
                    'score': score,
                    'class': class_name,
                })
        except Exception as e:
            print(f"⚠️  Error parsing ai-edge-litert outputs: {e}")
        
        return detections

    def _parse_tflite_runtime_outputs(self):
        """Parse tflite_runtime outputs using get_tensor."""
        detections = []
        
        try:
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            num_dets = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        except (IndexError, KeyError):
            return []
        
        # Class names for EfficientDet (COCO dataset)
        class_names = {
            16: 'cat',    # COCO class 16 = cat
            17: 'dog',    # COCO class 17 = dog
        }
        
        # Parse detections
        for i in range(min(num_dets, len(scores[0]))):
            score = float(scores[0][i])
            if score < self.min_detection_confidence:
                continue
            
            class_id = int(classes[0][i])
            class_name = class_names.get(class_id, f'class_{class_id}')
            
            # Bounding box (normalized coordinates)
            y1, x1, y2, x2 = boxes[0][i]
            origin_x = float(x1)
            origin_y = float(y1)
            width = float(x2 - x1)
            height = float(y2 - y1)
            
            detections.append({
                'bbox': (origin_x, origin_y, width, height),
                'score': score,
                'class': class_name,
            })
        
        return detections
