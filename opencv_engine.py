# opencv_engine.py
from os import path, makedirs
import cv2
import numpy as np
from thumbor.engines import BaseEngine
from thumbor.utils import logger

class Engine(BaseEngine):
    def __init__(self, context):
        super(Engine, self).__init__(context)
        self.context = context
        self.image = None
        self.image_data = None
        self.extension = None
        #self.face_detector = None
        # Face detection settings
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # VVIP face recognition
        self.face_recognizer = self._init_face_recognizer()
        self.vvip_faces = []  # Will store locations of VVIP faces

    def _init_face_recognizer(self):
        # Check if we have OpenCV with DNN face recognition support
        if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Load the recognizer model if it exists
            model_path = path.join(path.dirname(path.abspath(__file__)),'vvip_faces', 'vvip_faces_model.yml')
            if path.exists(model_path):
                try:
                    recognizer.read(model_path)
                    print(f'✓ VVIP face recognition model loaded successfully')
                    logger.info("VVIP face recognition model loaded successfully")
                    return recognizer
                except Exception as e:
                    logger.error(f"Error loading face recognition model: {str(e)}")
        
        logger.warning("VVIP face recognition not available")
        return None

    def load(self, buffer, extension):
        self.extension = extension
        self.image_data = buffer
        img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            logger.error("OpenCV cannot decode this image.")
            return False
            
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        self.image = img

        # Save debug image before face detection
        self.save_debug_image("before")

        # Run face detection when image is loaded
        self._detect_vvip_faces()

        # Save debug image after face detection
        self.save_debug_image("after")

        return True

    def _detect_vvip_faces(self):
        """Detect and recognize VVIP faces in the image"""
        self.vvip_faces = []
        
        # Convert to grayscale for face detection
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.image
            
        # Detect frontal faces
        frontal_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # # Detect profile faces (side view)
        # profile_faces = self.profile_cascade.detectMultiScale(
        #     gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        # )
        
        # # Flip image and detect profiles from other side
        # flipped = cv2.flip(gray, 1)
        # profile_faces2 = self.profile_cascade.detectMultiScale(
        #     flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        # )
        
        if len(frontal_faces) == 0:
            print("No faces detected")
            return
        
        print(f"Detected {len(frontal_faces)} faces")

        # # Adjust coordinates for flipped faces
        # h, w = gray.shape[:2]
        # for (x, y, w_face, h_face) in profile_faces2:
        #     profile_faces = np.vstack([profile_faces, np.array([w - x - w_face, y, w_face, h_face])]) if len(profile_faces) else np.array([[w - x - w_face, y, w_face, h_face]])
        profile_faces=[]
        all_faces = np.vstack([frontal_faces, profile_faces]) if len(frontal_faces) and len(profile_faces) else (frontal_faces if len(frontal_faces) else profile_faces)
        
        # if len(all_faces) == 0:
        #     return
            
        # Process all detected faces
        for (x, y, w, h) in all_faces:
            # If we have a face recognizer, try to identify the face
            is_vvip = False
            
            if self.face_recognizer:
                face_roi = gray[y:y+h, x:x+w]
                try:
                    # Resize to a standard size
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Predict who this face belongs to
                    label, confidence = self.face_recognizer.predict(face_roi)
                    print(f'label- {label}, confidence- {confidence}')
                    # Lower confidence is better in LBPH
                    if confidence <80: #< 70:  # Threshold for VVIP recognition
                        is_vvip = True
                        logger.info(f"VVIP detected with confidence: {confidence}")
                        print(f"VVIP detected with confidence: {confidence}")
                except Exception as e:
                    print(f"Error during face recognition: {str(e)}")
                    logger.error(f"Error during face recognition: {str(e)}")
            
            # For development/testing, consider all faces as VVIPs            
            #is_vvip = True # Remove this in production when you have a trained model
            
            if is_vvip:
                # Add a margin around the face for better cropping
                margin = 0.5  # 50% extra margin
                x_ext = int(x - (w * margin/2))
                y_ext = int(y - (h * margin/2))
                w_ext = int(w * (1 + margin))
                h_ext = int(h * (1 + margin))
                
                # Ensure coordinates are within image bounds
                x_ext = max(0, x_ext)
                y_ext = max(0, y_ext)
                w_ext = min(w_ext, self.image.shape[1] - x_ext)
                h_ext = min(h_ext, self.image.shape[0] - y_ext)
                
                self.vvip_faces.append((x_ext, y_ext, w_ext, h_ext))
        print(f'is_vvip - {is_vvip}')

    def draw_rectangle(self, x, y, width, height):
        cv2.rectangle(self.image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 0, 0), 2)

    # def crop(self, left, top, right, bottom):
    #     self.image = self.image[int(top):int(bottom), int(left):int(right)]
    def crop(self, left, top, right, bottom):
        """Smart crop that ensures VVIP faces are included in the frame"""
        # If no VVIP faces are detected, proceed with normal crop
        if not self.vvip_faces:
            self.image = self.image[int(top):int(bottom), int(left):int(right)]
            return
            
        # Calculate dimensions of the requested crop
        crop_width = right - left
        crop_height = bottom - top
        img_height, img_width = self.image.shape[:2]
        
        # Adjust crop to include all VVIP faces
        new_left, new_top = left, top
        new_right, new_bottom = right, bottom
        
        for (x, y, w, h) in self.vvip_faces:
            face_center_x = x + w/2
            face_center_y = y + h/2
            
            # Check if this face would be cut out
            if (face_center_x < left or face_center_x > right or 
                face_center_y < top or face_center_y > bottom):
                
                # Adjust crop to include this face
                # Try to keep the original crop dimensions if possible
                if face_center_x < left:
                    shift = left - face_center_x + w/2
                    new_left = max(0, left - shift)
                    new_right = min(img_width, new_left + crop_width)
                
                elif face_center_x > right:
                    shift = face_center_x - right + w/2
                    new_right = min(img_width, right + shift)
                    new_left = max(0, new_right - crop_width)
                
                if face_center_y < top:
                    shift = top - face_center_y + h/2
                    new_top = max(0, top - shift)
                    new_bottom = min(img_height, new_top + crop_height)
                
                elif face_center_y > bottom:
                    shift = face_center_y - bottom + h/2
                    new_bottom = min(img_height, bottom + shift)
                    new_top = max(0, new_bottom - crop_height)
        
        # Apply the adjusted crop
        self.image = self.image[int(new_top):int(new_bottom), int(new_left):int(new_right)]
        
        # After cropping, update VVIP face coordinates
        self._adjust_face_coords_after_crop(new_left, new_top)


    def _adjust_face_coords_after_crop(self, crop_left, crop_top):
        """Update face coordinates after cropping"""
        adjusted_faces = []
        for (x, y, w, h) in self.vvip_faces:
            # Adjust coordinates relative to new cropped image
            new_x = x - crop_left
            new_y = y - crop_top
            
            # Check if face is still in the frame
            if (new_x + w > 0 and new_x < self.image.shape[1] and 
                new_y + h > 0 and new_y < self.image.shape[0]):
                
                # Clip coordinates to image boundaries
                new_x = max(0, new_x)
                new_y = max(0, new_y)
                new_w = min(w, self.image.shape[1] - new_x)
                new_h = min(h, self.image.shape[0] - new_y)
                
                adjusted_faces.append((new_x, new_y, new_w, new_h))
        
        self.vvip_faces = adjusted_faces


    # def resize(self, width, height):
    #     self.image = cv2.resize(self.image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    def resize(self, width, height):
        # Store the original dimensions for scaling face coordinates
        orig_h, orig_w = self.image.shape[:2]
        
        # Perform the resize
        self.image = cv2.resize(self.image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        
        # Scale the face coordinates
        if self.vvip_faces:
            scale_x = width / orig_w
            scale_y = height / orig_h
            
            scaled_faces = []
            for (x, y, w, h) in self.vvip_faces:
                scaled_faces.append((
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                ))
            
            self.vvip_faces = scaled_faces

    # def flip_horizontally(self):
    #     self.image = cv2.flip(self.image, 1)
    def flip_horizontally(self):
        self.image = cv2.flip(self.image, 1)
        
        # Adjust face coordinates for horizontal flip
        if self.vvip_faces:
            width = self.image.shape[1]
            flipped_faces = []
            
            for (x, y, w, h) in self.vvip_faces:
                new_x = width - (x + w)
                flipped_faces.append((new_x, y, w, h))
            
            self.vvip_faces = flipped_faces

    # def flip_vertically(self):
    #     self.image = cv2.flip(self.image, 0)
    def flip_vertically(self):
        self.image = cv2.flip(self.image, 0)
        
        # Adjust face coordinates for vertical flip
        if self.vvip_faces:
            height = self.image.shape[0]
            flipped_faces = []
            
            for (x, y, w, h) in self.vvip_faces:
                new_y = height - (y + h)
                flipped_faces.append((x, new_y, w, h))
            
            self.vvip_faces = flipped_faces


    def read(self, extension=None, quality=None):
        # Draw rectangles around VVIP faces for debugging
        # Comment this out in production if you don't want visible rectangles
        for (x, y, w, h) in self.vvip_faces:
            cv2.rectangle(self.image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        if quality is None:
            quality = self.context.config.QUALITY
            
        ext = extension or self.extension
        
        if ext.lower() == '.jpg' or ext.lower() == '.jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext.lower() == '.webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        elif ext.lower() == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, quality // 10]
        else:
            params = []
            
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            # Convert RGB to BGR for saving
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        else:
            img = self.image
            
        success, img_buffer = cv2.imencode(ext, img, params)
        
        if not success:
            logger.error("OpenCV cannot encode this image with extension %s", ext)
            return None
            
        return img_buffer.tobytes()

    def get_image_data(self):
        return self.read()

    def set_image_data(self, data):
        self.load(data, self.extension)

    def get_image_mode(self):
        if len(self.image.shape) == 3 and self.image.shape[2] == 4:
            return "RGBA"
        elif len(self.image.shape) == 3 and self.image.shape[2] == 3:
            return "RGB"
        else:
            return "L"

    @property
    def size(self):
        height, width = self.image.shape[:2]
        return (width, height)

    # this method is to visualize the VVIP faces (for debugging)
    def debug_vvip_faces(self):
        """Creates a debug copy of the image with VVIP faces highlighted"""
        debug_img = self.image.copy()
        for (x, y, w, h) in self.vvip_faces:
            cv2.rectangle(debug_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
        return debug_img

    def save_debug_image(self, tag="debug"):
        """Save a debug image showing all faces and marking VVIPs"""
        debug_img = self.image.copy()
        
        # Convert to BGR for OpenCV saving
        if len(debug_img.shape) == 3 and debug_img.shape[2] == 3:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        
        # Get all faces for comparison
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.image
        
        frontal_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Draw all detected faces in RED
        for (x, y, w, h) in frontal_faces:
            cv2.rectangle(debug_img, (int(x), int(y)), 
                        (int(x + w), int(y + h)), (0, 0, 255), 2)
            
        # Draw VVIP faces in GREEN
        for (x, y, w, h) in self.vvip_faces:
            cv2.rectangle(debug_img, (int(x), int(y)), 
                        (int(x + w), int(y + h)), (0, 255, 0), 3)
            
        # Save the debug image
        debug_dir = '/tmp/thumbor_debug'
        if not path.exists(debug_dir):
            makedirs(debug_dir)
        
        import time
        timestamp = int(time.time())
        debug_path = path.join(debug_dir, f'face_{tag}_{timestamp}.jpg')
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved to {debug_path}")
        return debug_path