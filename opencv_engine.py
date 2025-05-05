# opencv_engine.py
from os import path, makedirs
import cv2
import numpy as np
from thumbor.engines import BaseEngine
from thumbor.utils import logger

# Import our modern face recognizer
#from modern_face_recognition import VVIPFaceRecognizer
from vvip_face_recognizer_insight import VVIPFaceRecognizerInsight as VVIPFaceRecognizer


class Engine(BaseEngine):
    def __init__(self, context):
        super(Engine, self).__init__(context)
        self.context = context
        self.image = None
        self.image_data = None
        self.extension = None
        self.original_mode = None
        self.grayscale_image = None  # For storing grayscale version of the image
        
        # Initialize the modern face recognizer
        self.face_recognizer = VVIPFaceRecognizer(
            vvip_faces_dir=path.join(path.dirname(path.abspath(__file__)), 'vvip_faces'),
            models_dir=path.join(path.dirname(path.abspath(__file__)), 'models'),
            tolerance=0.4  # Adjust this based on testing (lower = stricter)
        )
        
        self.vvip_faces = []  # Will store locations of VVIP faces

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
            self.original_mode = "RGB"
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # Convert BGRA to RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            self.original_mode = "RGBA"
        else:
            # Grayscale
            self.original_mode = "L"
            
        self.image = img

        # # Save debug image before face detection
        # self.save_debug_image("before")

        # Run face detection when image is loaded
        self._detect_vvip_faces()

        # Save debug image after face detection
        self.save_debug_image("after")

        return True

    def _detect_vvip_faces(self):
        """Detect and recognize VVIP faces in the image"""
        self.vvip_faces = []
        
        # Use the modern face recognizer
        detected_faces = self.face_recognizer.recognize_faces(self.image)
        
        if not detected_faces:
            logger.info("No VVIP faces detected")
            return
            
        # Store the face locations (and names, if needed)
        for (x, y, w, h, name, confidence) in detected_faces:
            self.vvip_faces.append((x, y, w, h, name, confidence))
            logger.info(f"VVIP detected: {name} at ({x}, {y}, {w}, {h})")
            print(f"✓ VVIP detected: {name} at (x={x}, y={y}, w={w}, h={h})")

    def draw_rectangle(self, x, y, width, height):
        cv2.rectangle(self.image, (int(x), int(y)), (int(x + width), int(y + height)), (255, 0, 0), 2)

    def crop(self, left, top, right, bottom):
        """Smart crop that ensures VVIP faces are included in the frame"""
        # If no VVIP faces are detected, proceed with normal crop
        if not self.vvip_faces:
            self.image = self.image[int(top):int(bottom), int(left):int(right)]
            if self.grayscale_image is not None:
                self.grayscale_image = self.grayscale_image[int(top):int(bottom), int(left):int(right)]
            return
            
        # Calculate dimensions of the requested crop
        crop_width = right - left
        crop_height = bottom - top
        img_height, img_width = self.image.shape[:2]
        
        # Adjust crop to include all VVIP faces
        new_left, new_top = left, top
        new_right, new_bottom = right, bottom
        
        vvip_count = len(self.vvip_faces)
        for (x, y, w, h, _, _) in self.vvip_faces:
            shift_thrishold = 3*w if vvip_count == 1 else w

            face_center_x = x + w/2
            face_center_y = y + h/2
            
            # Check if this face would be cut out
            if (face_center_x < left or face_center_x > right or 
                face_center_y < top or face_center_y > bottom):
                
                # Adjust crop to include this face
                # Try to keep the original crop dimensions if possible
                if face_center_x < left:
                    shift = left - face_center_x + w/2 + shift_thrishold
                    new_left = max(0, left - shift)
                    new_right = min(img_width, new_left + crop_width)
                
                elif face_center_x > right:
                    shift = face_center_x - right + w/2 + shift_thrishold
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
        
        # Also crop the grayscale image if it exists
        if self.grayscale_image is not None:
            self.grayscale_image = self.grayscale_image[int(new_top):int(new_bottom), int(new_left):int(new_right)]
        
        # After cropping, update VVIP face coordinates
        self._adjust_face_coords_after_crop(new_left, new_top)

    def _adjust_face_coords_after_crop(self, crop_left, crop_top):
        """Update face coordinates after cropping"""
        adjusted_faces = []
        for (x, y, w, h, name, confidence) in self.vvip_faces:
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
                
                adjusted_faces.append((new_x, new_y, new_w, new_h, name, confidence))
        
        self.vvip_faces = adjusted_faces

    def resize(self, width, height):
        # Store the original dimensions for scaling face coordinates
        orig_h, orig_w = self.image.shape[:2]
        
        # Perform the resize
        self.image = cv2.resize(self.image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        
        # Also resize grayscale image if it exists
        if self.grayscale_image is not None:
            self.grayscale_image = cv2.resize(self.grayscale_image, (int(width), int(height)), 
                                            interpolation=cv2.INTER_AREA)
        
        # Scale the face coordinates
        if self.vvip_faces:
            scale_x = width / orig_w
            scale_y = height / orig_h
            
            scaled_faces = []
            for (x, y, w, h, name, confidence) in self.vvip_faces:
                scaled_faces.append((
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y), name, confidence
                ))
            
            self.vvip_faces = scaled_faces

    def flip_horizontally(self):
        self.image = cv2.flip(self.image, 1)
        
        # Also flip grayscale if it exists
        if self.grayscale_image is not None:
            self.grayscale_image = cv2.flip(self.grayscale_image, 1)
        
        # Adjust face coordinates for horizontal flip
        if self.vvip_faces:
            width = self.image.shape[1]
            flipped_faces = []
            
            for (x, y, w, h, name, confidence) in self.vvip_faces:
                new_x = width - (x + w)
                flipped_faces.append((new_x, y, w, h, name, confidence))
            
            self.vvip_faces = flipped_faces

    def flip_vertically(self):
        self.image = cv2.flip(self.image, 0)
        
        # Also flip grayscale if it exists
        if self.grayscale_image is not None:
            self.grayscale_image = cv2.flip(self.grayscale_image, 0)
        
        # Adjust face coordinates for vertical flip
        if self.vvip_faces:
            height = self.image.shape[0]
            flipped_faces = []
            
            for (x, y, w, h, name, confidence) in self.vvip_faces:
                new_y = height - (y + h)
                flipped_faces.append((x, new_y, w, h, name, confidence))
            
            self.vvip_faces = flipped_faces

    def convert_to_grayscale(self, update_image=True, alpha=True):
        """Convert the image to grayscale.
        This method is required by Thumbor's feature detector.
        """
        if len(self.image.shape) == 2:
            # Image is already grayscale
            if update_image:
                pass  # Already grayscale
            else:
                self.grayscale_image = self.image.copy()
        else:
            # Color image - convert to grayscale
            if update_image:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                if len(self.image.shape) == 3 and self.image.shape[2] == 4:
                    # RGBA image
                    if alpha:
                        # Keep alpha channel
                        rgb = cv2.cvtColor(self.image[:,:,:3], cv2.COLOR_RGB2GRAY)
                        self.grayscale_image = np.zeros((rgb.shape[0], rgb.shape[1], 2), dtype=np.uint8)
                        self.grayscale_image[:,:,0] = rgb
                        self.grayscale_image[:,:,1] = self.image[:,:,3]  # Alpha channel
                    else:
                        # Ignore alpha channel
                        self.grayscale_image = cv2.cvtColor(self.image[:,:,:3], cv2.COLOR_RGB2GRAY)
                else:
                    # RGB image
                    self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        
        return self.grayscale_image

    def get_grayscale_image(self):
        """Return the grayscale version of the image.
        This is used by some detectors.
        """
        if self.grayscale_image is None:
            self.convert_to_grayscale(update_image=False)
        return self.grayscale_image

    def read(self, extension=None, quality=None):
        # Draw rectangles around VVIP faces for debugging
        # Comment this out in production if you don't want visible rectangles
        for (x, y, w, h, name, confidence) in self.vvip_faces:
            cv2.rectangle(self.image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(self.image, f'{name}: {confidence}', (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
        elif len(self.image.shape) == 3 and self.image.shape[2] == 4:
            # Convert RGBA to BGRA for saving
            img = cv2.cvtColor(self.image, cv2.COLOR_RGBA2BGRA)
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

    def save_debug_image(self, tag="debug"):
        """Save a debug image showing all faces and marking VVIPs"""
        debug_img = self.image.copy()
        
        # Convert to BGR for OpenCV saving
        if len(debug_img.shape) == 3 and debug_img.shape[2] == 3:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        elif len(debug_img.shape) == 3 and debug_img.shape[2] == 4:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGBA2BGRA)
        
        # Draw VVIP faces in GREEN
        for (x, y, w, h, _, _) in self.vvip_faces:
            cv2.rectangle(debug_img, (int(x), int(y)), 
                        (int(x + w), int(y + h)), (0, 255, 0), 3)
            
        # Save the debug image
        debug_dir = './tmp/thumbor_debug'
        if not path.exists(debug_dir):
            makedirs(debug_dir)
        
        import time
        timestamp = int(time.time())
        debug_path = path.join(debug_dir, f'face_{tag}_{timestamp}.jpg')
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved to {debug_path}")
        return debug_path