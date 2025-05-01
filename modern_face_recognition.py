# modern_face_recognition.py
import os
import face_recognition
import cv2
import numpy as np
import pickle
from thumbor.utils import logger

class VVIPFaceRecognizer:
    def __init__(self, vvip_faces_dir="vvip_faces", models_dir="models", tolerance=0.6):
        """
        Initialize the VVIP face recognizer using face_recognition library
        
        Args:
            vvip_faces_dir: Directory containing VVIP face images
            models_dir: Directory to save/load the trained model
            tolerance: Face recognition tolerance (lower = stricter)
        """
        self.vvip_faces_dir = vvip_faces_dir
        self.models_dir = models_dir
        self.tolerance = tolerance
        
        # Will store known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Path to the model file
        self.model_path = os.path.join(self.models_dir, "vvip_encodings.pkl")
        
        # Try to load an existing model
        self.load_model()
        
    def load_model(self):
        """Load the saved face recognition model if it exists"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.known_face_encodings = model_data['encodings']
                self.known_face_names = model_data['names']
                print(f"✓ Loaded {len(self.known_face_names)} VVIP faces from model")
                logger.info(f"Loaded {len(self.known_face_names)} VVIP faces from model")
                return True
        except Exception as e:
            logger.error(f"Error loading face recognition model: {str(e)}")
            print(f"Error loading face recognition model: {str(e)}")
        
        return False
    
    def train_model(self):
        """Train a new face recognition model from the VVIP faces directory"""
        if not os.path.exists(self.vvip_faces_dir):
            logger.error(f"VVIP faces directory not found: {self.vvip_faces_dir}")
            return False
            
        # Reset the model data
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Process each VVIP directory
        for person_name in os.listdir(self.vvip_faces_dir):
            person_dir = os.path.join(self.vvip_faces_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            print(f"Processing VVIP: {person_name}")
            
            # Process each image in the person's directory
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(person_dir, img_file)
                
                try:
                    # Load image and find face encodings
                    image = face_recognition.load_image_file(img_path)
                    
                    # Find all faces in the image
                    face_locations = face_recognition.face_locations(image, model="hog")
                    
                    # If no faces are found, try CNN model (more accurate but slower)
                    if len(face_locations) == 0 and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        face_locations = face_recognition.face_locations(image, model="cnn")
                    
                    if len(face_locations) == 0:
                        print(f"  No faces found in {img_file}, skipping")
                        continue
                        
                    # Get face encodings - use the largest face if multiple are found
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    
                    if len(face_encodings) > 0:
                        # Add the face encoding to our model
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person_name)
                        print(f"  Added face from {img_file}")
                    
                except Exception as e:
                    print(f"  Error processing {img_file}: {str(e)}")
        
        # Save the model if we have any face encodings
        if len(self.known_face_encodings) > 0:
            try:
                model_data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                    
                print(f"✓ Saved model with {len(self.known_face_names)} VVIP faces")
                logger.info(f"Saved model with {len(self.known_face_names)} VVIP faces")
                return True
            except Exception as e:
                logger.error(f"Error saving face recognition model: {str(e)}")
                print(f"Error saving face recognition model: {str(e)}")
        else:
            print("No faces found in any of the VVIP images")
            
        return False
    
    def recognize_faces(self, image):
        """
        Recognize VVIP faces in an image
        
        Args:
            image: OpenCV/numpy image array (RGB format)
            
        Returns:
            List of tuples (x, y, w, h, name) for each VVIP face detected
        """
        if len(self.known_face_encodings) == 0:
            logger.warning("No VVIP face models loaded")
            return []
            
        # Scale down image for faster processing if it's large
        height, width = image.shape[:2]
        scale = 1.0
        
        # If image is very large, scale it down for face detection
        if width > 1200 or height > 1200:
            scale = 1200 / max(width, height)
            small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        else:
            small_image = image
        
        # Get all face locations and encodings in the image
        face_locations = face_recognition.face_locations(small_image, model="hog")
        face_encodings = face_recognition.face_encodings(small_image, face_locations)
        
        # If no faces found and CUDA is available, try CNN model
        if len(face_locations) == 0 and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            face_locations = face_recognition.face_locations(small_image, model="cnn")
            if len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(small_image, face_locations)
        
        vvip_faces = []
        
        # Check each face against our known VVIP faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # If we scaled the image, adjust coordinates back to original scale
            if scale != 1.0:
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
            
            # Calculate width and height
            w = right - left
            h = bottom - top
            
            name = "Unknown"
            confidence = 0.0

            # Compare this face against all known VVIP faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=self.tolerance
            )
            
            # If we found a match
            if True in matches:
                # Calculate face distances for all encodings
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )

                # Get the index of the best match
                best_match_index = np.argmin(face_distances)
                confidence = 1.0 - face_distances[best_match_index]

                # Only if it's a strong enough match
                if matches[best_match_index] and confidence >= (1.0 - self.tolerance):
                    name = self.known_face_names[best_match_index]
                    
                    # Find the index of the first match
                    match_index = matches.index(True)
                    name = self.known_face_names[match_index]
                    
                    # Add margin around face for better cropping
                    margin = 0.5  # 50% extra margin
                    x_ext = int(left - (w * margin/2))
                    y_ext = int(top - (h * margin/2))
                    w_ext = int(w * (1 + margin))
                    h_ext = int(h * (1 + margin))
                    
                    # Ensure coordinates are within image bounds
                    x_ext = max(0, x_ext)
                    y_ext = max(0, y_ext)
                    w_ext = min(w_ext, image.shape[1] - x_ext)
                    h_ext = min(h_ext, image.shape[0] - y_ext)
                    
                    vvip_faces.append((x_ext, y_ext, w_ext, h_ext, name, confidence))
                    #print(f"✓ Detected VVIP: {name}")
                    logger.info(f"Detected VVIP: {name}. confidence: {confidence}")
                
        return vvip_faces
    
    def create_test_script(self):
        """Create a test script to validate the face recognition"""
        test_script = """#!/usr/bin/env python3
# test_vvip_recognition.py - Test for modern VVIP face recognition
import face_recognition
import cv2
import sys
import os
import pickle
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_vvip_recognition.py <image_path>")
        return
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load model
    model_path = "models/vvip_encodings.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        known_face_encodings = model_data['encodings']
        known_face_names = model_data['names']
        print(f"Loaded {len(known_face_names)} VVIP faces from model")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load the image
    image = face_recognition.load_image_file(image_path)
    
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image, model="hog")
    print(f"Found {len(face_locations)} faces in image")
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Convert image to OpenCV format for display
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known VVIP faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        
        name = "Unknown"
        confidence = "N/A"
        
        # If we found a match
        if True in matches:
            # Find the best match
            match_index = matches.index(True)
            name = known_face_names[match_index]
            
            # Calculate a confidence score based on face distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            confidence = f"{(1 - face_distances[best_match_index]) * 100:.1f}%"
        
        # Draw rectangle - green if VVIP, red otherwise
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image_cv, (left, top), (right, bottom), color, 2)
        
        # Put text
        cv2.putText(image_cv, f"{name} ({confidence})", 
                   (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display result
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, image_cv)
    print(f"Result saved to {output_path}")
    
    # Try to display image if running in graphical environment
    try:
        cv2.imshow("VVIP Recognition Test", image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
"""
    
        test_script_path = os.path.join(self.models_dir, 'test_vvip_recognition.py')
        with open(test_script_path, 'w') as f:
            f.write(test_script)
        
        # Make executable
        try:
            os.chmod(test_script_path, 0o755)
        except:
            pass
        
        print(f"Test script created: {test_script_path}")
        return test_script_path