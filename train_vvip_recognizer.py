# train_vvip_recognizer.py
import cv2
import os
import numpy as np

def train_vvip_recognizer(vvip_faces_dir):
    """
    Train a face recognizer with VVIP faces
    
    Parameters:
    vvip_faces_dir: Directory containing subdirectories named after each VVIP with their face images
    
    Returns:
    A trained face recognizer model
    """
    print("Training VVIP face recognizer...")
    
    # Create face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Initialize face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize lists for training data
    faces = []
    labels = []
    label_map = {}  # Maps person name to label number
    
    # Load images from each VVIP subdirectory
    label_counter = 0
    
    for person_name in os.listdir(vvip_faces_dir):
        person_dir = os.path.join(vvip_faces_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        # Assign a label to this person
        label_map[person_name] = label_counter
        
        print(f"Processing images for VVIP: {person_name} (Label: {label_counter})")
        
        # Process each image for this person
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(person_dir, img_file)
            
            # Load and convert to grayscale
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            detected_faces = face_detector.detectMultiScale(gray, 1.1, 5)
            
            # Use the whole image if no face is detected
            if len(detected_faces) == 0:
                # Resize to standard size
                face_img = cv2.resize(gray, (100, 100))
                faces.append(face_img)
                labels.append(label_counter)
                print(f"  Using whole image: {img_file}")
            else:
                # Process each detected face
                for (x, y, w, h) in detected_faces:
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (100, 100))
                    faces.append(face_img)
                    labels.append(label_counter)
                    print(f"  Processed face in: {img_file}")
        
        # Increment label counter for next person
        label_counter += 1
    
    print(f"Total training images: {len(faces)}")
    
    # Train the recognizer if we have data
    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        
        # Save the model
        model_path =os.path.join(vvip_faces_dir, 'vvip_faces_model.yml')
        recognizer.write(model_path)
        print(f"Model saved to {model_path}")
        
        # Save the label mapping
        with open(os.path.join(vvip_faces_dir,'vvip_labels.txt'), 'w') as f:
            for person, label in label_map.items():
                f.write(f"{label}:{person}\n")
        print(f"Label mapping saved to vvip_labels.txt")
    else:
        print("No training data collected!")
    
    return recognizer

if __name__ == "__main__":
    # Specify directory containing VVIP face images
    # Structure should be:
    # vvip_faces/
    #   vvip_name1/
    #     image1.jpg
    #     image2.jpg
    #   vvip_name2/
    #     image1.jpg
    #     ...
    vvip_dir = "vvip_faces"
    
    # Create directory if it doesn't exist
    if not os.path.exists(vvip_dir):
        os.makedirs(vvip_dir)
        print(f"Created directory {vvip_dir}. Please add VVIP face images there.")
    else:
        # Train the recognizer
        train_vvip_recognizer(vvip_dir)