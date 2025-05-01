#!/usr/bin/env python3
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
