#!/usr/bin/env python3
# train_vvip_recognizer.py - Train the modern face recognition model
import os
import sys
import cv2

# Import our modern face recognizer
from modern_face_recognition import VVIPFaceRecognizer

def main():
    """Train the face recognition model with VVIP images"""
    # Specify directory containing VVIP face images
    vvip_dir = "vvip_faces"
    models_dir = "models"
    
    # Create directory if it doesn't exist
    if not os.path.exists(vvip_dir):
        os.makedirs(vvip_dir)
        print(f"Created directory {vvip_dir}. Please add VVIP face images.")
        print("Directory structure should be:")
        print("  vvip_faces/")
        print("    person_name1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    person_name2/")
        print("      ...")
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Initialize the face recognizer
    recognizer = VVIPFaceRecognizer(
        vvip_faces_dir=vvip_dir,
        models_dir=models_dir
    )
    
    # Check if face_recognition library is installed
    try:
        import face_recognition
        print("✓ face_recognition library is installed")
    except ImportError:
        print("ERROR: face_recognition library is not installed")
        print("Please install it with: pip install face_recognition")
        sys.exit(1)
    
    # Train the model
    print("Training VVIP face recognition model...")
    success = recognizer.train_model()
    
    if success:
        print("✓ Training completed successfully")
        
        # Create test script
        recognizer.create_test_script()
        
        print("\nTesting instructions:")
        print("1. To test recognition on an image:")
        print(f"   python {os.path.join(models_dir, 'test_vvip_recognition.py')} <path_to_test_image>")
        print("\n2. To use in Thumbor:")
        print("   Make sure opencv_engine.py imports modern_face_recognition.py")
        print("   Restart your Thumbor server")
    else:
        print("✗ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()