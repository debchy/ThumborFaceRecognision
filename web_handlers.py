import os
import json
import time
import mimetypes
import tornado.web
import tornado.ioloop
from typing import List, Dict, Any, Optional, Tuple
import threading
import urllib.parse
import requests
from datetime import datetime
from direct_smart_crop import Rendition, SmartCropper

# Create a global smart cropper instance
cropper = SmartCropper("./thumbor.conf")

# Ensure directories exist
UPLOAD_DIR = "thumbor_images/uploads"
OUTPUT_DIR = "thumbor_images/outputs/thumbor"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define image dimensions for processing
CROP_DIMENSIONS = [
    (1080, 1920),  # Portrait
    (1920, 1080),  # Landscape
    (1080, 1080)   # Square
]

class MainHandler(tornado.web.RequestHandler):
    """Handler for the main page"""
    def get(self):
        with open("./thumbor_images/upload_interface.html", "r") as file:
            self.write(file.read())

class UploadHandler(tornado.web.RequestHandler):
    """Handler for image uploads and processing"""
    async def post(self):
        try:
            files = self.request.files.get('images', [])
            
            if not files:
                self.set_status(400)
                self.write({"success": False, "error": "No files uploaded"})
                return
                
            processed_count = 0
            errors = []
            
            for file_info in files:
                try:
                    # Save the uploaded file
                    filename = file_info['filename']
                    safe_filename = get_safe_filename(filename)
                    file_path = os.path.join(UPLOAD_DIR, safe_filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(file_info['body'])
                    
                    # Process the image using Thumbor
                    await process_image_with_thumbor(file_path, safe_filename)
                    
                    processed_count += 1
                    
                except Exception as e:
                    errors.append(f"Error processing {filename}: {str(e)}")
            
            if errors:
                self.write({
                    "success": processed_count > 0,
                    "processed_count": processed_count,
                    "errors": errors
                })
            else:
                self.write({
                    "success": True,
                    "processed_count": processed_count
                })
                
        except Exception as e:
            self.set_status(500)
            self.write({"success": False, "error": str(e)})

class ListImagesHandler(tornado.web.RequestHandler):
    """Handler to list all processed images"""
    def get(self):
        try:
            images = list_processed_images()
            self.write({"success": True, "images": images})
        except Exception as e:
            self.set_status(500)
            self.write({"success": False, "error": str(e)})

class ImageHandler(tornado.web.RequestHandler):
    """Handler to serve images from the output directory"""
    def get(self, image_path):
        try:
            # Ensure image_path is just a filename, not a path
            filename = os.path.basename(image_path)
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            if not os.path.exists(file_path):
                self.set_status(404)
                self.write("Image not found")
                return
                
            # Determine the content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type:
                self.set_header("Content-Type", content_type)
                
            # Serve the file
            with open(file_path, 'rb') as f:
                self.write(f.read())
                
        except Exception as e:
            self.set_status(500)
            self.write(f"Error: {str(e)}")

def get_safe_filename(filename: str) -> str:
    """Generate a safe filename with timestamp to avoid conflicts"""
    base_name, ext = os.path.splitext(filename)
    timestamp = int(time.time())
    safe_name = f"{base_name}_{timestamp}{ext}"
    return safe_name

async def process_image_with_thumbor(file_path: str, filename: str) -> None:
    """Process an image with Thumbor to create different crop dimensions"""
    # Get the relative path from upload directory to the file
    relative_path = os.path.relpath(file_path, UPLOAD_DIR)
    encoded_path = urllib.parse.quote(relative_path, safe='')
    
    base_name, ext = os.path.splitext(filename)
    renditions = [] 
    # Create crops for each dimension
    for width, height in CROP_DIMENSIONS:
        # Use Thumbor's smart crop
        output_filename = f"{base_name}_{width}x{height}{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        renditions.append(Rendition(width=width, height= height, output_path=output_path))

    try:
        await cropper.smart_crop_multiple(
            image_path      = file_path,
            renditions      = renditions,
            extension       = ext,
            save_image      = True,
            keep_fullsized  = False
        )
    except Exception as e:
        print(f"Error processing", e)
    
def list_processed_images() -> List[Dict[str, Any]]:
    """List all processed images with their metadata"""
    result = []
    
    # Dictionary to track VVIP detection by original image
    vvip_detection = {}
    
    # Get all files in the output directory
    files = os.listdir(OUTPUT_DIR)
    
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            continue
            
        file_path = os.path.join(OUTPUT_DIR, file)
        file_stats = os.stat(file_path)
        
        # Extract dimensions from filename (base_name_WIDTHxHEIGHT.ext)
        parts = file.split('_')
        dimensions = parts[-1].split('.')[0]  # Extract WIDTHxHEIGHT
        
        # Extract the original filename (everything before the dimensions)
        original_name_parts = parts[:-1]
        timestamp = original_name_parts[-1]  # The timestamp is the last part before dimensions
        original_name_parts = original_name_parts[:-1]  # Remove timestamp
        original_name = '_'.join(original_name_parts)
        
        # Check if this file has VVIP detection (if applicable)
        has_vvip = False
        
        # We could check for a flag file or determine based on file analysis
        # For demonstration, we'll randomly assign VVIP detection
        # In a real implementation, you'd want to check if Thumbor detected VVIPs
        # This could be done by checking debug logs or having Thumbor modify the filename
        
        # Group by original filename for VVIP detection tracking
        if original_name not in vvip_detection:
            # In a real implementation, check if VVIP was detected
            # For demo purposes, assuming no VVIP detection
            vvip_detection[original_name] = False
        
        has_vvip = vvip_detection[original_name]
        
        # Add image info to result
        result.append({
            "filename": file,
            "original_name": original_name,
            "dimensions": dimensions,
            "timestamp": int(timestamp),
            "size": file_stats.st_size,
            "url": f"/images/{file}",
            "has_vvip": has_vvip
        })
    
    return result