import os
import logging
import json
from tornado.web import RequestHandler
from direct_smart_crop import SmartCropper

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('thumbor_integration')

# Directory setup
UPLOAD_DIR = "thumbor_images/uploads"
OUTPUT_DIR = "thumbor_images/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a global smart cropper instance
cropper = SmartCropper("./thumbor.conf")

# Custom handlers that use direct smart cropping instead of internal HTTP requests

class ProcessImageHandler(RequestHandler):
    """
    Handler to process images directly without making internal HTTP requests
    """
    async def post(self):
        try:
            # Parse request parameters
            data = json.loads(self.request.body)
            image_filename = data.get('image')
            width = int(data.get('width', 800))
            height = int(data.get('height', 600))
            format_ext = data.get('format', 'jpg')
            
            # Validate input
            if not image_filename:
                self.set_status(400)
                self.write({"error": "Image filename is required"})
                return
                
            # Form paths
            input_path = os.path.join(UPLOAD_DIR, image_filename)
            output_filename = f"smart_{width}x{height}_{image_filename}"
            if '.' in output_filename:
                output_name, _ = os.path.splitext(output_filename)
                output_filename = f"{output_name}.{format_ext}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            if not os.path.exists(input_path):
                self.set_status(404)
                self.write({"error": f"Image {image_filename} not found"})
                return
            
            # Process the image directly without HTTP requests
            logger.info(f"Processing image {image_filename} to {width}x{height}")
            output_file = await cropper.save_smart_cropped_image(
                image_path=input_path,
                width=width,
                height=height,
                output_path=output_path,
                extension=format_ext
            )
            
            # Return the result
            result = {
                "success": True,
                "original": image_filename,
                "processed": output_filename,
                "width": width,
                "height": height,
                "format": format_ext,
                "url": f"/images/{output_filename}"
            }
            
            self.set_header("Content-Type", "application/json")
            self.write(result)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            self.set_status(500)
            self.write({"error": str(e)})


class BatchProcessHandler(RequestHandler):
    """
    Handler to batch process multiple images
    """
    async def post(self):
        try:
            # Parse request parameters
            data = json.loads(self.request.body)
            batch_jobs = data.get('jobs', [])
            
            if not batch_jobs:
                self.set_status(400)
                self.write({"error": "No jobs specified"})
                return
            
            results = []
            errors = []
            
            # Process each job
            for job in batch_jobs:
                try:
                    image_filename = job.get('image')
                    width = int(job.get('width', 800))
                    height = int(job.get('height', 600))
                    format_ext = job.get('format', 'jpg')
                    
                    # Validate job
                    if not image_filename:
                        errors.append({"job": job, "error": "Image filename is required"})
                        continue
                        
                    # Form paths
                    input_path = os.path.join(UPLOAD_DIR, image_filename)
                    output_filename = f"smart_{width}x{height}_{image_filename}"
                    if '.' in output_filename:
                        output_name, _ = os.path.splitext(output_filename)
                        output_filename = f"{output_name}.{format_ext}"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    
                    if not os.path.exists(input_path):
                        errors.append({"job": job, "error": f"Image {image_filename} not found"})
                        continue
                    
                    # Process the image directly
                    output_file = cropper.save_smart_cropped_image(
                        image_path=input_path,
                        width=width,
                        height=height,
                        output_path=output_path,
                        extension=format_ext
                    )
                    
                    # Add to results
                    results.append({
                        "original": image_filename,
                        "processed": output_filename,
                        "width": width,
                        "height": height,
                        "format": format_ext,
                        "url": f"/images/{output_filename}"
                    })
                    
                except Exception as e:
                    errors.append({"job": job, "error": str(e)})
            
            # Return all results
            response = {
                "success": len(errors) == 0,
                "processed": len(results),
                "total": len(batch_jobs),
                "results": results
            }
            
            if errors:
                response["errors"] = errors
                
            self.set_header("Content-Type", "application/json")
            self.write(response)
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            self.set_status(500)
            self.write({"error": str(e)})


# How to integrate these handlers into your existing server
def add_direct_processing_handlers(application):
    """
    Add the direct processing handlers to your existing Tornado application
    """
    application.add_handlers(r".*", [
        (r"/process", ProcessImageHandler),
        (r"/batch_process", BatchProcessHandler),
    ])
    
    logger.info("Added direct image processing handlers")
    
    return application