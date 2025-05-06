import os
import logging
import json
from tornado.web import RequestHandler
from direct_smart_crop import SmartCropper
import requests
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
    Sample Request:
    {
    "image": "https://asset.hamdan.ae/CPDAssets/News/2025/5290/1042025171313442.JPG",
    "dimensions" : [
        {
        "width": 1080,
        "height": 1920
        },
        {
        "width": 1920,
        "height": 1080
        }
    ],
    "save_image" : true,
    "format": "jpeg"
    }
    """
    async def post(self):
        try:
            # Parse request parameters
            data = json.loads(self.request.body)
            image_url = data.get('image')            
            dimensions = data.get('dimensions', [{'width':960,'height':540}])            
            format_ext = data.get('format', 'jpg')
            save_image : bool = data.get('save_image', False)

            # Validate input
            if not image_url:
                self.set_status(400)
                self.write({"error": "Image URL is required"})
                return

            image_filename = os.path.basename(image_url)
            input_path = os.path.join(UPLOAD_DIR, image_filename)

            # Download image
            response = requests.get(image_url, stream=True)
            if response.status_code != 200:
                self.set_status(400)
                self.write({"error": "Failed to download image"})
                return

            with open(input_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            if not os.path.exists(input_path):
                self.set_status(404)
                self.write({"error": f"Image {image_filename} is not downloaded"})
                return 

            crop_dimensions_list = []
            # Process the image directly without HTTP requests
            for dim in dimensions:
                width = dim.get('width',960)
                height = dim.get('height',540)
                logger.info(f"Processing image {image_filename} to {width}x{height}")

                # Form paths
                output_filename = f"smart_{width}x{height}_{image_filename}"
                if '.' in output_filename:
                    output_name, _ = os.path.splitext(output_filename)
                    output_filename = f"{output_name}.{format_ext}"
                output_path = os.path.join(OUTPUT_DIR, output_filename)                
                
                crop_dimensions = await cropper.save_smart_cropped_image(
                    image_path=input_path,
                    width=width,
                    height=height,
                    output_path=output_path,
                    extension=format_ext,
                    save_image = save_image,
                    keep_fullsized= False
                )
                crop_dimensions_list.append(
                    {
                        "rendition" : f'{width}x{height}',
                        "crop_dimensions" :{
                            "x" : crop_dimensions[0],
                            "y" : crop_dimensions[1],
                            "right" : crop_dimensions[2],
                            "bottom" : crop_dimensions[3] 
                        }
                    }
                )
            
            # Return the result
            result = {
                "success": True,
                "original": image_filename,
                "format": format_ext,                
                "crop_info" : crop_dimensions_list
            }
            
            self.set_header("Content-Type", "application/json")
            self.write(result)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            self.set_status(500)
            self.write({"error": str(e)})


# How to integrate these handlers into your existing server
def add_direct_processing_handlers(application):
    """
    Add the direct processing handlers to your existing Tornado application
    """
    application.add_handlers(r".*", [
        (r"/process", ProcessImageHandler),
    ])
    
    logger.info("Added direct image processing handlers")
    
    return application