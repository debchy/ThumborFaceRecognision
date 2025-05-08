import copy
import os
import io
import asyncio
import logging
from typing import Optional
from PIL import Image
from thumbor.engines.pil import Engine as PILEngine
from thumbor.point import FocalPoint
from thumbor.detectors.feature_detector import Detector as FeatureDetector
from thumbor.detectors.face_detector import Detector as FaceDetector
from thumbor.detectors.profile_detector import Detector as ProfileDetector
from thumbor.context import Context, ServerParameters
from thumbor.config import Config
from thumbor.importer import Importer
from thumbor.utils import logger
from opencv_engine import Engine as OpenCVEngine
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('direct_smart_crop')

class MockRequest:
    """
    Mock request object to satisfy Thumbor's detector requirements
    """
    def __init__(self, image_url):
        self.url = image_url
        self.headers = {}
        self.successful_operations = []
        self.defuser = None
        self.depth = 0
        self.query_params = {}
        self.extension = None
        self.focal_points = []
        self.image = None
        self.engine = None
        self.should_crop = True
        self.crop = None
        self.buffer = None
        self.operations = []
        self.filters = []
        self.detection_error = None
        self.max_bytes = None
        self.auto_png_to_jpg = None

class Rendition:
    """
    A class representing a crop rendition with dimensions and output path
    """
    def __init__(self, width: int, height: int, output_path: Optional[str] = None):
        self.width = width
        self.height = height
        self.output_path = output_path
        self.crop_dimensions = None
        self.img_buffer = None
        self.fullsized_img_buffer = None
        
    def __str__(self):
        return f"Rendition({self.width}x{self.height})"

class SmartCropper:
    """
    A class that performs Thumbor's smart cropping directly without HTTP requests
    """
    def __init__(self, config_path=None):
        """Initialize the smart cropper with optional config path"""
        # Use provided config or create a default one
        if config_path and os.path.exists(config_path):
            self.config = Config.load(config_path)
            if not hasattr(self.config, 'DETECTORS') or not self.config.DETECTORS:
                self.config.DETECTORS = [
                    'thumbor.detectors.face_detector',
                    'thumbor.detectors.feature_detector',
                    'thumbor.detectors.profile_detector'
                ]
            self.config.MAX_AGE = 0
        else:
            logger.info(f'Using default configuration, path not found: {config_path}')
            self.config = Config()
            # Set minimum required configuration
            self.config.SECURITY_KEY = 'THUMBOR_SECURITY_KEY'
            self.config.ALLOW_UNSAFE_URL = True
            self.config.STORAGE = 'thumbor.storages.no_storage'
            self.config.LOADER = 'thumbor.loaders.file_loader'
            self.config.FILE_LOADER_ROOT_PATH = '.'
            self.config.DETECTORS = [
                'thumbor.detectors.face_detector',
                'thumbor.detectors.feature_detector',
            ]
            self.config.FACE_DETECTOR_CASCADE_FILE = 'haarcascade_frontalface_default.xml'
        
        # Set up a mock server parameters
        server_parameters = ServerParameters(
            port=8888,
            ip='127.0.0.1',
            config_path=config_path,
            keyfile=None,
            log_level='debug',
            app_class='thumbor.app.ThumborServiceApp',
            fd=None,
            debug= True
        )
        
        # Import Thumbor modules
        self.importer = Importer(self.config)
        self.importer.import_modules()
        
        # Create context
        self.context = Context(server=server_parameters, config=self.config, importer=self.importer)

    def _create_engine(self, image_buffer, extension='.jpg'):
        """Create a Thumbor engine with the image"""
        
        if hasattr(self.config, 'ENGINE') and self.config.ENGINE == "opencv_engine" :
            engine = OpenCVEngine(self.context)
            engine.load(image_buffer, extension)
        else:
            engine = PILEngine(self.context)
            engine.load(image_buffer, None)

        return engine

    async def _run_detector(self, detector_class, index, name):
        """Run a specific detector and handle exceptions"""
        try:
            self.context.request.focal_points = [] #clean up result for previous detection
            detector = detector_class(self.context, index, [])  # Empty list to prevent next() calls
            detector.detector_name = name
            
            # Setup any detector-specific attributes
            if hasattr(detector, 'setup'):
                await detector.setup()
            
            fp = await detector.detect()
            logger.info(f"{name} found {len(self.context.request.focal_points)} focal points")
            return self.context.request.focal_points
        except Exception as e:
            logger.warning(f"{name} detection failed: {str(e)}")
            return []
    
    def _intersects(self, fp1, fp2):
        """Check if two FocalPoints intersect"""
        x1, y1, w1, h1 = fp1.x, fp1.y, fp1.width, fp1.height
        x2, y2, w2, h2 = fp2.x, fp2.y, fp2.width, fp2.height
        return (
            x1 <= x2 + w2 and
            x1 + w1 >= x2 and
            y1 <= y2 + h2 and
            y1 + h1 >= y2
        )

    async def _get_focal_points_async(self, engine, image_path):
        """Get focal points for smart cropping"""
        focal_points = []
        vvip_count = 0
        
        # Setup mock request
        mock_request = MockRequest(image_path)
        self.context.request = mock_request
        self.context.modules.engine = engine
        
        # Also set the engine in the request
        mock_request.engine = engine
        mock_request.image = engine
        
        # Check if we have VVIP faces from OpenCV engine
        if hasattr(engine, 'vvip_faces') and engine.vvip_faces:
            vvip_count = len(engine.vvip_faces)
            for (x, y, w, h, name, confidence) in engine.vvip_faces:
                # Give VVIP faces a very high weight to ensure they're prioritized
                weight = 10000  # High weight to ensure VVIP faces are prioritized
                fp = FocalPoint(
                    x + w/2,  # Center X of face
                    y + h/2,  # Center Y of face
                    width=w,
                    height=h,
                    weight= w*h, #weight*confidence,
                    origin="VVIP"
                )
                focal_points.append(fp)
                logger.info(f"Using VVIP face as focal point: {name} at ({x}, {y})")
        
        if vvip_count<2 :
            # Run face detector
            face_points = await self._run_detector(FaceDetector, 0, "face_detector")
            focal_points.extend(face_points)

            # Run feature detector
            feature_points = await self._run_detector(FeatureDetector, 2, "feature_detector")
            focal_points.extend(feature_points)

            # Run profile detector
            profile_points = await self._run_detector(ProfileDetector, 1, "profile_detector")
            focal_points.extend(profile_points)

        
        #if only single vvip is detected then, adjust vvip weight when other person's weight is higher
        if vvip_count==1 and len(focal_points)>1:   
            # Get the weight from focal_points where origin=='VVIP'         
            vvip_fp = next((fp for fp in focal_points if fp.origin == 'VVIP'), None)
            vvip_weight = vvip_fp.weight if vvip_fp else 0

            #if any fp overlaps with vvip_fp, then remove it from feature_points
            if vvip_fp:                
                overlapped_points = [fp for fp in focal_points if fp.origin != 'VVIP' and self._intersects(vvip_fp, fp)]
                logger.info(f'overlapped points: {overlapped_points}' )                 
                focal_points = [fp for fp in focal_points if fp not in overlapped_points] if len(overlapped_points)>0 else focal_points
            
            # Cap weights to prevent extreme values
            MAX_WEIGHT = 1000  # Choose appropriate max value
            capped_focal_points = [
                FocalPoint(fp.x, fp.y, fp.width, fp.height, 
                        min(fp.weight, MAX_WEIGHT + 100 if fp.origin=="VVIP" else 0), 
                        fp.origin) 
                for fp in focal_points
            ]
            focal_points = capped_focal_points

            # # Get the FoculPoint from focal_points where weight is max
            # max_weight_fp = max([fp for fp in focal_points if fp.origin != 'VVIP'], key=lambda fp: fp.weight, default=None)    
            # #adjust vvip weight when other person's weight is higher
            # if vvip_fp and max_weight_fp and max_weight_fp.weight >= vvip_weight and vvip_weight>0:                
            #     for i, fp in enumerate(focal_points):
            #         if fp.origin == 'VVIP':
            #             focal_points[i].weight = max_weight_fp.weight+100
            #             break
            #     logger.info(f"Increasing VVIP face weight as max focal point wight is greater than his one")

            

        # If still no focal points, use center of image
        if not focal_points:
            logger.info("No faces or features detected, using center of image")
            width, height = engine.size
            focal_points = [FocalPoint(width / 2, height / 2, 1)]
        
        print('focal_points:\n',focal_points)
        return focal_points

    def _calculate_crop_dimensions(self, engine, width, height, focal_points):
        """
        Calculate the crop dimensions based on the focal points
        This replicates Thumbor's focal point based cropping algorithm
        """
        source_width, source_height = engine.size
        
        if width > source_width or height > source_height:
            logger.warning("Target dimensions exceed source image dimensions")
        
        # Calculate the target aspect ratio
        target_aspect = float(width) / float(height) if height > 0 else 1
        source_aspect = float(source_width) / float(source_height) if source_height > 0 else 1
        
        # Determine crop dimensions
        if source_aspect > target_aspect:
            # Image is wider than needed
            new_height = source_height
            new_width = int(target_aspect * new_height)
        else:
            # Image is taller than needed
            new_width = source_width
            new_height = int(new_width / target_aspect)
        
        # Calculate crop coordinates based on focal points
        if not focal_points:
            # Center crop if no focal points
            left = int((source_width - new_width) / 2)
            top = int((source_height - new_height) / 2)
        else:
            # Calculate weighted center point
            total_weight = sum(fp.weight for fp in focal_points)
            weighted_x = sum(fp.x * fp.weight for fp in focal_points) / total_weight if total_weight > 0 else source_width / 2
            weighted_y = sum(fp.y * fp.weight for fp in focal_points) / total_weight if total_weight > 0 else source_height / 2
            
            # Calculate crop coordinates
            left = max(0, min(source_width - new_width, int(weighted_x - new_width / 2)))
            top = max(0, min(source_height - new_height, int(weighted_y - new_height / 2)))
        
        # Ensure coordinates are within bounds
        left = max(0, min(left, source_width - new_width))
        top = max(0, min(top, source_height - new_height))
        
        return (left, top, left + new_width, top + new_height)

    
    def draw_focal_points(self, image_path, focal_points, renditions, original_dimensions):
        """Draw focal points and multiple crop dimensions on the image for debugging."""
        try:
            image = Image.open(image_path)  # Open the image
            # Create a drawing context
            draw = ImageDraw.Draw(image)

            # Draw each focal point as a circle with size and color based on weight and origin
            colors = {
                "VVIP": "yellow",
                "face_detector": "red",
                "profile_detector": "blue",
                "feature_detector": "green",
                "unknown": "white"
            }

            # Calculate weighted center point for visualization
            if focal_points:
                total_weight = sum(fp.weight for fp in focal_points)
                weighted_x = sum(fp.x * fp.weight for fp in focal_points) / total_weight if total_weight > 0 else image.width / 2
                weighted_y = sum(fp.y * fp.weight for fp in focal_points) / total_weight if total_weight > 0 else image.height / 2
            else:
                weighted_x, weighted_y = image.width / 2, image.height / 2

            # Draw the weighted center point as a larger purple circle
            center_radius = 10
            draw.ellipse([
                weighted_x - center_radius, 
                weighted_y - center_radius, 
                weighted_x + center_radius, 
                weighted_y + center_radius
            ], outline="purple", width=3)
            draw.text((weighted_x + center_radius + 2, weighted_y), "Weighted Center", fill="purple")
            
            for fp in focal_points:
                # Base radius on weight, but keep it reasonable
                weight_factor = min(1.0, fp.weight / 1000) if fp.weight > 1000 else fp.weight / 1000
                base_radius = 5
                radius = base_radius + (base_radius * weight_factor)
                
                x, y = fp.x, fp.y
                # Calculate the bounds of the circle
                left = x - radius
                top = y - radius
                right = x + radius
                bottom = y + radius
                
                # Get origin if available, otherwise use unknown
                origin = getattr(fp, 'origin', 'unknown')
                color = colors.get(origin, colors['unknown'])
                
                # Draw circle with color based on origin
                draw.ellipse([left, top, right, bottom], outline=color, width=2)
                
                # Add weight and origin as text
                draw.text((x + radius + 2, y - radius), f"{origin}: {fp.weight:.1f}", fill=color)

            # Create separate visualization for each rendition
            for idx, rendition in enumerate(renditions):
                # Create a copy of the original image for this rendition's visualization
                rendition_img = image.copy()
                rendition_draw = ImageDraw.Draw(rendition_img)
                
                # Draw the same focal points on the rendition image
                for fp in focal_points:
                    # Base radius on weight, but keep it reasonable
                    weight_factor = min(1.0, fp.weight / 1000) if fp.weight > 1000 else fp.weight / 1000
                    base_radius = 5
                    radius = base_radius + (base_radius * weight_factor)
                    
                    x, y = fp.x, fp.y
                    # Calculate the bounds of the circle
                    left = x - radius
                    top = y - radius
                    right = x + radius
                    bottom = y + radius
                    
                    # Get origin if available, otherwise use unknown
                    origin = getattr(fp, 'origin', 'unknown')
                    color = colors.get(origin, colors['unknown'])
                    
                    # Draw circle with color based on origin
                    rendition_draw.ellipse([left, top, right, bottom], outline=color, width=2)
                
                # Draw the weighted center point
                rendition_draw.ellipse([
                    weighted_x - center_radius, 
                    weighted_y - center_radius, 
                    weighted_x + center_radius, 
                    weighted_y + center_radius
                ], outline="purple", width=3)
                
                # Draw crop dimensions for this specific rendition
                if rendition.crop_dimensions:
                    left, top, right, bottom = rendition.crop_dimensions
                    
                    # Create semi-transparent overlay
                    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    
                    # Draw transparent rectangle over the whole image
                    overlay_draw.rectangle([(0, 0), image.size], fill=(0, 0, 0, 128))
                    
                    # Draw transparent hole where the crop will be
                    overlay_draw.rectangle([left, top, right, bottom], fill=(0, 0, 0, 0))
                    
                    # Draw border around crop area with rendition dimensions as label
                    overlay_draw.rectangle([left, top, right, bottom], outline="white", width=4)
                    overlay_draw.text(
                        (left + 10, top + 10), 
                        f"Rendition {idx+1}: {rendition.width}x{rendition.height}", 
                        fill="white"
                    )
                    
                    # Convert overlay to same mode as original image if needed
                    if rendition_img.mode == 'RGB':
                        # Draw the crop rectangle directly
                        rendition_draw.rectangle([left, top, right, bottom], outline="white", width=4)
                        rendition_draw.text(
                            (left + 10, top + 10), 
                            f"Rendition {idx+1}: {rendition.width}x{rendition.height}", 
                            fill="white"
                        )
                    else:
                        # Paste with alpha for RGBA images
                        rendition_img.paste(overlay, (0, 0), overlay)

                # Make sure the debug directory exists
                debug_dir = "./tmp/thumbor_debug/"
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir, exist_ok=True)

                # Save this rendition's visualization
                rendition_output = f"{debug_dir}debug_rendition_{rendition.width}x{rendition.height}.jpg"
                rendition_img.save(rendition_output)
                logger.info(f"Debug image for rendition {rendition.width}x{rendition.height} saved to {rendition_output}")
            
            # Create a composite visualization with all crop boxes
            composite_img = image.copy()
            composite_draw = ImageDraw.Draw(composite_img)
            
            # Draw the focal points and weighted center on the composite
            for fp in focal_points:
                origin = getattr(fp, 'origin', 'unknown')
                color = colors.get(origin, colors['unknown'])
                composite_draw.ellipse([
                    fp.x - 5, fp.y - 5, fp.x + 5, fp.y + 5
                ], outline=color, width=2)
            
            composite_draw.ellipse([
                weighted_x - center_radius, 
                weighted_y - center_radius, 
                weighted_x + center_radius, 
                weighted_y + center_radius
            ], outline="purple", width=3)
            
            # Draw all crop rectangles with different colors
            crop_colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
            for idx, rendition in enumerate(renditions):
                if rendition.crop_dimensions:
                    color = crop_colors[idx % len(crop_colors)]
                    left, top, right, bottom = rendition.crop_dimensions
                    composite_draw.rectangle([left, top, right, bottom], outline=color, width=3)
                    composite_draw.text(
                        (left + 10, top + 10), 
                        f"{rendition.width}x{rendition.height}", 
                        fill=color
                    )
            
            # Add original dimensions
            if original_dimensions:
                w, h = original_dimensions
                composite_draw.text((10, 10), f"Original: {w}x{h}", fill="white")
            
            # Save the composite visualization
            composite_output = f"{debug_dir}debug_all_renditions.jpg"
            composite_img.save(composite_output)
            logger.info(f"Composite debug image with all renditions saved to {composite_output}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating debug visualization: {str(e)}")
            return False

    async def smart_crop_multiple(self, image_path, renditions, extension=None, save_image=True, keep_fullsized=False):
        """
        Perform smart cropping for multiple renditions in a single pass
        
        Args:
            image_path: Path to the input image file
            renditions: List of Rendition objects with width, height and output_path
            extension: Output format extension (None for same as input)
            save_image: Whether to save the image
            keep_fullsized: Whether to also save the full-sized cropped image
            
        Returns:
            list: List of updated Rendition objects with crop dimensions and image buffers
        """
        logger.info(f"Smart cropping image {image_path} for {len(renditions)} renditions")
        
        # Read image file
        with open(image_path, 'rb') as f:
            image_buffer = f.read()
        
        # Get file extension if not provided
        if not extension:
            _, file_ext = os.path.splitext(image_path)
            extension = file_ext
        
        # Make sure extension is in correct format (with dot)
        if extension and isinstance(extension, str) and not extension.startswith('.'):
            extension = '.' + extension

        # Create engine and load image
        engine = self._create_engine(image_buffer, extension)
        original_dimensions = engine.size
        
        # Get focal points for smart cropping - this is done only once!
        focal_points = await self._get_focal_points_async(engine, image_path)
        
        engine_image = copy.deepcopy(engine.image)
        engine_grayscale_image = copy.deepcopy(engine.grayscale_image)
        engine_vvip_faces = copy.deepcopy(engine.vvip_faces)
        
        # Process each rendition
        for rendition in renditions:
            # Clone the engine for this rendition
            rendition_engine = engine
            rendition_engine.image = copy.deepcopy(engine_image)
            rendition_engine.grayscale_image = copy.deepcopy(engine_grayscale_image)
            rendition_engine.vvip_faces = copy.deepcopy(engine_vvip_faces)

            # Calculate crop dimensions based on focal points
            crop_dimensions = self._calculate_crop_dimensions(
                rendition_engine, rendition.width, rendition.height, focal_points
            )
            rendition.crop_dimensions = crop_dimensions
            
            logger.info(f"Crop dimensions for {rendition.width}x{rendition.height}: {crop_dimensions}")
            
            # Crop the image
            rendition_engine.crop(*crop_dimensions)
            
            # Save full-sized cropped version
            rendition.fullsized_img_buffer = rendition_engine.read(extension)
            
            # Resize to final dimensions
            rendition_engine.resize(rendition.width, rendition.height)
            
            # Get final image buffer
            rendition.img_buffer = rendition_engine.read(extension)
            
            # Save the image if requested and output path is provided
            if save_image and rendition.output_path:
                # Make sure the output directory exists
                output_dir = os.path.dirname(rendition.output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # Save the processed image
                with open(rendition.output_path, 'wb') as f:
                    f.write(rendition.img_buffer)
                
                logger.info(f"Saved rendition {rendition.width}x{rendition.height} to {rendition.output_path}")
                
                # Save full-sized cropped version if requested
                if keep_fullsized:
                    _, file_ext = os.path.splitext(rendition.output_path)
                    output_path_without_ext, _ = os.path.splitext(rendition.output_path)
                    output_path_original = f"{output_path_without_ext}_original{file_ext}"
                    
                    with open(output_path_original, 'wb') as f:
                        f.write(rendition.fullsized_img_buffer)
                    
                    logger.info(f"Saved full-sized crop for {rendition.width}x{rendition.height} to {output_path_original}")
        
        # Create debug visualization with all renditions
        self.draw_focal_points(image_path, focal_points, renditions, original_dimensions)
        
        return renditions

    def smart_crop_multiple_sync(self, image_path, renditions, extension=None, save_image=True, keep_fullsized=False):
        """
        Synchronous wrapper for the async smart_crop_multiple method
        
        Args:
            image_path: Path to the input image file
            renditions: List of Rendition objects or tuples of (width, height, output_path)
            extension: Output format extension (None for same as input)
            save_image: Whether to save the images
            keep_fullsized: Whether to also save the full-sized cropped images
            
        Returns:
            list: List of Rendition objects with crop dimensions and image buffers
        """
        # Convert simple tuples to Rendition objects if needed
        processed_renditions = []
        for item in renditions:
            if isinstance(item, Rendition):
                processed_renditions.append(item)
            elif isinstance(item, dict):
                width = item.get('width')
                height = item.get('height')
                output_path = item.get('output_path')
                processed_renditions.append(Rendition(width, height, output_path))
            elif isinstance(item, tuple) and len(item) >= 2:
                width, height = item[0], item[1]
                output_path = item[2] if len(item) > 2 else None
                processed_renditions.append(Rendition(width, height, output_path))
            else:
                raise ValueError(f"Invalid rendition format: {item}")
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.smart_crop_multiple(
                image_path, processed_renditions, extension, save_image, keep_fullsized
            )
        )

    
# Example usage
if __name__ == "__main__":
    # Create a smart cropper instance
    cropper = SmartCropper("./thumbor.conf")  # Use your thumbor config if available
    # Example image processing for multiple renditions
    input_image = "test_images/942025181911706.jpg"
    
    # Define different renditions
    renditions = [
        # Portrait - 9:16
        Rendition(width=1080, height=1920, output_path="thumbor_images/outputs/portrait.jpg"),
        # Landscape - 16:9
        Rendition(width=1920, height=1080, output_path="thumbor_images/outputs/landscape.jpg"),
        # Square - 1:1
        Rendition(width=1080, height=1080, output_path="thumbor_images/outputs/square.jpg"),
    ]
    
    # Process all renditions in a single pass
    result = cropper.smart_crop_multiple_sync(
        image_path=input_image,
        renditions=renditions,
        extension="jpg",
        keep_fullsized=True
    )
    
    print(f"Processed {len(result)} renditions")
    for rendition in result:
        print(f"  - {rendition.width}x{rendition.height}: {rendition.crop_dimensions}")