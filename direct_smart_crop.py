import os
import io
import asyncio
import logging
from PIL import Image
from thumbor.engines.pil import Engine as PILEngine
from thumbor.point import FocalPoint
from thumbor.detectors.feature_detector import Detector as FeatureDetector
from thumbor.detectors.face_detector import Detector as FaceDetector
from thumbor.context import Context, ServerParameters
from thumbor.config import Config
from thumbor.importer import Importer
from thumbor.utils import logger
from opencv_engine import Engine as OpenCVEngine


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
                ]
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

    async def _get_focal_points_async(self, engine, image_path):
        """Get focal points for smart cropping"""
        focal_points = []
        isVVIP = False
        
        # Setup mock request
        mock_request = MockRequest(image_path)
        self.context.request = mock_request
        
        # Also set the engine in the request
        mock_request.engine = engine
        mock_request.image = engine
        
        # Check if we have VVIP faces from OpenCV engine
        if hasattr(engine, 'vvip_faces') and engine.vvip_faces:
            for (x, y, w, h, name, confidence) in engine.vvip_faces:
                # Give VVIP faces a very high weight to ensure they're prioritized
                weight = 100  # High weight to ensure VVIP faces are prioritized
                fp = FocalPoint(
                    x + w/2,  # Center X of face
                    y + h/2,  # Center Y of face
                    weight
                )
                focal_points.append(fp)
                isVVIP = True
                logger.info(f"Using VVIP face as focal point: {name} at ({x}, {y})")

        # If no VVIP faces or want to try standard face detection anyway
        if not isVVIP or not focal_points:
            try:
                self.context.modules.engine = engine
                face_detector = FaceDetector(self.context, 0, self.config.DETECTORS)
                face_detector.detector_name = 'face_detector'
                fp = await face_detector.detect()
                if fp:
                    focal_points.extend(fp)
                    logger.info(f"Found {len(fp)} faces for focal points")
            except Exception as e:
                logger.warning(f"Face detection failed: {str(e)}")
        
        # If no faces found or want to try feature detection anyway
        if not isVVIP or not focal_points:
            try:
                self.context.modules.engine = engine
                feature_detector = FeatureDetector(self.context, 1, self.config.DETECTORS)
                feature_detector.detector_name = 'feature_detector'
                fp = await feature_detector.detect()
                if fp:
                    focal_points.extend(fp)
                    logger.info(f"Found {len(fp)} features for focal points")
            except Exception as e:
                logger.warning(f"Feature detection failed: {str(e)}")
        
        # If still no focal points, use center of image
        if not focal_points:
            logger.info("No faces or features detected, using center of image")
            width, height = engine.size
            focal_points = [FocalPoint(width / 2, height / 2, 1)]
        
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

    async def smart_crop(self, image_path, width, height, extension=None):
        """
        Perform smart cropping on an image
        
        Args:
            image_path: Path to the input image file
            width: Target width
            height: Target height
            extension: Output format extension (None for same as input)
            
        Returns:
            bytes: The processed image as bytes
        """
        logger.info(f"Smart cropping image {image_path} to {width}x{height}")
        
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
        
        # Get focal points for smart cropping
        focal_points = await self._get_focal_points_async(engine, image_path)
        
        # Calculate crop dimensions based on focal points
        crop_dimensions = self._calculate_crop_dimensions(engine, width, height, focal_points)
        
        # Log the crop dimensions for debugging
        logger.info(f"Crop dimensions: {crop_dimensions}")
        
        # Crop the image
        engine.crop(*crop_dimensions)
        
        # Resize to final dimensions
        engine.resize(width, height)
        
        # Convert to specified format if needed
        img_buffer = engine.read(extension)            
        
        return img_buffer

    async def save_smart_cropped_image(self, image_path, width, height, output_path, extension=None):
        """
        Perform smart cropping and save the result
        
        Args:
            image_path: Path to the input image file
            width: Target width
            height: Target height
            output_path: Path to save the output image
            extension: Output format extension (None for same as input)
            
        Returns:
            str: Path to the saved image
        """
        img_buffer = await self.smart_crop(image_path, width, height, extension)
        
        # Make sure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the processed image
        with open(output_path, 'wb') as f:
            f.write(img_buffer)
        
        logger.info(f"Saved smart cropped image to {output_path}")
        return output_path

    def smart_crop_sync(self, image_path, width, height, output_path=None, extension=None):
        """
        Synchronous wrapper for the async smart_crop method
        
        Args:
            image_path: Path to the input image file
            width: Target width
            height: Target height
            output_path: Path to save the output image (optional)
            extension: Output format extension (None for same as input)
            
        Returns:
            bytes or str: The processed image as bytes or the path to the saved image
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if output_path:
            return loop.run_until_complete(
                self.save_smart_cropped_image(image_path, width, height, output_path, extension)
            )
        else:
            return loop.run_until_complete(
                self.smart_crop(image_path, width, height, extension)
            )


# Example usage
if __name__ == "__main__":
    # Create a smart cropper instance
    cropper = SmartCropper("./thumbor.conf")  # Use your thumbor config if available
    
    # Example image processing - replace with your actual image path
    input_image = "thumbor_images/uploads/294202522024320_1746198457.jpeg"
    output_image = "thumbor_images/outputs/smart_cropped.jpg"
    
    # Process the image - use the synchronous method for direct scripts
    result = cropper.smart_crop_sync(
        image_path=input_image,
        width=1920,
        height=1080,
        output_path=output_image,
        extension="jpg"
    )
    
    print(f"Image processed and saved to {result}")