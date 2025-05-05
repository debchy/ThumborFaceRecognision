import os
import sys
import logging
from thumbor.app import ThumborServiceApp
from thumbor.config import Config
from thumbor.context import Context, ServerParameters
from thumbor.server import get_application
from tornado.web import RequestHandler, StaticFileHandler
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from thumbor.importer import Importer
from favicon_handler import FaviconHandler # Custom handler for favicon
from web_handlers import MainHandler, UploadHandler, ListImagesHandler, ImageHandler
from direct_smart_crop import SmartCropper  # Import our direct smart cropper
from custom_thumbor_integration import ProcessImageHandler, BatchProcessHandler  # Import our custom handlers

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('thumbor_app.log')
    ]
)
logger = logging.getLogger('thumbor_server')

# Ensure directories exist
UPLOAD_DIR = "thumbor_images/uploads"
OUTPUT_DIR = "thumbor_images/outputs/thumbor"
STATIC_DIR = "thumbor_images"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# Save the HTML interface to the static directory
def save_interface_file():
    interface_path = os.path.join(STATIC_DIR, "upload_interface.html")
    
    # Check if we need to create the interface file
    if not os.path.exists(interface_path):
        # In a real implementation, we'd have this file in the repo
        # For this example, we'll create it with placeholder content
        with open(interface_path, "w") as f:
            f.write("<!-- This file will be replaced by the actual interface -->")

# Use a default config if the config file has errors
def create_default_config():
    config = Config()
    # Set required configuration
    config.AUTO_WEBP = True
    config.LOADER = 'thumbor.loaders.file_loader_http_fallback'
    config.STORAGE = 'thumbor.storages.file_storage'
    config.FILE_LOADER_ROOT_PATH = os.path.abspath('.')
    config.ALLOW_UNSAFE_URL = True
    config.MAX_AGE = 60 * 60 * 24  # 1 day cache
    config.QUALITY = 85  # Good balance of quality vs size
    
    # Must have options that might be missing
    config.SECURITY_KEY = 'THUMBOR_SECURITY_KEY'
    config.STORAGE_EXPIRATION_SECONDS = 60 * 60 * 24 * 30  # 30 days
    config.MAX_WIDTH = 0
    config.MAX_HEIGHT = 0
    config.RESULT_STORAGE_EXPIRATION_SECONDS = 0
    config.ENGINE = 'thumbor.engines.pil'  # Default image engine
    
    return config

# Main function
def main():
    # Save the interface file
    save_interface_file()

    config_path = os.path.abspath('./thumbor.conf')
    
    # Try to load config, but use default if there's an error
    try:
        # First try to load from file
        config = Config.load(config_path)
        logger.info("Successfully loaded configuration from thumbor.conf")
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        logger.info("Using default configuration instead")
        config = create_default_config()
    
    # Ensure critical config options are set
    if not hasattr(config, 'LOADER') or not config.LOADER:
        config.LOADER = 'thumbor.loaders.file_loader_http_fallback'
    if not hasattr(config, 'STORAGE') or not config.STORAGE:
        config.STORAGE = 'thumbor.storages.file_storage'
    if not hasattr(config, 'FILE_LOADER_ROOT_PATH') or not config.FILE_LOADER_ROOT_PATH:
        config.FILE_LOADER_ROOT_PATH = os.path.abspath('.')
    if not hasattr(config, 'ALLOW_UNSAFE_URL'):
        config.ALLOW_UNSAFE_URL = True
    
    ## Create the smart cropper instance
    #smart_cropper = SmartCropper(config_path if os.path.exists(config_path) else None)
    
    server_parameters = ServerParameters(
        port=8888,
        ip='127.0.0.1',
        config_path=config_path if os.path.exists(config_path) else None,
        log_level='debug',
        app_class= 'custom_thumbor_server.ThumborServiceApp',
        fd=None,
        keyfile=None
    )

    importer = Importer(config)
    importer.import_modules()

    context = Context(server=server_parameters, config=config, importer=importer)

    # Get the app and add your custom handler
    application = get_application(context)
    
    # Add our custom handlers
    application.add_handlers(r'.*', [
        (r'/', MainHandler),
        (r'/upload', UploadHandler),
        (r'/list_images', ListImagesHandler),
        (r'/images/(.*)', ImageHandler),
        (r'/favicon.ico', FaviconHandler),
        # Add our direct processing handlers
        (r'/process', ProcessImageHandler),
        (r'/batch_process', BatchProcessHandler),
        # Serve static files (like uploaded images) for thumbor to access
        (r'/uploads/(.*)', StaticFileHandler, {'path': UPLOAD_DIR}),
        (r'/outputs/(.*)', StaticFileHandler, {'path': OUTPUT_DIR}),
    ])

    # Create HTTP server with higher max_buffer_size for larger images
    server = HTTPServer(
        application,
        max_buffer_size=10485760,  # 10MB buffer size
        xheaders=True
    )
    
    # Listen on the specified port
    server.listen(server_parameters.port, server_parameters.ip)
    
    print(f"Enhanced Thumbor server running at http://{server_parameters.ip}:{server_parameters.port}")
    print("Direct image processing endpoints available at:")
    print(f"  - http://{server_parameters.ip}:{server_parameters.port}/process")
    print(f"  - http://{server_parameters.ip}:{server_parameters.port}/batch_process")
    
    # Start the IO loop
    IOLoop.current().start()

if __name__ == "__main__":
    main()