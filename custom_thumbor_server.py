import os
import sys
import logging
from thumbor.config import Config
from thumbor.context import Context, ServerParameters
from thumbor.server import get_application
from tornado.web import StaticFileHandler
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from thumbor.importer import Importer
from favicon_handler import FaviconHandler # Custom handler for favicon
from web_handlers import MainHandler, UploadHandler, ListImagesHandler, ImageHandler
from api_handlers import ProcessImageHandler  # Import our custom handlers



# Ensure directories exist
UPLOAD_DIR = "thumbor_images/uploads"
OUTPUT_DIR = "thumbor_images/outputs/thumbor"
STATIC_DIR = "thumbor_images"
LOG_DIR = "log"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'thumbor_app.log'))
    ]
)
logger = logging.getLogger('thumbor_server')

# Save the HTML interface to the static directory
def save_interface_file():
    interface_path = os.path.join(STATIC_DIR, "upload_interface.html")
    
    # Check if we need to create the interface file
    if not os.path.exists(interface_path):
        # In a real implementation, we'd have this file in the repo
        # For this example, we'll create it with placeholder content
        with open(interface_path, "w") as f:
            f.write("<!-- This file will be replaced by the actual interface -->")

# Main function
def main():
    # Save the interface file
    save_interface_file()

    config_path = os.path.abspath('./thumbor.conf')
    
    # load config
    config = Config.load(config_path)
    
    # Ensure critical config options are set
    if not hasattr(config, 'LOADER') or not config.LOADER:
        config.LOADER = 'thumbor.loaders.file_loader_http_fallback'
    if not hasattr(config, 'STORAGE') or not config.STORAGE:
        config.STORAGE = 'thumbor.storages.file_storage'
    if not hasattr(config, 'FILE_LOADER_ROOT_PATH') or not config.FILE_LOADER_ROOT_PATH:
        config.FILE_LOADER_ROOT_PATH = os.path.abspath('.')
    if not hasattr(config, 'ALLOW_UNSAFE_URL'):
        config.ALLOW_UNSAFE_URL = True
    
    
    server_parameters = ServerParameters(
        port        = config.HTTP_PORT,
        ip          = config.HTTP_HOST,
        config_path = config_path if os.path.exists(config_path) else None,
        log_level   = 'debug',
        app_class   = 'thumbor.app.ThumborServiceApp',
        fd          = None,
        keyfile     = None
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
        # Serve static files (like uploaded images) for thumbor to access
        (r'/uploads/(.*)', StaticFileHandler, {'path': UPLOAD_DIR}),
        (r'/outputs/(.*)', StaticFileHandler, {'path': OUTPUT_DIR}),
        # Add our direct processing handlers
        (r'/api/smart_crop', ProcessImageHandler),
    ])

    # Create HTTP server with higher max_buffer_size for larger images
    server = HTTPServer(
        application,
        max_buffer_size=10485760,  # 10MB buffer size
        xheaders=True
    )
    
    # Listen on the specified port
    server.listen(server_parameters.port, server_parameters.ip)
    
    logger.info(f"Thumbor server running at http://{server_parameters.ip}:{server_parameters.port}")
    logger.info(f"Direct Smart Crop endpoint: http://{server_parameters.ip}:{server_parameters.port}/api/smart_crop")
    
    # Start the IO loop
    IOLoop.current().start()

if __name__ == "__main__":
    main()