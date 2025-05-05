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
#from favicon_handler import FaviconHandler # Custom handler for favicon
from web_handlers import MainHandler, UploadHandler, ListImagesHandler, ImageHandler


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./log/thumbor_app.log')
    ]
)

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

# Main function
def main():
    # Save the interface file
    save_interface_file()

    config_path = os.path.abspath('./thumbor.conf')
    config = Config.load(config_path)
    config.AUTO_WEBP = True  # Optional: Customize config
    config.LOADER = 'thumbor.loaders.file_loader_http_fallback' #thumbor.loaders.http_loader'  # Ensure a loader is set
    #config.LOADER = 'thumbor.loaders.file_loader'
    config.STORAGE = 'thumbor.storages.file_storage'
    config.FILE_LOADER_ROOT_PATH = os.path.abspath('.')
    config.ALLOW_UNSAFE_URL = True
    
    
    server_parameters = ServerParameters(
        port=config.HTTP_PORT,
        ip=config.HTTP_HOST,
        config_path=config_path,
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
    #application.add_handlers(r'.*', [(r'/favicon.ico', FaviconHandler)])
    application.add_handlers(r'.*', [
        (r'/', MainHandler),
        (r'/upload', UploadHandler),
        (r'/list_images', ListImagesHandler),
        (r'/images/(.*)', ImageHandler),
        #(r'/favicon.ico', FaviconHandler),
        # Serve static files (like uploaded images) for thumbor to access
        (r'/uploads/(.*)', StaticFileHandler, {'path': UPLOAD_DIR}),
    ])

    # Start the server
    server = HTTPServer(application)

    # Enable multiple processes for concurrent request handling
    server.listen(server_parameters.port, server_parameters.ip)

    print(f"Thumbor server running at http://{server_parameters.ip}:{server_parameters.port}")
    
    # Start the IO loop
    IOLoop.instance().start()

if __name__ == "__main__":
    main()