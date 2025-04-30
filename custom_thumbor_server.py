import os
import sys
from thumbor.app import ThumborServiceApp
from thumbor.config import Config
from thumbor.context import Context, ServerParameters
from thumbor.server import get_application
from tornado.web import RequestHandler
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from thumbor.importer import Importer
from favicon_handler import FaviconHandler
# Custom handler for favicon


# Main function
def main():
    config_path = os.path.abspath('./thumbor.conf')
    config = Config.load(config_path)
    config.AUTO_WEBP = True  # Optional: Customize config
    config.LOADER = 'thumbor.loaders.http_loader'  # Ensure a loader is set

    
    server_parameters = ServerParameters(
        port=8888,
        ip='0.0.0.0',
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
    application.add_handlers(r'.*', [(r'/favicon.ico', FaviconHandler)])

    # Start the server
    server = HTTPServer(application)
    server.listen(server_parameters.port, server_parameters.ip)
    print(f"Thumbor server running at http://{server_parameters.ip}:{server_parameters.port}")
    IOLoop.instance().start()

if __name__ == "__main__":
    main()