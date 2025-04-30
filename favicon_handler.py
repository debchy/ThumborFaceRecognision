import os
import tornado.web

# class FaviconHandler(tornado.web.RequestHandler):
#     def get(self):
#         # Path to your favicon file
#         favicon_path = os.path.join(self.application.loader.root_path, 'favicon.ico')
#         print(f'favicon_path - {favicon_path}')
#         if os.path.exists(favicon_path):
#             with open(favicon_path, 'rb') as f:
#                 self.set_header('Content-Type', 'image/x-icon')
#                 self.write(f.read())
#         else:
#             self.set_status(404)
#             self.write('Favicon not found')

class FaviconHandler(tornado.web.RequestHandler):
    def get(self):
        favicon_path = os.path.join('thumbor_images', 'favicon.ico')
        if os.path.exists(favicon_path):
            with open(favicon_path, 'rb') as f:
                self.set_header('Content-Type', 'image/x-icon')
                self.write(f.read())
        else:
            self.set_status(404)
            self.write('Favicon not found')