import argparse
import os
from os import path
from os.path import join

import tornado.ioloop
import tornado.web

class MyStaticFileHandler(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path):
        # Disable cache
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')

class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, html_dir):
        super().initialize()
        # print('initialize html_dir', html_dir)
        self.html_dir = html_dir

    def get(self, *args, **kwargs):
        print('args', args)
        print('kwargs', kwargs)
        self.set_header('Content-type', 'text/html')
        self.set_status(200)
        self.set_header('Server', '')
        self.write('<html><body>')
        for file in sorted(os.listdir(self.html_dir)):
            if path.isfile(join(self.html_dir, file)):
                self.write(f'<a href="{file}">{file}</a><br />')
        self.write('</body></html>')
        self.flush()
        self.finish()


def run(html_dir, port):
    app = tornado.web.Application([
        (r"/index", IndexHandler, {'html_dir': html_dir}),
        (r"/(.*)", MyStaticFileHandler, {"path": html_dir}),
        ],
        # static_path=html_dir
    )
    app.listen(args.port)
    print(f'serving directory "{html_dir}" on port {port}')
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--html-dir', type=str, default='html')
    args = parser.parse_args()
    run(**args.__dict__)
