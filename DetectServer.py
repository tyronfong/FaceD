import cgi
import logging
from HandReco import HandReco
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTTPHandler(BaseHTTPRequestHandler):
    gesture = "nothing"
    handReco = HandReco()
    handReco.start()
    logger.info('Hand Recoder started')

    def do_GET(self):
        self.protocal_version = 'HTTP/1.1'
        self.send_response(200)
        self.send_header("Welcome", "Contect")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(self.handReco.currentGesture)

    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers.getheader('content-length'))
            postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
        else:
            postvars = {}

        self.gesture = postvars.get('gesture', 'nothing')[0]

        print postvars.get('gesture', 'nothing')[0]

        self.protocal_version = 'HTTP/1.1'
        self.send_response(200)
        self.send_header("Welcome", "Contect")
        self.end_headers()
        self.wfile.write('OK')


def start_server(port):
    http_server = HTTPServer(('', int(port)), HTTPHandler)
    http_server.serve_forever()

start_server(8000)