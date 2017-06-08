from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

class HTTPHandler(BaseHTTPRequestHandler):
    gesture = 'notReady'
    def do_GET(self):
        fo = open("gesture", "r")
        gesture = fo.read(10)
        fo.close()

        print self.path

        self.protocal_version = 'HTTP/1.1'
        self.send_response(200)
        self.send_header("Welcome", "Contect")
        self.end_headers()
        self.wfile.write(gesture)


def start_server(port):
    http_server = HTTPServer(('', int(port)), HTTPHandler)
    http_server.serve_forever()

start_server(8000)