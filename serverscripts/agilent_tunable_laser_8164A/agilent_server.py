import socketserver
import socket
import pyvisa
import threading
import time
from tunable_laser import HP8164A


# define simpleserver class, which essentially maps a serial COM port to a local IP address
class SimpleServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    # Ctrl-C will cleanly kill all spawned threads
    daemon_threads = True
    # much faster rebinding
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass):
        socketserver.TCPServer.__init__(self, server_address, RequestHandlerClass)
        print('server established')


class RequestHandler(socketserver.StreamRequestHandler):
    global HP8164A
    _lock = threading.Lock()

    def handle(self):
        print('Connection from: ', self.client_address[0])

        while True:
            self.msgin = self.rfile.readline().strip().decode()
            print("{} wrote:".format(self.client_address[0]))
            print(self.msgin + '\n')
            cmd = self.msgin.split()
            msgout = ''

            '''
            list of commands starts here!!!
            '''
            if len(cmd) < 1:
                print("No command")

            # print device info
            elif cmd[0].lower() == '*idn?':
                msgout = laser.idn()

            # print wavelength
            elif cmd[0].lower() == 'wavelength?':
                msgout = laser.getWavelength()

            # set wavelength
            elif cmd[0].lower() == 'wavelength':
                if not 1.525 <= cmd[1] <= 1.725:
                    msgout = "Invalid wavelength. Should be between 1.525 and 1.725 (micrometers)"
                else:
                    laser.setWavelength(cmd[1])
                    msgout = laser.getWavelength()

            else:
                msgout = "Invalid Command"

            print((cmd, msgout))
            if msgout == None:
                msgout = ''
            msgout = str(msgout)
            self.wfile.write(msgout.encode())
            self.wfile.write('ack\n'.encode())


laser = HP8164A()


if __name__ == '__main__':

    address = ('169.254.142.114', 55000)
    server = SimpleServer(address, RequestHandler)
    ip, port = server.server_address  # find out what port we were given
    print(ip, port)

    # start the server; little trick, in spyder you can clear out the server by restarting the kernel...
    server.serve_forever()
