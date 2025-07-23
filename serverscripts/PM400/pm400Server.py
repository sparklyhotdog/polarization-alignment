# -*- coding: utf-8 -*-
"""
Script created to communicate with the ThorLabs PM400 Optical Power Meter

whenever one wants to restart the server, remember to also restart the corresponding python kernel (ctl+.) to avoid blockage of COM port.
"""
import socketserver
import socket
import pyvisa
import threading
import time

'''
PM400 class for communicating with the optical power meter
'''


class PM400:

    def __init__(self, name='USB0::0x1313::0x8075::P5006483::INSTR'):
        rm = pyvisa.ResourceManager()
        deviceList = rm.list_resources()
        print(deviceList)
        FoundDevice = False
        # looking through the device list
        for i in deviceList:
            # print(i)
            if name in i:
                device = i
                print("Power Meter found, ID:", device)
                FoundDevice = True

        if FoundDevice:
            try:
                self.powermeter = rm.open_resource(device)
                self.powermeter.timeout = 1000
                self.powermeter.read_termination = '\n'
                self.powermeter.write("SENS:POW:UNIT W")
                print("Power Meter connected.")
                print("Power units set to W.")

            except OSError:
                print('Cannot open device.')
        else:
            print("Device not found.")
            sys.exit()

    def displayInfo(self):
        """"" Display manufacturer, model code, serial number, and firmware revision levels of the power meter"""
        try:
            out = self.powermeter.query("*IDN?", delay=0.05)
        except:
            out = 'Info unavailable.\r\n'
        return out.strip('\r\n')

    def getPower(self):
        """Query the power measurement"""
        try:
            out = self.powermeter.query(":MEAS:POW?", delay=0.05)
        except:
            out = 'Power unavailable.\r\n'
        return out.strip('\r\n')

    def close(self):
        self.powermeter.close()
        return 0


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
    global PM400
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
                msgout = pm.displayInfo()

            # print power measurement
            elif cmd[0].lower() == 'pow?':
                msgout = pm.getPower()

            else:
                msgout = "Invalid Command"
            print((cmd, msgout))
            if msgout == None:
                msgout = ''
            msgout = str(msgout)
            self.wfile.write(msgout.encode())
            self.wfile.write('ack\n'.encode())


pm = PM400(name='USB0::0x1313::0x8075::P5006483::INSTR')


if __name__ == '__main__':
    import threading

    # still need to check this, but I think for local operation everything between 127.0.0.0 to 127.255.255.255 should be fine
    # address = ('129.6.224.99', 56000)#Analyzer
    # address = ('10.10.101.'21', 55000)
    address = ('127.0.0.3', 55000)
    server = SimpleServer(address, RequestHandler)
    ip, port = server.server_address  # find out what port we were given
    print(ip, port)

    # start the server; little trick, in spyder you can clear out the server by restarting the kernel...
    server.serve_forever()

