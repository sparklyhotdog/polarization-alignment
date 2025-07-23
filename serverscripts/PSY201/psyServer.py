# -*- coding: utf-8 -*-
"""
Script created to communicate with the General Photonics PSY-201 Polarization Synthesizer

whenever one wants to restart the server, remember to also restart the corresponding python kernel (ctl+.) to avoid blockage of COM port.

Currently implemented commands:
    "*idn?"                 : queries the identity of the connected polarization synthesizer;
    "sop?"                  : queries the current measured polarization state;
    "sop [s1] [s2] [s3]"    : sets the polarization state with stokes parameters (normalized before tracking);
    "angle [theta] [phi]"   : sets the polarization state with polar coordinates (in degrees);
    "h"                     : sets the polarization state to H
    "v"                     : sets the polarization state to V
    "d"                     : sets the polarization state to D
    "a"                     : sets the polarization state to A
    "r"                     : sets the polarization state to R
    "l"                     : sets the polarization state to L
"""
import socketserver
import socket
import serial
import threading
import time

'''
PSY201 class for communicating with the polarization synthesizer
'''


class PSY201:

    def __init__(self, com_port='COM7'):
        try:
            self.port = serial.Serial(port=com_port, baudrate=9600, bytesize=8, parity='N', rtscts=False, timeout=1)
            self.port.rts = True
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()

        except serial.SerialException:
            self.port = serial.Serial(port=com_port, baudrate=9600, bytesize=8, parity='N', rtscts=False, timeout=1)
            self.port.close()
            print("serial port error, port occupied?")

    def portStatus(self):
        status = (self.port.name, "open?:", self.port.is_open, self.port.baudrate, 'in_waiting', self.port.in_waiting,
                  'out_waiting', self.port.out_waiting)
        return status


    '''
    read device info ('i'); 
    '''

    def displayInfo(self):
        self.port.write(b"*IDN?\r\n")
        ret = self.port.readline()
        return ret

    '''
    Returns the measured SOP (Stokes parameters) S1, S2, S3.
    '''

    def displaySOP(self):
        self.port.write(b":MEASure:SOP?\r\n")
        ret = self.port.readline()
        return ret

    '''
    Set the polarization state with stokes parameters (normalized before tracking)
    '''

    def setSOP(self, s1, s2, s3):
        # message = ":CONTrol:SOP " + str(round(s1, 4)) + ", " + str(round(s2, 4)) + ", " + str(round(s3, 4)) + "\r\n"
        message = ":CONTrol:SOP " + s1 + "," + s2 + "," + s3 + "\r\n"
        print(message)
        self.port.write(message.encode('utf-8'))
        return 0

    '''
    Set the polarization state with angles Set SOP (in spherical coordinates) to be tracked.
    Ranges: f1=theta: 0 to 360
            f2=phi: 0 to 180
    Unit: Degrees.
    '''

    def setAngles(self, theta, phi):
        message = ":CONTrol:ANGLe " + theta + "," + phi + "\r\n"
        print(message)
        self.port.write(message.encode('utf-8'))
        return 0

    '''
    close COM port
    '''

    def close(self):
        self.port.close()
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
    global PSY201
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
                msgout = psy.displayInfo()

            # print current SOP
            elif cmd[0].lower() == 'sop?':
                msgout = psy.displaySOP()

            # set SOP (in stokes parameters)
            elif cmd[0].lower() == 'sop':
                psy.setSOP(cmd[1], cmd[2], cmd[3])
                msgout = psy.displaySOP()

            # set SOP (in polar coordinates)
            elif cmd[0].lower() == 'angle':
                psy.setAngles(cmd[1], cmd[2])
                msgout = psy.displaySOP()

            # set SOP to H (1, 0, 0)
            elif cmd[0].lower() == 'h':
                psy.setSOP("1", "0", "0")
                msgout = psy.displaySOP()

            # set SOP to V (-1, 0, 0)
            elif cmd[0].lower() == 'h':
                psy.setSOP("-1", "0", "0")
                msgout = psy.displaySOP()

            # set SOP to D (0, 1, 0)
            elif cmd[0].lower() == 'd':
                psy.setSOP("0", "1", "0")
                msgout = psy.displaySOP()

            # set SOP to A (0, -1, 0)
            elif cmd[0].lower() == 'a':
                psy.setSOP("0", "-1", "0")
                msgout = psy.displaySOP()

            # set SOP to R (0, 0, 1)
            elif cmd[0].lower() == 'r':
                psy.setSOP("0", "0", "1")
                msgout = psy.displaySOP()

            # set SOP to L (0, 0, -1)
            elif cmd[0].lower() == 'l':
                psy.setSOP("0", "0", "-1")
                msgout = psy.displaySOP()

            else:
                msgout = "Invalid Command"
            print((cmd, msgout))
            if msgout == None:
                msgout = ''
            msgout = str(msgout)
            self.wfile.write(msgout.encode())
            self.wfile.write('ack\n'.encode())


psy = PSY201(com_port='COM7')
# pa = polAnalyzer(com_port='COM9')


if __name__ == '__main__':
    import threading

    address = ('10.10.101.20', 56010)
    server = SimpleServer(address, RequestHandler)
    ip, port = server.server_address  # find out what port we were given
    print(ip, port)

    # start the server; little trick, in spyder you can clear out the server by restarting the kernel...
    server.serve_forever()

