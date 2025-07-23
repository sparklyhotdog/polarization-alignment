# -*- coding: utf-8 -*-
"""
Script created to communicate with the NuCrypt Polarization Analyzer 1000 (PA 1000);

whenever one wants to restart the server, remember to also restart the corresponding python kernel (ctl+.) to avoid blockage of COM port.

Currently implemented commands:
    "*idn?"        : queries the identity of the connected polarization analyzer;
    "retardances?" : queries the current retardances for the 6 waveplates;
    "retard [0-5] [0-420]" : sets the retardance to [0-420](degrees) on waveplate [0-5];
    "h"("v/d/a/r/l"): sets the analyzer to one of the tomography basis;
    "zeroall" : setting the retardances to zero on all 6 waveplates
"""
import socketserver
import socket
import serial
import threading
import time

'''
pol_analyzer class for communicating with PA devices; 
currently only compatible with PA 1000 (PA 2000 requires some modification)
'''

class polAnalyzer():
    
    def __init__(self, com_port='COM9'):        
        try:
            self.port = serial.Serial(port=com_port, baudrate=115200, bytesize=8, parity='N', rtscts=False, timeout=0.5)
            self.port.rts = True
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()
            
        except serial.SerialException:
            self.port = serial.Serial(port=com_port, baudrate=115200, bytesize=8, parity='N', rtscts=False, timeout=0.5)
            self.port.close()
            print("serial port error, port occupied?")

    def portStatus(self):
        status = (self.port.name, "open?:", self.port.is_open, self.port.baudrate,'in_waiting',self.port.in_waiting, 'out_waiting',self.port.out_waiting)
        return status

    def displayMenu(self):
        """always return to the top menu before executing something else;
        basically keep entering 'e' (exit) until the words 'PA Main Menu' appear"""
        self.port.write('i\r'.encode())
        self.port.write('e\r'.encode())
        ret=self.port.read_until('device info').decode()
        self.port.reset_input_buffer()
        
        while not ('PA Main Menu:' in ret):
            self.port.write('e\r'.encode())
            ret=self.port.read_until('>').decode()
            self.port.reset_input_buffer()
            print("back to upper level menu")
        return ret

    def displayPAInfo(self):
        """read device info ('i'); search key-word "PA 1000" or "PA 2000"""
        self.displayMenu()
        self.port.write('i\r'.encode())
        ret=self.port.read_until('>').decode()
        self.port.reset_input_buffer()
        for i in ret.split("\n"):
            if ("PA 1000" in i):
                ret = i.strip()
                break
            else:
                ret="PA info not available..."
        time.sleep(0.05)
        return ret

    def zeroAll(self):
        """Set all retarders to zero"""
        self.displayMenu()
        self.port.write('2\r'.encode())
        return 0

    def setRetardances(self, channel=0, retardance=0.00):
        """Set the PA measurement basis to one of the six tomography basis: H, V, D, A, R, L"""
        self.port.write('h'.encode())
        self.port.write((str(channel)+"\r").encode())
        self.port.write((str(retardance)+"\r").encode())
        return 0

    def setTomo(self, basis='H'):
        """Set the PA measurement basis to one of the six tomography basis: H, V, D, A, R, L"""
        self.port.write('c'.encode())

        if basis == 'H':
            self.port.write('H'.encode())
        elif basis == 'V':
            self.port.write('V'.encode())
        elif basis == 'D':
            self.port.write('D'.encode())
        elif basis == 'A':
            self.port.write('A'.encode())
        elif basis == 'R':
            self.port.write('R'.encode())
        elif basis == 'L':
            self.port.write('L'.encode())
        else:
            print('Invalid basis choice...')
        return 0

    def displayRetardances(self):
        """Display the current waveplate retardances in degrees (all 6 WPs)"""
        #self.displayMenu()
        self.port.write('p'.encode())
        retardance=[]
        ret=self.port.read_until('>').decode()
        for i in ret.split("\n"):
            if   ("WP0" in i):
                retardance.append(i.split()[2])
            elif ("WP1" in i):
                retardance.append(i.split()[2])
            elif ("WP2" in i):
                retardance.append(i.split()[2])
            elif ("WP3" in i):
                retardance.append(i.split()[2])
            elif ("WP4" in i):
                retardance.append(i.split()[2])
            elif ("WP5" in i):
                retardance.append(i.split()[2])
        return retardance

    def close(self):
        """Close COM port"""
        self.port.close()
        return 0


# define simpleserver class, which essentially maps a serial COM port to a local IP address
class SimpleServer(socketserver.ThreadingMixIn,socketserver.TCPServer):
    # Ctrl-C will cleanly kill all spawned threads
    daemon_threads = True
    # much faster rebinding
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass):
        socketserver.TCPServer.__init__(self, server_address, RequestHandlerClass)
        print('server established')


class RequestHandler(socketserver.StreamRequestHandler):
    global polAnalyzer
    _lock = threading.Lock()
    def handle(self):
        print('Connection from: ', self.client_address[0])
        
        while True:
            self.msgin = self.rfile.readline().strip().decode()
            print("{} wrote:".format(self.client_address[0]))
            print(self.msgin+'\n')
            cmd = self.msgin.split()
            msgout = ''
            
            '''
            list of commands starts here!!!
            '''
            if len(cmd)<1:
                print("No command")
            
            # print analyzer device info
            elif cmd[0].lower() == '*idn?':
                msgout = pa.displayPAInfo()
            
            # print current retardance angles
            elif cmd[0].lower() == 'retardances?':
                msgout = pa.displayRetardances()
            
            elif cmd[0].lower() == 'retard':
                pa.setRetardances(channel=cmd[1], retardance=cmd[2])
                msgout = pa.displayRetardances()

            elif cmd[0].lower() == 'h':
                pa.setTomo(basis='H')
                msgout = pa.displayRetardances()
                
            elif cmd[0].lower() == 'v':
                pa.setTomo(basis='V')
                msgout = pa.displayRetardances()
            
            elif cmd[0].lower() == 'd':
                pa.setTomo(basis='D')
                msgout = pa.displayRetardances()
            
            elif cmd[0].lower() == 'a':
                pa.setTomo(basis='A')
                msgout = pa.displayRetardances()
            
            elif cmd[0].lower() == 'r':
                pa.setTomo(basis='R')
                msgout = pa.displayRetardances()
                
            elif cmd[0].lower() == 'l':
                pa.setTomo(basis='L')
                msgout = pa.displayRetardances()
                
            elif cmd[0].lower() == 'zeroall':
                pa.zeroAll()
                msgout = pa.displayRetardances()
                
            else:
                msgout = "Invalid Command"
            print((cmd, msgout))
            if msgout is None:
                msgout = ''
            msgout = str(msgout)
            self.wfile.write(msgout.encode())
            self.wfile.write('ack\n'.encode())


pa = polAnalyzer(com_port='COM7')
#pa = polAnalyzer(com_port='COM9')


if __name__ == '__main__':
    
    
    import threading
    
    #still need to check this, but I think for local operation everything between 127.0.0.0 to 127.255.255.255 should be fine
    #address = ('129.6.224.99', 56000)#Analyzer
    #address = ('10.10.101.21', 55000)
    address = ('127.0.0.1', 55000)
    server = SimpleServer(address, RequestHandler)
    ip, port = server.server_address # find out what port we were given
    print(ip, port)
    
    #start the server; little trick, in spyder you can clear out the server by restarting the kernel...
    server.serve_forever()

    '''
    pol=polAnalyzer(com_port='COM10')
    
    print(pol.displayMenu())
    print("blu")
    for i in range(5):
        pol.setRetardances(channel=i, retardance=80)
        
    print("bla")
    print(pol.displayRetardances())
    pol.close()
    
    
    #pol.close()
    '''