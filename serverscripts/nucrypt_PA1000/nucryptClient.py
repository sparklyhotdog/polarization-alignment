# -*- coding: utf-8 -*-
"""
Client script for communicating to a remote nucrypt server.

Currently implemented commands:
    "*idn?"        : querys the identity of the connected polarization analyzer;
    "retardances?" : querys the current retardances for the 6 waveplates;
    "retard [0-5] [0-420]" : sets the retardance to [0-420](degrees) on waveplate [0-5];
    "h"("v/d/a/r/l"): sets the analyzer to one of the tomography basis;
    "zeroall" : setting the retardances to zero on all 6 waveplates

@author: yns19
"""

import socket
import select
import time
# import TimeTagger
import numpy as np
import matplotlib


class nucryptClient():
    
    def __init__(self, ip='10.10.101.21', port=55000, timeout=3):
        self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect((ip, port))

    def _send(self, msg):
        self.s.send((msg + "\n").encode())
        msgResp = self.msgResponse()
        return(msgResp)
        
    def msgResponse(self):
        inputs = [self.s]
        outputs = []
        response = ''
        while inputs:
            readable, writable, exceptional = select.select(inputs, outputs, inputs)
            for s in readable:
                response += s.recv(1024).decode()
                #print 'respone:',repr(response)
            if len(response)==0:
                break
            if response.lower().endswith('ack\n'):
                break
        return response[:-4]
    
    def disconnect(self):
        self.s.close()
        print("disconnected")
        

if __name__ == '__main__':

    NISThost = '169.254.142.114'
    NISTport=56000

    mc = nucryptClient(NISThost, NISTport)
    mc._send('retard 3 360')
    bla = mc._send('retardances?')
    print(bla)
    #mc.disconnect()
