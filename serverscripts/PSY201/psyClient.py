# -*- coding: utf-8 -*-
"""
Client script for communicating to a remote psy server.

Currently implemented commands:
    "*idn?"                 : queries the identity of the connected polarization synthesizer;
    "sop?"                  : queries the current measured polarization state;
    "sop [s1] [s2] [s3]"    : sets the polarization state with stokes parameters (normalized before tracking);
    "angle [theta] [phi]"   : sets the polarization state with polar coordinates (in degrees);
    "h"                     : sets the polarization state to H
    "d"                     : sets the polarization state to D
"""

import socket
import select
import time
# import TimeTagger
import numpy as np
import matplotlib


class psyClient():

    def __init__(self, ip='10.10.101.21', port=55000, timeout=3):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect((ip, port))

    def _send(self, msg):
        self.s.send((msg + "\r\n").encode())
        msgResp = self.msgResponse()
        return (msgResp)

    def msgResponse(self):
        inputs = [self.s]
        outputs = []
        response = ''
        while inputs:
            readable, writable, exceptional = select.select(inputs, outputs, inputs)
            for s in readable:
                response += s.recv(1024).decode()
                # print 'respone:',repr(response)
            if len(response) == 0:
                break
            if response.lower().endswith('ack\n'):
                break
        return response[:-4]

    def disconnect(self):
        self.s.close()
        print("disconnected")


if __name__ == '__main__':
    NISThost = '10.10.101.20'
    NISTport = 56010

    mc = psyClient(NISThost, NISTport)
    print(mc._send("*idn?"))
    # mc._send("sop 0 1 0")
    # print(mc._send("sop?"))
    # mc._send("angle 0 90")
    mc._send("r")
    print(mc._send("sop?"))


    mc.disconnect()
