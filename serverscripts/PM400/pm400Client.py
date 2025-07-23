# -*- coding: utf-8 -*-
"""
Client script for communicating to a remote pm400 server.

Currently implemented commands:
    "*idn?"     : queries the connected optical power meter information;
    "pow?"      : queries the power measurement (W)

"""

import socket
import select
import time
import numpy as np
import matplotlib


class pm400Client():

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
    NISThost = '127.0.0.3'
    NISTport = 55000

    mc = pm400Client(NISThost, NISTport)
    print(mc._send("*idn?"))
    # print(mc._send("pow?"))


    mc.disconnect()
