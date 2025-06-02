import socket
import select


class Client:

    def __init__(self, ip='10.10.101.21', port=55000, timeout=3):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect((ip, port))

    def _send(self, msg):
        self.s.send((msg + "\n").encode())
        msgResp = self.msgResponse()
        return msgResp

    def msgResponse(self):
        inputs = [self.s]
        outputs = []
        response = ''
        while inputs:
            readable, writable, exceptional = select.select(inputs, outputs, inputs)
            for s in readable:
                response += s.recv(1024).decode()
            if len(response) == 0:
                break
            if response.lower().endswith('ack\n'):
                break
        return response[:-4]

    def disconnect(self):
        self.s.close()
        print("disconnected")


if __name__ == '__main__':
    PA_host = '169.254.142.114'
    PA_port = 56000

    pa = Client(PA_host, PA_port)
    pa._send('retard 3 360')
    print(pa._send('retardances?'))
    # mc.disconnect()
