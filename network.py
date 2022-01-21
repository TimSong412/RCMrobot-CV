import socket
import time


class Sender():
    def __init__(self, addr: str) -> None:
        self.sk = socket.socket()
        self.sk.connect((addr, 31901))
        print("CONNECTED!")

    def sendvec(self, x, y):
        self.sk.send((str(x)+','+str(y)).encode('UTF-8'))
        print("SENT vector: (", x, ", ", y, ")")

    def close(self):
        self.sk.close()
        print("socket CLOSED!")


if __name__ == '__main__':
    sd = Sender('192.168.71.60')
    for i in range(10):
        sd.sendvec(i/13, (i+1)/13)
        time.sleep(0.01)
    time.sleep(5)
    sd.close()
