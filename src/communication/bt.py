import socket
import struct
import asyncio

def f2b(x):
    return struct.pack("b", int(x * 127))
def b2f(x):
    return struct.unpack("b", x.to_bytes(1))[0] / 127

STATES =["Blind",
        "Chasing",
        "Shooting",
        "Defending",
        "Stalled",
        "KickOff"]

class Message:
    Undefined = 0
    Update = 1
    Command = 2
    Solo = 3
class BT:
    def __init__(self, mode: str = "server"):
        self.mode: str = mode
        self.host = "88:A2:9E:30:0D:9D"
        self.port = 3

        self.socket = None
        self.conn = None
        self.addr = None

        self.packet_ready = False

        self.msg_type = Message.Undefined
        self.state = 0
        self.x = 0
        self.y = 0
        self.ball_x = 0
        self.ball_y = 0

    async def start_server(self):
        self.socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)

        """
        bit representation
        total 40 bits, 5 bytes

        type of message [motion data or switch to goalie] (bool) 4 bit
        current state (int) 4 bits

        position (float, float) 8 + 8 bits
        ball position (float, float) 8 + 8 bits

        pos values: -1 to 1. Multiply by 127. Range becomes -127 to 127.
            use as 8bit signed integer.

        inverse: convert 8 bits to signed integer.
        divide by 127.
        there is the position values.
        """

        print("Waiting for client robot to connect")

        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.conn, self.addr = self.socket.accept()
        self.socket.settimeout(0.01)

        with self.conn:
            print('Connected by', self.addr)

            while True:
                try:
                    if not self.packet_ready:
                        # Unpack teammate data
                        try:
                            data = self.conn.recv(5)
                            
                            msg_type = data[0] >> 4
                            state = data[0] - msg_type * 16
                            x = b2f(data[1])
                            y = b2f(data[2])
                            ball_x = b2f(data[3])
                            ball_y = b2f(data[4])

                            yield {"type": msg_type, "state": state, "pos": (x, y), "ball": (ball_x, ball_y)}
                        except TimeoutError:
                            yield None

                        # ~ await asyncio.sleep(0.01)
                        continue

                    first_byte = struct.pack("b", int(16 * self.msg_type + self.state))
                    x_byte = f2b(self.x)
                    y_byte = f2b(self.y)
                    ball_x_byte = f2b(self.ball_x)
                    ball_y_byte = f2b(self.ball_y)

                    packet = first_byte + x_byte + y_byte + ball_x_byte + ball_y_byte
                    self.conn.sendall(packet)

                    self.packet_ready = False

                    # Unpack teammate data
                    try:
                        data = self.conn.recv(5)
                        
                        msg_type = data[0] >> 4
                        state = data[0] - msg_type * 16
                        x = b2f(data[1])
                        y = b2f(data[2])
                        ball_x = b2f(data[3])
                        ball_y = b2f(data[4])

                        yield {"type": msg_type, "state": state, "pos": (x, y), "ball": (ball_x, ball_y)}
                    except TimeoutError:
                        yield None
                        
                # Retry server
                except OSError:
                    if self.socket is not None:
                        self.socket.shutdown(socket.SHUT_RDWR)
                        self.socket.close()
                    return

                # Retry server
                except ConnectionResetError:
                    if self.socket is not None:
                        self.socket.shutdown(socket.SHUT_RDWR)
                        self.socket.close()
                    return

    async def start_client(self):
        self.socket = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)

        # Wait for connect
        connected = False
        while not connected:
            try:
                self.socket.connect((self.host, self.port))
                connected = True
                print("Connected to server")
            except OSError:
                print("Waiting for server to start")

        while True:
            data = self.socket.recv(5)
            if not data: break

            # Receive teammate position
            msg_type = data[0] >> 4
            state = data[0] - msg_type * 16
            x = b2f(data[1])
            y = b2f(data[2])
            ball_x = b2f(data[3])
            ball_y = b2f(data[4])

            # Send back your own position
            first_byte = struct.pack("b", int(16 * self.msg_type + self.state))
            x_byte = f2b(self.x)
            y_byte = f2b(self.y)
            ball_x_byte = f2b(self.ball_x)
            ball_y_byte = f2b(self.ball_y)

            packet = first_byte + x_byte + y_byte + ball_x_byte + ball_y_byte
            self.socket.sendall(packet)

            yield {"type": msg_type, "state": state, "pos": (x, y), "ball": (ball_x, ball_y)}

        if self.socket is not None:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()

    def start(self):
        if self.mode == "server":
            return self.start_server()
        elif self.mode == "client":
            return self.start_client()

        raise Exception('BT module cannot be started as its mode is not either "server" or "client".')

if __name__ == "__main__":
    b = BT()

    async def main():
        async for data in b.start():
            print(data)
    asyncio.run(main())
