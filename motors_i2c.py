"""
Brushless DC Motor Library ported to python
"""

import smbus2
import time
import struct
import asyncio

MAX_SPEED = 105_000_000

def clamp(value, a, b):
    return max(a, min(value, b))

class Motor:
    def __init__(self, address, bus_number: int = 1,
                 current_limit_FOC: int = 65536 * 2,
                 id_PID_constants: tuple[int] = (1500, 200),
                 iq_PID_constants: tuple[int] = (1500, 200),
                 speed_PID_constants: tuple[int] = (0.04, 0.0004, 0.03),
                 elec_angle_offset: int = 1510395136,
                 sin_cos_centre: int = 1251,
                 operating_mode_and_sensor: tuple[int] = (3, 1),
                 command_mode: int = 12,
                 max_speed: int = None,
                 event_loop_delay: float = 0.005):
        self.i2c_address = address
        self.bus = smbus2.SMBus(bus_number)
        self.QDRformat = 0

        # Default initialisation sequence
        self.set_current_limit_FOC(current_limit_FOC)
        self.set_id_PID_constants(*id_PID_constants)
        self.set_iq_PID_constants(*iq_PID_constants)
        self.set_speed_PID_constants(*speed_PID_constants)
        self.set_elec_angle_offset(elec_angle_offset)
        self.set_sin_cos_centre(sin_cos_centre)
        self.configure_operating_mode_and_sensor(*operating_mode_and_sensor)
        self.configure_command_mode(command_mode)
        
        self.max_speed = max_speed if max_speed is not None else MAX_SPEED
        self.current_speed: float = 0
        self.last_speed: float = 0
        
        self.event_loop_delay: float = event_loop_delay

    def event_loop(self):
        while True:
            if abs(self.last_speed - self.current_speed) >= 0.01:
                data = struct.pack("<i", self.current_speed)
                self.bus.write_i2c_block_data(self.i2c_address, 0x12, list(data))
            
            time.sleep(self.event_loop_delay)

    def set_speed(self, speed: int, immediate: bool = False):
        if not immediate:
            self.last_speed = self.current_speed
            self.current_speed = int(self.max_speed * clamp(speed, -1.0, 1.0))
            return
            
        try:
            self.last_speed = self.current_speed
            self.current_speed = int(self.max_speed * clamp(speed, -1.0, 1.0))
            data = struct.pack("<i", self.current_speed)
            self.bus.write_i2c_block_data(self.i2c_address, 0x12, list(data))
        except Exception as e:
            print(f"Error setting Speed: {e}")

    def set_iq_PID_constants(self, kp, ki):
        try:
            data = struct.pack("<ii", kp, ki)
            self.bus.write_i2c_block_data(self.i2c_address, 0x40, list(data))
        except Exception as e:
            print(f"Error setting Iq PID constants: {e}")

    def set_id_PID_constants(self, kp, ki):
        try:
            data = struct.pack("<ii", kp, ki)
            self.bus.write_i2c_block_data(self.i2c_address, 0x41, list(data))
        except Exception as e:
            print(f"Error setting Id PID constants: {e}")

    def set_speed_PID_constants(self, kp, ki, kd):
        try:
            data = struct.pack("<fff", kp, ki, kd)
            self.bus.write_i2c_block_data(self.i2c_address, 0x42, list(data))
        except Exception as e:
            print(f"Error setting Speed PID constants: {e}")

    def configure_operating_mode_and_sensor(self, operatingmode, sensortype):
        try:
            self.bus.write_byte_data(self.i2c_address, 0x20, operatingmode + (sensortype << 4))
        except Exception as e:
            print(f"Error configuring Operating Mode and Sensor: {e}")

    def configure_command_mode(self, commandmode):
        try:
            self.bus.write_byte_data(self.i2c_address, 0x21, commandmode)
        except Exception as e:
            print(f"Error configuring Command Mode: {e}")

    def set_torque(self, torque):
        try:
            data = struct.pack("<i", torque)
            self.bus.write_i2c_block_data(self.i2c_address, 0x11, list(data))
        except Exception as e:
            print(f"Error setting Torque: {e}")

    def set_position(self, position, elecangle):
        try:
            data = struct.pack("<I", position)
            self.bus.write_i2c_block_data(self.i2c_address, 0x13, list(data))
            self.send8bitvalue(elecangle)
        except Exception as e:
            print(f"Error setting Position: {e}")

    def set_current_limit_FOC(self, current):
        try:
            data = struct.pack("<i", current)
            self.bus.write_i2c_block_data(self.i2c_address, 0x33, list(data))
        except Exception as e:
            print(f"Error setting Current Limit FOC: {e}")

    def set_elec_angle_offset(self, ELECANGLEOFFSET):
        try:
            data = struct.pack("<I", ELECANGLEOFFSET)
            self.bus.write_i2c_block_data(self.i2c_address, 0x30, list(data))
        except Exception as e:
            print(f"Error setting ELECANGLEOFFSET: {e}")

    def set_sin_cos_centre(self, SINCOSCENTRE):
        try:
            data = struct.pack("<i", SINCOSCENTRE)
            self.bus.write_i2c_block_data(self.i2c_address, 0x32, list(data))
        except Exception as e:
            print(f"Error setting SINCOSCENTRE: {e}")
            
    def read(self):
        self.bus.write_i2c_block_data(self.i2c_address, 0x10, [0x1])
        res = self.bus.read_i2c_block_data(self.i2c_address, 0x10, 10)
        position = int.from_bytes(bytes(res[:4]), byteorder='little', signed=True)
        speed = int.from_bytes(bytes(res[4:8]), byteorder='little', signed=True)
        error1 = res[8]
        error2 = res[9]
        return [position, speed, error1, error2]

if __name__ == "__main__":
    motors = {
        0: Motor(address=0x19),
        1: Motor(address=0x1a),
        3: Motor(address=0x1c),
        2: Motor(address=0x1b),
        "dribbler": Motor(address=0x1e, max_speed=260_000_000)
    }
    
    try:
        motors["dribbler"].set_speed(-1.0, immediate=True)
        while True:
            time.sleep(.01)
    except KeyboardInterrupt:
        motors["dribbler"].set_speed(0, immediate=True)
