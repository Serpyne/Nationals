"""
Brushless DC Motor Library ported to python
"""

import smbus2
import time
import struct
import asyncio

MAX_SPEED = 95_000_000
TICK_DURATION = 0.1

def clamp(value, a, b):
    return max(a, min(value, b))

class TimedEvent:
    def __init__(self, speed_value: int, duration: float = -1):
        self.speed = speed_value
        self.duration = duration
    @property
    def indefinite(self):
        return bool(self.duration < 0)

class Motor:
    def __init__(self, address, bus_number: int = 1,
                 current_limit_FOC: int = 65536 * 2,
                 id_PID_constants: tuple[int] = (1500, 200),
                 iq_PID_constants: tuple[int] = (1500, 200),
                 speed_PID_constants: tuple[int] = (0.04, 0.0004, 0.03),
                 elec_angle_offset: int = 1510395136,
                 sin_cos_centre: int = 1251,
                 operating_mode_and_sensor: tuple[int] = (3, 1),
                 command_mode: int = 12):
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

        self.state = TimedEvent(0)
        self.events = []

    async def event_loop(self):
        while True:
            if len(self.events) > 0:
                event = self.events.pop(0)
                #print(f"Setting {self.i2c_address} to {event.speed}.")
                # WRITE DATA AND THEN SET CURRENT STATE TO SPEED
                if self.state != event:
                    data = struct.pack("<i", event.speed)
                    self.bus.write_i2c_block_data(self.i2c_address, 0x12, list(data))
                    self.state = event
                await asyncio.sleep(event.duration)
            else:
                if not self.state.indefinite:
                    if self.state.speed != 0: self.set_speed(0)
                    self.state = TimedEvent(0)
            await asyncio.sleep(TICK_DURATION)

    def set_speed_for(self, speed: int, duration: float):
        speed = clamp(speed, -1, 1)
        speed = int(MAX_SPEED * speed)
        self.events.append(TimedEvent(speed, duration))

    def set_immediately(self, speed: int):
        data = struct.pack("<i", speed)
        self.bus.write_i2c_block_data(self.i2c_address, 0x12, list(data))

    def set_speed(self, speed: int, force=True):
        """Set the speed of the motor, taking into account the direction (via sign)
        ; force[bool] bypasses motor timed events.
        """
        try:
            speed = clamp(speed, -1.0, 1.0)
            speed = int(MAX_SPEED * speed)
            data = struct.pack("<i", speed)
            if self.state.speed == speed: return
            if force: self.events = [TimedEvent(speed)]
            #self.bus.write_i2c_block_data(self.i2c_address, 0x12, list(data))
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
        "dribbler": Motor(address=0x1e)
    }
    
    async def initialise_event_loop(main_func):
        for index in motors:
            motor = motors[index]
            asyncio.create_task(motor.event_loop())
                
        main_task = asyncio.create_task(main_func())
        await asyncio.gather(main_task)

    sequence = [x / 10 for x in range(0, 11)] + [0]
    # sequence = [0, 1, 0, 1, 0]
    async def main():
        set_speed = []
        measured_speed = []
        times = []

        t = 0
        i = 0
        ticks = 0
        duration = 1
        while i < len(sequence) - 1:
            if t >= duration * 10:
                t = 0
                i += 1
            if t == 0: 
                print(f"setting speed {sequence[i]}")
                motors[0].set_speed(sequence[i])
                
            res = motors[0].read()
            print(res)
            times.append(ticks)
            set_speed.append(sequence[i])
            measured_speed.append(res[1] / MAX_SPEED)
            await asyncio.sleep(0.1)
            t += 1
            ticks += 1
        motors[0].set_speed(0)
        ...
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 4))
        plt.plot(times, measured_speed, marker='o', label='Measured Speed')
        plt.plot(times, set_speed, linestyle='--', color='gray', label='Ideal (y=x)')
        plt.xlabel('Set Motor Speed (RPM)')
        plt.ylabel('Measured Motor Speed (RPM)')
        plt.title('Motor Set Speed vs. Measured Speed')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialise_event_loop(main))
    loop.run_forever()
