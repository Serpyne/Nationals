from gpiozero import DigitalOutputDevice
from time import sleep
from motors_i2c import Motor
import asyncio

from pathlib import Path
import json



class Solenoid:
    def __init__(self, activation_pin: int):
        self.pin = activation_pin
        self.device = DigitalOutputDevice(self.pin, active_high=False, initial_value=False)
        
    async def shoot(self):
        self.device.on()
        await asyncio.sleep(0.05)
        self.device.off()
        await asyncio.sleep(0.1)



# For testing that the kicker won't exceed the maximum power.
# A pass is when the robot is placed in the backleft corner of the defending goal
# and kicked into the attacking goal, the ball bounces off the back of the goal and sits in the penalty area.

if __name__ == "__main__":
    s = Solenoid(23)
    m = Motor(0x1B, max_speed = 200_000_000)

    runMotors = False
    m.set_speed(0, True)

    async def main():
        while True:
            if runMotors: m.set_speed(-1, True)
            
            await asyncio.sleep(0.1)
            input()
            
            if runMotors: m.set_speed(0.5, True)
            await asyncio.sleep(0.15)
            await s.shoot()
                
            await asyncio.sleep(0.1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        m.set_speed(0, True)
