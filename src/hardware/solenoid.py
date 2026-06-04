from gpiozero import DigitalOutputDevice
from time import sleep
from .motors import Motor
import asyncio

from pathlib import Path
import json
import math

from gpiozero import LED

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
    m = Motor(0x1F, max_speed = 280_000_000)
    motors = {
        0: Motor(25),
        1: Motor(26),
        2: Motor(27),
        3: Motor(28)
    }
    
    def drive(angle, speed, contribution = 1):
        FL = math.sin(math.radians(35 - angle))
        FR = math.sin(math.radians(35 + angle))

        if abs(FL) >= abs(FR):
            FR = (speed / abs(FL)) * FR
            FL = (speed / FL) * abs(FL)
        elif abs(FL) < abs(FR):
            FL = (speed / abs(FR)) * FL
            FR = (speed / FR) * abs(FR)

        motors[0].set_speed(FL * contribution, immediate=True)
        motors[1].set_speed(FR * contribution, immediate=True)
        motors[2].set_speed(-FL * contribution, immediate=True)
        motors[3].set_speed(-FR * contribution, immediate=True)

    runMotors = 1
    DRIBBLER_SPEED = 1
    
    m.set_speed(0, True)

    status_light = LED(26)
    
    async def main():
        speed = 0.5
        # ~ [motors[i].set_speed(0.15, True) for i in range(4)]
        # ~ drive(0, 0.001)
        
        ticks = 90
        angle = 90
        while True:
            if runMotors: m.set_speed(-DRIBBLER_SPEED, True)
            
            # ~ if ticks % 110 == 0:
                # ~ angle += 180
            # ~ ticks += 1
            # ~ drive(angle, .35)
            # ~ speed -= 0.1
            # ~ speed = max(-1.0, speed)
            
            await asyncio.sleep(0.1)
            input()
            
            status_light.on()
            
            if runMotors: m.set_speed(0.5, True)
            await asyncio.sleep(0.15)
            await s.shoot()
                
            await asyncio.sleep(0.1)
            status_light.off()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        status_light.off()
        m.set_speed(0, True)
        [motors[i].set_speed(0, 1) for i in range(4)]
