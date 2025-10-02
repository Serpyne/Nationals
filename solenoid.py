from gpiozero import LED
from time import sleep

from pathlib import Path
import json


class Solenoid:
    def __init__(self, activation_pin: int):
        self.pin = activation_pin
        self.device = LED(self.pin)
    def shoot(self):
        # DECIDE IF THIS FUNCTION IS ASYNC OR BLOCKING.
        self.device.on()
        sleep(0.5)
        self.device.off()


# For testing that the kicker won't exceed the maximum power.
# A pass is when the robot is placed in the backleft corner of the defending goal
# and kicked into the attacking goal, the ball bounces off the back of the goal and sits in the penalty area.
if __name__ == "__main__":
    with open(Path(__file__).parent / "config.json", "r") as f:
        config = json.load(f)
        f.close()
        
    self.utils.switch_left = Button(config["addresses"]["switchLeft"], pull_up=False)
    self.utils.switch_right = Button(config["addresses"]["switchRight"], pull_up=False)
    
    solenoid = Solenoid(17)
    
    while True:
        if self.utils.switch_left.is_pressed or self.utils.switch_right.is_pressed:
            solenoid.shoot()
            
        sleep(0.01)
