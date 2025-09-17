from gpiozero import LED
from time import sleep


class Solenoid:
    def __init__(self, activation_pin: int):
        self.pin = activation_pin
        self.device = LED(self.pin)
    def shoot(self):
        # DECIDE IF THIS FUNCTION IS ASYNC OR BLOCKING.
        self.device.on()
        sleep(0.5)
        self.device.off()
