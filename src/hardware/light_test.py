

from gpiozero import LED
from time import sleep

if __name__ == "__main__":
    status_light = LED(26)
    
    try:
        while True:
            status_light.on()
            print("on")
            sleep(.5)
            status_light.off()
            print('off')
            sleep(.5)
            
    except KeyboardInterrupt:
        status_light.off()
    
