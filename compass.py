import numpy as np
import board
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler
from time import sleep, perf_counter as now

from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (BNO_REPORT_LINEAR_ACCELERATION,
                            BNO_REPORT_GYROSCOPE,
                            BNO_REPORT_MAGNETOMETER)

class Compass:
    def __init__(self, address: int = 0x4a):
        self.i2c = board.I2C()
        self.bno = BNO08X_I2C(self.i2c, address=address)
        sleep(0.1)
        
        self.madg = None
        self.enabled = False
        try:
            self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
            self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self.bno.enable_feature(BNO_REPORT_MAGNETOMETER)
            
            print(f"Compass {hex(address)} initialised")
            
            self.enabled = True
            sleep(0.1)
            
            self.madg = Madgwick(mag=self.bno.magnetic)
            self.madg_sample = np.array([1.,0.,0.,0.])
            self.prev = now()
            
        except Exception as e:
            print(f"NOT INITIALISED: Compass {hex(address)}; ", e)
    
    def read(self):
        "Get the yaw component of the BNO08x compass sensor"
        if self.madg is None: return
        
        curr = now()
        self.madg.Dt = curr - self.prev
        self.prev = curr
        
        self.madg_sample = self.madg.updateIMU(q = self.madg_sample, gyr = self.bno.gyro, acc = self.bno.linear_acceleration)
        
        return np.degrees(q2euler(self.madg_sample)[2]) # returns yaw component

if __name__ == "__main__":
	c = Compass(0x4a)
	
	sleep(0.5)
	initial = c.read()
	sleep(0.5)
	while True:
		print(f"{c.read() - initial:.3f}")
		sleep(1 / 60)
