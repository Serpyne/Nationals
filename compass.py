import numpy as np
import board
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR
import time

class Compass:
    def __init__(self, address: int = 0x4a):
        self.i2c = board.I2C()
        self.bno = BNO08X_I2C(self.i2c, address=address)
        time.sleep(0.41)
        self.enabled = False
        try:
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            print(f"Compass {hex(address)} initialised")
            self.enabled = True
        except Exception as e:
            print(f"NOT INITIALISED: Compass {hex(address)}; ", e)
    
    def read(self):
        "Get the yaw component of the BNO08x compass sensor"
        quat = self.bno.quaternion
        w, x, y, z = quat[3], quat[0], quat[1], quat[2]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        yaw_deg = np.degrees(yaw) % 360
        return yaw_deg

    def calibration(self):
        return self.bno.calibration_status
        
if __name__ == "__main__":
	c = Compass(0x4a)
	
	time.sleep(0.5)
	initial = c.read()
	time.sleep(0.5)
	while True:
		print(f"{c.read() - initial:.3f}")
		time.sleep(1 / 60)
