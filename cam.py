from picamera2 import Picamera2
import cv2
import numpy as np

import asyncio
import json
from pathlib import Path
from time import perf_counter as now
import sys
import math
from colorsys import hsv_to_rgb

from masks import *



def normalise(angle_degrees: float) -> float:
    return (angle_degrees + 180) % 360 - 180
def angle_lerp(alpha: float, beta: float, t: float) -> float:
    return alpha + normalise(beta - alpha) * t
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
def clamp(x, a: float = -1, b: float = 1) -> float:
    return min(max(x, a), b)
    
def convert_hsv(hsv: tuple, format_: str = "rgb") -> tuple:
    """
    H: 0 - 180
    S: 0 - 255
    V: 0 - 255
    """
    ret = tuple([int(x * 255) for x in hsv_to_rgb(hsv[0] / 180, hsv[1] / 255, hsv[2] / 255)])
    if format_ == "bgr": return ret[::-1]
    return ret

def load_config(filename="config.json") -> dict:
    with open(Path(__file__).parent / filename, "r") as f:
        data = json.load(f)
        f.close()
    return data

class Conglomerator:
    def compute_maximal_conglomerate(self, contours: np.array,
                    maximum_span: float,
                    minimum_contour_size: int,
                    minimum_conglomerate: int = 3) -> np.array:
        if len(contours) == 0: return
        
        contours_by_size = sorted(contours, key=lambda contour: contour.size, reverse=True)
        
        points = []
        previous_center = None
        for i, contour in enumerate(contours_by_size):
            if contour.size < minimum_contour_size: continue
            
            M = cv2.moments(contour)
            if M["m00"] == 0: continue
            
            center = [ M["m10"] / M["m00"], M["m01"] / M["m00"] ]
            if previous_center is None:
                last_span = 0
            else:
                last_span = -cv2.pointPolygonTest(contour, previous_center, True)
                # ~ last_span = math.sqrt(pow(center[0] - previous_center[0], 2) + pow(center[1] - previous_center[1], 2))
                
            if last_span > maximum_span: continue
            
            previous_center = center.copy()
            points += [point for point in contour]
            
        if len(points) <= minimum_conglomerate: return
            
        conglomerate = cv2.convexHull(np.array(points, dtype=np.int32))
        if conglomerate.size <= minimum_conglomerate: return
        
        return conglomerate

class ColorRange:
    def __init__(self, min_: list, max_: list):
        self.min = np.array(min_, dtype=np.float32)
        self.max = np.array(max_, dtype=np.float32)
    def set(self, lower_upper: int, index: int, value: float):
        if lower_upper == 0:
            self.min[index] = value
        elif lower_upper == 1:
            self.max[index] = value
    def __repr__(self) -> str:
        return f"{[round(x, 2) for x in self.min]}-{[round(x, 2) for x in self.max]}"
class AsyncCam:
    def __init__(self, size = [640, 480], center = None):
        self.size = size
        self.center = center
        if self.center is None:
            self.center = [self.size[0] // 2, self.size[1] // 2]
        
        config = load_config()
        self.controls = {"ExposureTime": 8410, "Saturation": 3}
        self.conglomerate_settings = config["settings"]
        for x in ["ExposureTime", "Saturation"]:
            if x in config: self.controls[x] = config[x]
        
        self.stream = Picamera2(0)
        self.stream.configure(self.stream.create_video_configuration(
            main={"format": "XRGB8888", "size": size},
            raw=self.stream.sensor_modes[0],
            buffer_count=6,
            controls={
                "FrameRate": 120.05,
                "Contrast": 1.0,
                "ScalerCrop": (1980//2, 528, 1980, 1980)
            },
        ))
        self.stream.controls.ExposureTime = self.controls["ExposureTime"]
        self.stream.controls.Saturation = self.controls["Saturation"]
        self.stream.start()
        # ~ print(self.stream.camera_properties)
        print(f"Initialised camera with resolution {size[0]}x{size[1]}, exposure time of {self.controls['ExposureTime'] * 0.001} ms, saturation of {self.controls['Saturation']}")
    
        self.image_ready: bool = False
        self.current_frame: cv2.typing.MatLike = None
        
        self.prev: float = now()
        self.ticks: int = 0
        self.elapsed: float = 0
        self.fps = 0
        
        self.conglomerator = Conglomerator()
        
        self.config = load_config()
        colours = self.config["colours"]
        self.ball = ColorRange(colours["ball"][:3], colours["ball"][3:])
        self.yellow = ColorRange(colours["yellow"][:3], colours["yellow"][3:])
        self.blue = ColorRange(colours["blue"][:3], colours["blue"][3:])

        self.debug: bool = False
        
        self.angle: float = None
        self.true_angle: float = None
        self.distance: float = None
        self.true_distance: float = None
        
        self.body_masks = []
        
    def set_masks(self, masks: list):
        "{type, args*}"
        self.body_masks.clear()
        for mask in masks:
            if mask["type"] == "circle":
                self.body_masks.append(Circle(mask["center"], mask["radius"], mask["colour"]))
            elif mask["type"] == "rect":
                self.body_masks.append(Rect(mask["x"], mask["y"], mask["w"], mask["h"], mask["colour"]))
            elif mask["type"] == "sector":
                self.body_masks.append(Sector(mask["center"], mask["radius"], mask["startAngle"], mask["endAngle"], mask["colour"]))
                
    def calculate_distance(self, radial_distance: float, a: float = 677.87957, b: float = 325.553) -> float:
        return pow(a / (radial_distance - b), 2)
    
    def process(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.ball.min, self.ball.max) 
        # ~ mask = cv2.inRange(hsv, self.yellow.min, self.yellow.max)
        # ~ mask = cv2.inRange(hsv, self.blue.min, self.blue.max)
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        conglomerate = self.conglomerator.compute_maximal_conglomerate(contours, **self.conglomerate_settings["ball"])
        if conglomerate is None: return frame
        
        ellipse = cv2.fitEllipse(conglomerate)
        center, size, angle = ellipse
        dx, dy = center[0] - self.center[0], center[1] - self.center[1]
        
        self.true_angle = math.degrees(math.atan2(dy, dx))
        self.true_distance = self.calculate_distance(math.sqrt(pow(dx, 2) + pow(dy, 2)) * 540 / self.size[1])
        
        if self.angle is None: self.angle = self.true_angle
        self.angle = angle_lerp(self.angle, self.true_angle, t = 0.67)
        if self.distance is None: self.distance = self.true_distance
        self.distance = lerp(self.distance, self.true_distance, t = 0.67)
        
        # ~ print(f"{math.sqrt(pow(dx, 2) + pow(dy, 2)):.0f} {self.true_distance:.0f} {self.distance:.0f}")
        
        if not self.debug: return frame
        
        cv2.drawContours(frame, contours, -1, (0, 220, 0), 1)
        cv2.drawContours(frame, [conglomerate], 0, (255, 255, 255))
        
        pos = [int(center[0]), int(center[1])]
        cv2.ellipse(frame, ellipse, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(frame, pos, (255, 255, 255))
        cv2.line(frame, self.center, pos, convert_hsv(self.ball.min, "bgr"), 5)
        
        return frame
    
    def on_capture_complete(self, job):
        self.current_frame = job.get_result()
        self.image_ready = True
    
    async def main(self):
        while True:
            try:
                self.image_ready = False
                self.stream.capture_array(signal_function=self.on_capture_complete)
                while not self.image_ready:
                    await asyncio.sleep(0.001)
                
                curr = now()
                dt = curr - self.prev
                self.prev = curr
                
                self.ticks += 1
                self.elapsed += dt
                
                if not self.debug:
                    yield self.current_frame
                    continue
                
                if self.elapsed >= 1:
                    self.elapsed -= 1
                    self.fps = self.ticks
                    self.ticks = 0
                
                if self.ticks % 6 == 0:
                    cv2.imshow("Camera", self.process(self.current_frame.copy()))
                    cv2.setWindowTitle("Camera", f"Camera FPS: {self.fps}")
                    cv2.waitKey(1)
                
                yield self.current_frame
                
            except Exception as exc:
                print(exc)

if __name__ == "__main__":
    cam = AsyncCam([800, 800])
    cam.debug = True
    async def start():
        async for _ in cam.main():
            pass
    asyncio.run(start())
