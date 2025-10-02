import tkinter as tk
from tkinter import ttk
from RangeSlider.RangeSlider import RangeSliderH
from compass import Compass
from cam import normalise, calculate_distance, angle_lerp, lerp

from picamera2 import Picamera2
import cv2
import numpy as np

import asyncio
import json
from pathlib import Path
from time import perf_counter as now, sleep
import sys
import math

from masks import *



def load_config(filename="config.json") -> dict:
    with open(Path(__file__).parent / filename, "r") as f:
        data = json.load(f)
        f.close()
    return data
    
def save_setting_to_config(config_key: str, setting: str, value: any, filename="config.json"):
    config = load_config()
    config["settings"][config_key][setting] = value
    with open(Path(__file__).parent / filename, "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)
        f.close()
        
    print(f"Saved {config_key} configuration to '{filename}'")

def save_colour_to_config(config_key: str, colour_range: list, filename="config.json"):
    config = load_config()
    config["colours"][config_key] = colour_range
    with open(Path(__file__).parent / filename, "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)
        f.close()
        
    print(f"Saved {config_key} configuration to '{filename}'")

def config_and_save(config_key: str, value: any, filename="config.json"):
    config = load_config()
    config[config_key] = value
    with open(Path(__file__).parent / filename, "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)
        f.close()
        
    print(f"Saved {config_key} in '{filename}'")

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
    def __init__(self, app, size = [640, 480]):
        self.app = app
        self.size = size
        config = load_config()
        if "center" in config:
            self.center = config["center"]
        else:
            self.center = [self.size[0] // 2, self.size[1] // 2]
        
        self.controls = {"ExposureTime": 8410, "Saturation": 3}
        self.conglomerate_settings = config["settings"]
        for x in ["ExposureTime", "Saturation"]:
            if x in config: self.controls[x] = config[x]
        
        self.stream = Picamera2(0)
        self.stream.configure(self.stream.create_video_configuration(
            main={"format": "XRGB8888", "size": size},
            raw=self.stream.sensor_modes[0],
            buffer_count=6,
            controls={"FrameRate": 120.05, "ScalerCrop": (990, 528, 1980, 1980)},
        ))
        self.stream.controls.ExposureTime = self.controls["ExposureTime"]
        self.stream.controls.Saturation = self.controls["Saturation"]
        self.stream.start()
    
        self.image_ready: bool = False
        self.current_frame: cv2.typing.MatLike = None
        
        self.prev: float = now()
        self.ticks: int = 0
        self.elapsed: float = 0
        self.fps = 0
        
        self.mask_types = ["ball", "yellow", "hands", "blue", "lines", "field"]
        self.current_mask: int = 0
        self.show_hull: bool = True
        self.conglomerator = Conglomerator()
        
        self.current_contours: np.array = None
        self.selected_contour: int = None
        
        self.config = load_config()
        self.body_masks = []
        self.set_masks(self.config["cameraMasks"])
        
        colours = self.config["colours"]
        self.ball = ColorRange(colours["ball"][:3], colours["ball"][3:])
        self.yellow = ColorRange(colours["yellow"][:3], colours["yellow"][3:])
        self.blue = ColorRange(colours["blue"][:3], colours["blue"][3:])
        self.lines = ColorRange(colours["lines"][:3], colours["lines"][3:])
        self.field = ColorRange(colours["field"][:3], colours["field"][3:])
        self.hands = ColorRange(colours["hands"][:3], colours["hands"][3:])
        
        self.angle = {"yellow": None, "blue": None}
        self.distance = {"yellow": None, "blue": None}
        
    def set_control(self, control: str, value: float):
        if control not in self.controls: return
        self.controls[control] = value
        self.stream.set_controls({control: value})
        
    def set_setting(self, mask_type: str, setting: str, value: any):
        if mask_type not in self.conglomerate_settings: return
        if setting not in self.conglomerate_settings[mask_type]: return
        
        self.conglomerate_settings[mask_type][setting] = value
        
    def configure(self, mask_type: str, hs_or_v: int, lower_upper: int, value: float):
        if mask_type == "ball":
            self.ball.set(lower_upper, hs_or_v, value)
        elif mask_type == "yellow":
            self.yellow.set(lower_upper, hs_or_v, value)
        elif mask_type == "blue":
            self.blue.set(lower_upper, hs_or_v, value)
        elif mask_type == "lines":
            self.lines.set(lower_upper, hs_or_v, value)
        elif mask_type == "field":
            self.field.set(lower_upper, hs_or_v, value)
        elif mask_type == "hands":
            self.hands.set(lower_upper, hs_or_v, value)
        
    def set_hull_visibility(self, value: bool):
        self.show_hull = bool(value)
        
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
    def draw_body_masks(self, frame, filled=True):
        _fill = 1
        if filled: _fill = -1
        
        for bmask in self.body_masks:
            if type(bmask) == Circle:
                cv2.circle(frame, bmask.center, bmask.radius, bmask.colour, _fill)
            elif type(bmask) == Rect:
                cv2.rectangle(frame, (bmask.x, bmask.y), (bmask.x + bmask.w, bmask.y + bmask.h), bmask.colour, _fill)
            elif type(bmask) == Sector:
                cv2.ellipse(frame, bmask.center, (bmask.radius, bmask.radius),
                0, bmask.start_angle, bmask.end_angle, bmask.colour, _fill)
                
    def on_capture_complete(self, job):
        self.current_frame = job.get_result()
        self.image_ready = True
    
    def process(self, frame: cv2.typing.MatLike, mask_index = None) -> cv2.typing.MatLike:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.circle(frame, [296, 314], 333, (0, 0, 0), 34)
        self.draw_body_masks(frame, filled=False)
        self.draw_body_masks(hsv, filled=True)
        
        if mask_index is None: mask_index = self.current_mask
        
        if mask_index == 0:
            mask = cv2.inRange(hsv, self.ball.min, self.ball.max) 
        elif mask_index == 1:
            mask = cv2.inRange(hsv, self.yellow.min, self.yellow.max)
        elif mask_index == 2:
            mask = cv2.inRange(hsv, self.hands.min, self.hands.max)
        elif mask_index == 3:
            mask = cv2.inRange(hsv, self.blue.min, self.blue.max)
        elif mask_index == 4:
            mask = cv2.inRange(hsv, self.lines.min, self.lines.max)
        elif mask_index == 5:
            mask = cv2.inRange(hsv, self.field.min, self.field.max)
        else: # Corners
            for i in [1, 3]:
                frame = self.process(frame, i)
            
            return frame
            
        # Filter hands
        hands_mask = cv2.inRange(hsv, self.hands.min, self.hands.max)
        hand_contours, _ = cv2.findContours(hands_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, hand_contours, -1, (0, 0, 0), -1)
        if mask_index == 2:
            return frame
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if mask_index is None or 1:
            self.current_contours = contours
            
            if self.selected_contour is not None:
                if self.selected_contour in range(len(contours)):
                    cv2.drawContours(frame, contours, self.selected_contour, (0, 0, 255), 5)
                else:
                    self.selected_contour = None
            
        cv2.drawContours(frame, contours, -1, (50, 255, 50), 1)
        
        if not self.show_hull:
            return frame
        
        if mask_index in range(len(self.mask_types)):
            name = self.mask_types[mask_index]
            conglomerate = self.conglomerator.compute_maximal_conglomerate(contours, **self.conglomerate_settings[name])
            if conglomerate is None:
                if name not in self.angle: return frame
                
                self.angle[name] = None
                self.distance[name] = None
                return frame
            else:
                cv2.drawContours(frame, [conglomerate], 0, (255, 255, 255))
                
                if name not in self.angle:
                    return frame
                
                M = cv2.moments(conglomerate)
                centroid = [M['m10'] // M['m00'], M['m01'] // M['m00']]
                
                dx, dy = centroid[0] - self.center[0], centroid[1] - self.center[1]
                true_angle = normalise(math.degrees(math.atan2(dy, dx)))
                true_distance = calculate_distance(pow(cv2.pointPolygonTest(conglomerate, self.center, True), 2))
                
                if self.angle[name] is None: self.angle[name] = true_angle
                self.angle[name] = angle_lerp(self.angle[name], true_angle, t = 0.67)
                if self.distance[name] is None: self.distance[name] = true_distance
                self.distance[name] = lerp(self.distance[name], true_distance, t = 0.67)
        
        return frame
    
    async def main(self, mouse_event):
        
        def set_control(value: float, control: str):
            if value == self.controls[control]: return
            self.controls[control] = value
            self.stream.set_controls({control: value})
            
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
                
                if self.elapsed >= 1:
                    self.elapsed -= 1
                    self.fps = self.ticks
                    self.ticks = 0
                
                if self.ticks % 6 == 0:
                    self.app.angle = self.angle.copy()
                    self.app.distance = self.distance.copy()
                    
                    frame = self.process(self.current_frame.copy())
                    heading = self.app.root.get_orientation()
                    v = {
                        "yellowGlobalAngle": round(self.angle["yellow"] - heading, 2) if self.angle["yellow"] else None,
                        "blueGlobalAngle": round(self.angle["blue"] - heading, 2) if self.angle["blue"] else None,
                        "yellowDistance": round(self.distance["yellow"], 2) if self.distance["yellow"] else None,
                        "blueDistance": round(self.distance["blue"], 2) if self.distance["blue"] else None
                    }
                    y = 0
                    for key, val in v.items():
                        cv2.putText(frame, f"{key}: {val}", (10, 30 + y * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA, False)
                        y += 1
                    cv2.imshow("Camera", frame)
                    cv2.setWindowTitle("Camera", f"Camera FPS: {self.fps}")
                    cv2.waitKey(1)
                    cv2.setMouseCallback("Camera", mouse_event)
                
            except Exception as exc:
                print("Exception:", exc)
    
class Tab(ttk.Frame):
    def __init__(self, notebook, title: str, initial_hsv_state: list, config_key: str):
        super().__init__()
        self.notebook = notebook
        self.title = title
        self.config_key = config_key
        
        self.hsvVars = []
        for i in range(3):
            self.hsvVars.append([tk.IntVar(value = initial_hsv_state[i]), tk.IntVar(value = initial_hsv_state[i + 3])])
        
        settings = {"font_family": "Adobe Caslon Pro", "padX": 20, "Width": 800, "font_size": 14, "digit_precision": ".0f", "line_width": 10, "bar_radius": 15}
        
        self.main_colour_frame = tk.Frame(self)
        self.main_colour_frame.pack(side=tk.LEFT)
        self.other_colour_frame = tk.Frame(self)
        self.other_colour_frame.pack(side=tk.RIGHT)
        
        tk.Label(self.main_colour_frame, text="Hue").grid(column=0, row=0) 
        self.hueSlider = RangeSliderH(self.main_colour_frame, self.hsvVars[0], min_val = 0, max_val = 180, suffix="°", **settings)
        self.hueSlider.grid(column=1, row=0) 
        
        tk.Label(self.main_colour_frame, text="Saturation").grid(column=0, row=1, padx = 50) 
        self.saturationSlider = RangeSliderH(self.main_colour_frame, self.hsvVars[1], min_val = 0, max_val = 255, **settings)
        self.saturationSlider.grid(column=1, row=1)
        
        tk.Label(self.main_colour_frame, text="Value").grid(column=0, row=2) 
        self.valueSlider = RangeSliderH(self.main_colour_frame, self.hsvVars[2], min_val = 0, max_val = 255, **settings)
        self.valueSlider.grid(column=1, row=2)
        
        self.save_button = tk.Button(self.main_colour_frame, text="Save " + title,
                        command = lambda: save_colour_to_config(config_key, [ self.hsvVars[i][b].get() for b in range(2) for i in range(3)]))
        self.save_button.grid(column=0,row=3)
        
        settings = load_config()["settings"][config_key]
        
        self.conglomerate_setting_names = ["maximum_span", "minimum_contour_size", "minimum_conglomerate"]
        
        self.conglomerate_vars = [
            tk.DoubleVar(value = settings["maximum_span"]),
            tk.IntVar(value = settings["minimum_contour_size"]),
            tk.IntVar(value = settings["minimum_conglomerate"])
        ]
        self.conglomerate_sliders = [
            tk.Scale(self.main_colour_frame, from_=0.1, to=1000.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.conglomerate_vars[0]),
            tk.Scale(self.main_colour_frame, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.conglomerate_vars[1]),
            tk.Scale(self.main_colour_frame, from_=10, to=60, orient=tk.HORIZONTAL, variable=self.conglomerate_vars[2])
        ]
        
        for i, label in enumerate(["Maximum Span", "Minimum Contour Size", "Minimum Conglomerate"]):
            tk.Label(self.main_colour_frame, text=label).grid(column=0, row=4 + i)
            self.conglomerate_sliders[i].grid(column=1,row=4 + i, sticky="EW")
        
class ColorUI:
    def __init__(self, root):
        self.root = root
        self.angle = None
        self.distance = None
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")
        
        
        config = load_config()
        colours = config["colours"]
        
        self.ball_tab = Tab(self.notebook, title="Ball Mask", initial_hsv_state = colours["ball"], config_key="ball")
        self.notebook.add(self.ball_tab, text="Ball")
        
        self.yellow_tab = Tab(self.notebook, title="Yellow Mask", initial_hsv_state = colours["yellow"], config_key="yellow")
        self.notebook.add(self.yellow_tab, text="Yellow")
        
        self.hands_tab = Tab(self.notebook, title="Hands Mask", initial_hsv_state = colours["hands"], config_key="hands")
        self.notebook.add(self.hands_tab, text="Hands")
        
        self.blue_tab = Tab(self.notebook, title="Blue Mask", initial_hsv_state = colours["blue"], config_key="blue")
        self.notebook.add(self.blue_tab, text="Blue")

        self.lines_tab = Tab(self.notebook, title="Lines Mask", initial_hsv_state = colours["lines"], config_key="lines")
        self.notebook.add(self.lines_tab, text="Lines")

        self.field_tab = Tab(self.notebook, title="Field Mask", initial_hsv_state = colours["field"], config_key="field")
        self.notebook.add(self.field_tab, text="Field")
        
        self.corners_tab = ttk.Frame()
        tk.Button(self.corners_tab, text="Reset Heading", command=self.root.reset_heading).pack()
        
        corners = ["TopLeft", "TopRight", "BottomLeft", "BottomRight"]
        def save_corner(corner_name, filename = "config.json"):
            if corner_name not in corners:
                print(f"{corner_name} is an invalid corner name.")
                return
            config = load_config()
        
            heading = self.root.get_orientation()
            
            yn = self.angle["yellow"] is None
            bn = self.angle["blue"] is None
            if yn or bn:
                if yn and bn:
                    print("Goals not seen. You must have both goals on camera.")
                elif yn:
                    print("Yellow goal is not seen.")
                else:
                    print("Blue goal is not seen")
                return
            
            # ~ if self.angle["yellow"] is None:
                # ~ gba = self.angle["blue"] - heading
                # ~ angle_distance_pair = {"blue": [gba, self.distance["blue"]]}
            # ~ elif self.angle["blue"] is None:
                # ~ gba = self.angle["yellow"] - heading
                # ~ angle_distance_pair = {"yellow": [gba, self.distance["yellow"]]}
            # ~ else:
            gya = self.angle["yellow"] - heading
            gba = self.angle["blue"] - heading
            
            # ~ print(self.angle["yellow"], self.angle["blue"])
            
            yellowFront, blueFront = abs(gya) <= 90, abs(gba) <= 90
            if int(yellowFront) + int(blueFront) != 1:
                print("Goals need to be front and back.")
                return
        
            data = {"yellow": {"angle": gya, "dist": self.distance["yellow"]}, "blue": {"angle": gba, "dist": self.distance["blue"]}, "front": ["yellow", "blue"][int(blueFront)]}
        
            config["corners"][corner_name] = data
            with open(Path(__file__).parent / filename, "w") as f:
                json.dump(config, f, sort_keys=True, indent=4)
                f.close()
            print(f"{corner_name} corner saved to {filename}.")
        for corner in corners:
            tk.Button(self.corners_tab, text=f"Save {corner}", command = lambda corner=corner: save_corner(corner)).pack()
        self.notebook.add(self.corners_tab, text="Corners")

class PropertyFrame(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.font = ("Consolas", 15)

    def pad(self, val: int) -> str:
        return ' '*(3 - len(str(val)))

    def set(self, properties: dict):
        for widget in self.winfo_children():
            widget.destroy()
            
        if properties is None: return
            
        index = properties.get("index", "N/A")
        tk.Label(self, text=f"Index: {index}").pack(anchor="w")
        
        size = properties.get("size", "N/A")
        tk.Label(self, text=f"Size: {size}").pack(anchor="w")
        
        min_col_hsv = properties.get("min_colour", (0, 0, 0))
        max_col_hsv = properties.get("max_colour", (180, 255, 255))
        min_col = tuple(cv2.cvtColor(np.array([[min_col_hsv]]), cv2.COLOR_HSV2RGB)[0, 0])
        max_col = tuple(cv2.cvtColor(np.array([[max_col_hsv]]), cv2.COLOR_HSV2RGB)[0, 0])
        
        min_col_str = f"Min: H = {min_col_hsv[0]},{self.pad(min_col_hsv[0])} S = {min_col_hsv[1]},{self.pad(min_col_hsv[1])} V = {min_col_hsv[2]}"
        tk.Label(self, text=min_col_str).pack(anchor="w")
        min_col_hex = '#%02x%02x%02x' % min_col
        min_col_canvas = tk.Canvas(self, width=40, height=20, bg=min_col_hex, bd=1, relief="sunken")
        min_col_canvas.pack(anchor="w", pady=2)
        
        max_col_str = f"Min: H = {max_col_hsv[0]},{self.pad(max_col_hsv[0])} S = {max_col_hsv[1]},{self.pad(max_col_hsv[1])} V = {max_col_hsv[2]}"
        tk.Label(self, text=max_col_str).pack(anchor="w")
        max_col_hex = '#%02x%02x%02x' % max_col
        # ~ print(max_col_hsv, max_col, max_col_hex)
        max_col_canvas = tk.Canvas(self, width=40, height=20, bg=max_col_hex, bd=1, relief="sunken")
        max_col_canvas.pack(anchor="w", pady=2)
        
        for widget in self.winfo_children():
            if type(widget) != tk.Label: continue
            widget.configure(font = self.font)

class Application(tk.Tk):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.tasks = [
            loop.create_task(self.event_loop())
        ]
        self.compasses = [Compass(0x4a), Compass(0x4b)]
        sleep(0.5)
        self.initial_headings = [0 for compass in self.compasses]
        self.reset_heading()
    def reset_heading(self):
        for i, compass in enumerate(self.compasses):
            if not compass.enabled: continue
            self.initial_headings[i] = compass.read()
    
    def get_orientation(self) -> float:
        angles = [normalise(compass.read() - self.initial_headings[i]) for i, compass in enumerate(self.compasses) if compass.enabled]
        if len(angles) == 0: return None
        return sum(angles) / len(angles)
        
    async def event_loop(self):
        while True:
            self.update()
            
            await asyncio.sleep(0.01)
        
    def close(self):
        for task in self.tasks:
            task.cancel()
        self.loop.stop()
        self.destroy()
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
        
    app = Application(loop)
    app.title("Color UI")
    app.geometry("+10+10")
    
    ui = ColorUI(app)
    cam = AsyncCam(ui, [600, 600])

    for b in range(2):
        for i in range(3):
            ui.ball_tab.hsvVars[i][b].trace("w", lambda *_, b=b, i=i: cam.configure("ball", i, b, ui.ball_tab.hsvVars[i][b].get()))
            ui.yellow_tab.hsvVars[i][b].trace("w", lambda *_, b=b, i=i: cam.configure("yellow", i, b, ui.yellow_tab.hsvVars[i][b].get()))
            ui.blue_tab.hsvVars[i][b].trace("w", lambda *_, b=b, i=i: cam.configure("blue", i, b, ui.blue_tab.hsvVars[i][b].get()))
            ui.lines_tab.hsvVars[i][b].trace("w", lambda *_, b=b, i=i: cam.configure("lines", i, b, ui.lines_tab.hsvVars[i][b].get()))
            ui.field_tab.hsvVars[i][b].trace("w", lambda *_, b=b, i=i: cam.configure("field", i, b, ui.field_tab.hsvVars[i][b].get()))
            ui.hands_tab.hsvVars[i][b].trace("w", lambda *_, b=b, i=i: cam.configure("hands", i, b, ui.hands_tab.hsvVars[i][b].get()))
        
            ui.ball_tab.conglomerate_vars[i].trace("w", lambda *_, i=i: cam.set_setting("ball", ui.ball_tab.conglomerate_setting_names[i], ui.ball_tab.conglomerate_vars[i].get()))
            ui.ball_tab.conglomerate_sliders[i].bind("<ButtonRelease-1>", lambda _, i=i: save_setting_to_config("ball", ui.ball_tab.conglomerate_setting_names[i], ui.ball_tab.conglomerate_vars[i].get()))
            ui.yellow_tab.conglomerate_vars[i].trace("w", lambda *_, i=i: cam.set_setting("yellow", ui.yellow_tab.conglomerate_setting_names[i], ui.yellow_tab.conglomerate_vars[i].get()))
            ui.yellow_tab.conglomerate_sliders[i].bind("<ButtonRelease-1>", lambda _, i=i: save_setting_to_config("yellow", ui.yellow_tab.conglomerate_setting_names[i], ui.yellow_tab.conglomerate_vars[i].get()))
            ui.blue_tab.conglomerate_vars[i].trace("w", lambda *_, i=i: cam.set_setting("blue", ui.blue_tab.conglomerate_setting_names[i], ui.blue_tab.conglomerate_vars[i].get()))
            ui.blue_tab.conglomerate_sliders[i].bind("<ButtonRelease-1>", lambda _, i=i: save_setting_to_config("blue", ui.blue_tab.conglomerate_setting_names[i], ui.blue_tab.conglomerate_vars[i].get()))
            ui.lines_tab.conglomerate_vars[i].trace("w", lambda *_, i=i: cam.set_setting("lines", ui.lines_tab.conglomerate_setting_names[i], ui.lines_tab.conglomerate_vars[i].get()))
            ui.lines_tab.conglomerate_sliders[i].bind("<ButtonRelease-1>", lambda _, i=i: save_setting_to_config("lines", ui.lines_tab.conglomerate_setting_names[i], ui.lines_tab.conglomerate_vars[i].get()))
            ui.field_tab.conglomerate_vars[i].trace("w", lambda *_, i=i: cam.set_setting("field", ui.field_tab.conglomerate_setting_names[i], ui.field_tab.conglomerate_vars[i].get()))
            ui.field_tab.conglomerate_sliders[i].bind("<ButtonRelease-1>", lambda _, i=i: save_setting_to_config("field", ui.field_tab.conglomerate_setting_names[i], ui.field_tab.conglomerate_vars[i].get()))
            ui.hands_tab.conglomerate_vars[i].trace("w", lambda *_, i=i: cam.set_setting("hands", ui.hands_tab.conglomerate_setting_names[i], ui.hands_tab.conglomerate_vars[i].get()))
            ui.hands_tab.conglomerate_sliders[i].bind("<ButtonRelease-1>", lambda _, i=i: save_setting_to_config("hands", ui.hands_tab.conglomerate_setting_names[i], ui.hands_tab.conglomerate_vars[i].get()))
        
    show_hull = tk.IntVar(value = cam.show_hull)
    show_hull_button = tk.Checkbutton(app, text=f"Show Hull", variable=show_hull, command=lambda: cam.set_hull_visibility(show_hull.get()))
    show_hull_button.pack()
    
    exposure = tk.IntVar(value = cam.controls["ExposureTime"])
    exposure_slider = tk.Scale(app, from_=500, to=10_000, orient=tk.HORIZONTAL, variable=exposure)
    exposure_slider.pack(fill="x")
    exposure.trace("w", lambda *_: cam.set_control("ExposureTime", exposure.get()))
    exposure_slider.bind("<ButtonRelease-1>", lambda _: config_and_save("ExposureTime", exposure.get()))
        
    saturation = tk.DoubleVar(value = cam.controls["Saturation"])
    saturation_slider = tk.Scale(app, from_=-16, to=16, resolution=0.2, orient=tk.HORIZONTAL, variable=saturation)
    saturation_slider.pack(fill="x")
    saturation.trace("w", lambda *_: cam.set_control("Saturation", saturation.get()))
    saturation_slider.bind("<ButtonRelease-1>", lambda _: config_and_save("Saturation", saturation.get()))
    
    def mouse_event(event_type, x, y, *_):
        if event_type != 1: return
        
        for i, contour in enumerate(cam.current_contours):
            
            signed_dist = cv2.pointPolygonTest(contour, (x, y), True)
            if signed_dist >= -5: break
        else: 
            cam.selected_contour = None
            properties_frame.set(None)
            return
            
        cam.selected_contour = i
        print(f"Selected contour {i}")
        
        contour = cam.current_contours[i]
        hsv = cv2.cvtColor(cam.current_frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        region = hsv[mask == 255]
        
        properties_frame.set({
            "index": i,
            "size": contour.size,
            "min_colour": region.min(axis=0),
            "max_colour": region.max(axis=0)
        })
        
    properties_frame = PropertyFrame(app)
    properties_frame.pack()
    
    cam.current_mask = 0
    def set_mask(index):
        cam.current_mask = index
    ui.notebook.bind("<<NotebookTabChanged>>", lambda x: set_mask(ui.notebook.index(ui.notebook.select())))
    
    app.tasks.append(loop.create_task(cam.main(mouse_event)))
    
    loop.run_forever()
