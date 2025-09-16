"""

Main TS robot code; I plan to use this file in competitions.

"""

from tof import TOF, TOFChain
from compass import Compass
from motors_i2c import Motor
from cam import AsyncCam, normalise, lerp, angle_lerp, clamp

from gpiozero import Button
from vector import Vector
import math
import asyncio
import json
import os
from time import perf_counter as now, sleep
from pathlib import Path



class Goal:
    Yellow = "Yellow"
    Blue = "Blue"
class Mode:
    Update = "Update"
    Idle = "Idle"
    Calibrate = "Calibrate"
class RobotState:
    ball_angle: float = 0
    ball_distance: float = 0
    heading: float = 0
    initial_headings: list[float] = None
    last_seen_ball: int = 0
    has_ball: int = 0
    tof_distances: list[float] = []
    
    blind_milliseconds: int = 670
    target_goal: int = Goal.Yellow
    drive_speed: float = 0.6
    top_speed: float = 0.8
    dribble_speed: float = 0.5
    maintain_orientation_speed: float = 1 / 67
    new_state: int = Mode.Idle
    center_speed: float = 0.41
class Utilities:
    def __init__(self, motors=None, camera=None, compasses: list=None, tofs=None, captureTof=None):
        self.motors = motors
        self.camera = camera
        self.compasses = compasses
        self.tofs = tofs
        self.switch_left: Button = None
        self.switch_right: Button = None
        self.captureTof = captureTof
class Robot:
    def __init__(self, loop, motors, camera, compasses: list, tofs, captureTof):
        self.loop = loop
        
        self.mode: int = Mode.Idle
        
        self.future_motor_speeds: list[float] = [0, 0, 0, 0]
        
        self.state: RobotState = RobotState()
        with open(Path(__file__).parent / "config.json", "r") as f:
            self.config = json.load(f)
            f.close()
            
        self.state.blind_milliseconds = self.config["blindnessTimer"]
        self.state.drive_speed = self.config["driveSpeed"]
        self.state.top_speed = self.config["topSpeed"]
        self.state.target_goal = Goal.Blue if self.config["targetGoal"] == "blue" else Goal.Yellow
        self.state.maintain_orientation_speed = self.config["maintainOrientationTurnSpeed"]
        self.state.dribble_speed = self.config["dribbleSpeed"]
        self.cameraOrientation = self.config["cameraOrientation"]
        
        self.state.initial_headings = [0 for x in compasses]
        
        self.speedBias = self.config["speedBias"]
        self.angleCoeff = self.config["anglePolyCoefficients"]
        
        self.utils: Utilities = Utilities(motors, camera, compasses, tofs, captureTof)
        self.utils.switch_left = Button(self.config["addresses"]["switchLeft"], pull_up=False)
        self.utils.switch_right = Button(self.config["addresses"]["switchRight"], pull_up=False)
        
        self.utils.camera.set_masks(self.config["cameraMasks"])
        self.prev: float = now()
        self.dt: float = 1/60
        self.update_interval: float = 0.00001
        
    def calculate_final_direction(self, angle: float, distance: float) -> float:
        def angle_poly(x: float) -> float:
            return (self.angleCoeff["x5"] * pow(x, 5)) + (self.angleCoeff["x4"] * pow(x, 4)) + (self.angleCoeff["x3"] * pow(x, 3)) + (self.angleCoeff["x2"] * pow(x, 2)) + (self.angleCoeff["x1"] * x)

        def f(x, a = 7, D = 5.0) -> float:
            return 1 / (1 + math.exp(-4 + (1 / a) * (x - D)))
            
        angle = normalise(angle)
        is_negative: bool = angle < 0
        
        mapped_angle: float = angle_poly(angle) if angle > 0 else -angle_poly(abs(angle))

        final_angle = angle + normalise(mapped_angle - angle) * f(distance)
        return normalise(final_angle)

    async def drive_in_direction(self, angle: float, speed: float, contribution: float = 1.0):
        FL = math.sin(math.radians(35 - angle))
        FR = math.sin(math.radians(35 + angle))

        if abs(FL) >= abs(FR):
            FR = (speed / abs(FL)) * FR
            FL = (speed / FL) * abs(FL)
        elif abs(FL) < abs(FR):
            FL = (speed / abs(FR)) * FL
            FR = (speed / FR) * abs(FR)
            
        self.future_motor_speeds[0] += FL * contribution
        self.future_motor_speeds[1] += FR * contribution
        self.future_motor_speeds[2] += -FL * contribution
        self.future_motor_speeds[3] += -FR * contribution
    async def turn(self, speed: float, contribution: float = 1.0):
        for i in range(4):
            self.future_motor_speeds[i] += clamp(speed * contribution, -1, 1)
    async def brake(self):
        self.utils.motors[0].set_speed(0)
        self.utils.motors[1].set_speed(0)
        self.utils.motors[2].set_speed(0)
        self.utils.motors[3].set_speed(0)

    async def confirm_drive(self):
        self.utils.motors[0].set_speed(self.future_motor_speeds[0])
        self.utils.motors[1].set_speed(self.future_motor_speeds[1])
        self.utils.motors[2].set_speed(self.future_motor_speeds[2])
        self.utils.motors[3].set_speed(self.future_motor_speeds[3])
        self.future_motor_speeds = [0, 0, 0, 0]
        
    async def enable_dribbler(self):
        self.utils.motors['dribbler'].set_speed(-1)
    async def stop_dribbler(self):
        self.utils.motors['dribbler'].set_speed(0)

    def drive_direction_bias(self, a: float) -> float:
        # Drives more forward when its forward
        a = normalise(a)
        return a - a / ((1 / self.config["directionBias"]) * pow(a, 4) + 1)
        
    def drive_speed_bias(self, a: float, d: float) -> float:
        
        # Drives at 100% sideways and drives 100% on the sides.
        f = 1 - 0.5 * self.speedBias["sideDamping"] * (1 - math.cos(math.radians(2 * a)))
        
        # Composite function to make it so that the forward peak is
        # less than the backwards peaks, and the sides are independent
        g = 1 - self.speedBias["forwardDamping"] / (1 + pow(0.0167 * f, 4))
        
        # Lerp between the target speed and 100% depending on distance
        angled_speed = lerp(g, 1, 1 / (1 + math.exp(15 - 0.5*a)))
        
        return angled_speed
    
    async def goalie(self):
    
        if self.state.target_goal == Goal.Blue:
            backing_distance = self.utils.camera.yellow_distance
            localisation_angle = self.utils.camera.yellow_angle
        else:
            backing_distance = self.utils.camera.blue_distance
            localisation_angle = self.utils.camera.blue_angle
            
        if backing_distance is not None:
            x = backing_distance - 40
            speed = 0.5 / (1 + 4*math.exp(-0.167 * x)) - 0.1
            
            await self.drive_in_direction(180, speed, 1.0)
    
        await self.turn(-self.state.heading * self.state.maintain_orientation_speed, contribution=0.4)
        
        if localisation_angle is not None:
            da = localisation_angle + self.state.heading
            a = 2 * ((da + 90) % 180 - 90)
            speed = (1 - math.cos(math.radians(a))) * self.state.center_speed
            if da > 0:
                await self.drive_in_direction(self.state.heading - 90, speed, 1.0)
            elif da < 0:
                await self.drive_in_direction(self.state.heading + 90, speed, 1.0)
        else:
            # SPIN
            await self.turn(0.1, 1.0)
        await self.confirm_drive()
            
    async def update(self):
        "Logic for the robot gameplay"
        return
        frontTofDistance = self.utils.captureTof.read()
        
        self.state.heading = self.get_orientation()
        
        print(frontTofDistance)
        await self.turn(0.1)
        await self.confirm_drive()
        return
        
        # ~ await self.goalie()
        # ~ return
        
        # OFFENSE
        if None in [self.utils.camera.angle, self.utils.camera.distance]:
            
            await asyncio.sleep(0.01)
            
            if self.state.last_seen_ball <= 0:
                # go to horizontal center of field
                await self.turn(-self.state.heading * self.state.maintain_orientation_speed, contribution=0.4)
                localisation_angle = self.utils.camera.blue_angle
                if localisation_angle is None:
                    localisation_angle = self.utils.camera.yellow_angle
                if localisation_angle is not None:
                    da = localisation_angle + self.state.heading
                    a = 2 * ((da + 90) % 180 - 90)
                    speed = (1 - math.cos(math.radians(a))) * self.state.center_speed
                    if da > 0:
                        await self.drive_in_direction(self.state.heading - 90, speed, 1.0)
                    elif da < 0:
                        await self.drive_in_direction(self.state.heading + 90, speed, 1.0)
                else:
                    # SPIN
                    await self.turn(0.1, 1.0)
                await self.confirm_drive()
                
                self.state.has_ball = 0
                await self.stop_dribbler()
                return
            else:
                self.state.last_seen_ball = max(0, self.state.last_seen_ball - self.dt) # Milliseconds
            
        else:
            self.state.last_seen_ball = self.state.blind_milliseconds
            
            self.state.ball_angle = (180 - (self.utils.camera.angle + self.cameraOrientation)) % 360
            self.state.ball_distance = self.utils.camera.distance
            
        
        normalised_ball_angle = normalise(self.state.ball_angle - self.state.heading)
        
        span = 26.7 # degrees
        held_distance = 14.0 # cm
        view_ball_as_captured = (abs(normalised_ball_angle) < span and self.state.ball_distance < held_distance)
        condition2 = frontTofDistance < 90.0 and abs(normalised_ball_angle) >= span
        condition2 = frontTofDistance < 90.0 and abs(normalised_ball_angle) >= span
        if condition2 or view_ball_as_captured:
            self.state.has_ball = min(400, self.state.has_ball + self.dt)
        else:
            self.state.has_ball = max(0, self.state.has_ball - self.dt)
        
        
        if self.state.has_ball > 0:
            await self.enable_dribbler()
        else:
            await self.stop_dribbler()
            
            
        if self.state.has_ball >= 100:
            if self.state.target_goal == Goal.Yellow:
                target_angle = self.utils.camera.yellow_angle
            else:
                target_angle = self.utils.camera.blue_angle
                
            if target_angle is not None:
                await self.turn(target_angle * .0067, contribution=0.8)
                if abs(target_angle) < 25:
                    await self.drive_in_direction(0, self.state.dribble_speed, contribution = 1.0)
                await self.confirm_drive()
            else:
                await self.drive_in_direction(0, self.state.dribble_speed, contribution = 1.0)
                await self.confirm_drive()
            
        else:
            await self.turn(-self.state.heading * self.state.maintain_orientation_speed, contribution=0.4)
        
            direction = self.calculate_final_direction(normalised_ball_angle, self.state.ball_distance)
            direction = self.drive_direction_bias(direction)
            speed = self.drive_speed_bias(direction, self.state.ball_distance) * self.state.drive_speed
            t = 1 / (1 + math.exp(-4 + (1 / 2.5) * (self.state.ball_distance - 16.7)))
            speed = lerp(self.state.top_speed, speed, t)
            
        
            await self.drive_in_direction(direction + self.state.heading, speed, contribution = 1.0)
            await self.confirm_drive()
            
            
        
    async def idle(self):
        "Robot is not moving or sensing"
        await self.brake()
        await self.stop_dribbler()
        await asyncio.sleep(2 * self.update_interval)
        
        # Attempt to reset things here because the thing kinda tweaks out after
        self.state.ball_angle = 0
        self.state.ball_distance = 1000
        self.state.last_seen_ball = 0
        self.state.has_ball = 0
    
    def get_orientation(self) -> float:
        angles = [normalise(compass.read() - self.state.initial_headings[i]) for i, compass in enumerate(self.utils.compasses) if compass.enabled]
        if len(angles) == 0: return None
        return sum(angles) / len(angles)
        
    async def calibrate(self):
        "Set initial heading"
        await self.brake()
        await self.stop_dribbler()
        for i, compass in enumerate(self.utils.compasses):
            if not compass.enabled: continue
            self.state.initial_headings[i] = compass.read()
        
        if self.utils.camera.yellow_angle is not None:
            if abs(self.utils.camera.yellow_angle) > 90:
                 self.state.target_goal = Goal.Blue
        if self.utils.camera.blue_angle is not None:
            if abs(self.utils.camera.blue_angle) > 90:
                 self.state.target_goal = Goal.Yellow
        
        await asyncio.sleep(2 * self.update_interval)
        
    async def start(self):
        
        await self.brake()
        await self.stop_dribbler()
        self.utils.camera.debug = True   
        
        # Sync up the robot event loop with the camera output 
        async for value in self.utils.camera.main():
        
            if self.utils.switch_left.is_pressed:
                state = Mode.Update
                await self.update()
                
            elif self.utils.switch_right.is_pressed:
                state = Mode.Calibrate
                await self.calibrate()
                
            else:
                state = Mode.Idle
                await self.idle()
                
            if self.state.new_state != state:
                self.state.new_state = state
                print(f"New state: {state}")
                
            await asyncio.sleep(self.update_interval)



def stop_motors():
    for index in motors:
        motor = motors[index]
        motor.set_speed(0)

if __name__ == "__main__":
    with open(Path(__file__).parent / "config.json", "r") as f:
        config = json.load(f)
        f.close()
    
    MOTOR_TOPRIGHT = 0x19
    MOTOR_TOPLEFT = 0x1a
    MOTOR_BOTTOMRIGHT = 0x1c
    MOTOR_BOTTOMLEFT = 0x1b
    DRIBBLER = 0x1e
    
    motors = {
        0: Motor(address=MOTOR_TOPRIGHT),
        1: Motor(address=MOTOR_TOPLEFT),
        3: Motor(address=MOTOR_BOTTOMRIGHT),
        2: Motor(address=MOTOR_BOTTOMLEFT),
        "dribbler": Motor(address=DRIBBLER)
    }
    camera = AsyncCam([800, 800])
    compasses = [Compass(0x4a), Compass(0x4b)]
    tofs = None#TOFChain([0x50, 0x51, 0x52, 0x53, 0x54])
    capture_tof = TOF(0x57)

    try:
        loop = asyncio.get_event_loop()
        ts = Robot(loop, motors, camera, compasses, tofs, capture_tof)
        loop.run_until_complete(ts.start())
        loop.run_forever()
    except Exception as e:
        print(e)
        
        sleep(0.5)
        
        stop_motors()
        
        print("Program Halted")
        
