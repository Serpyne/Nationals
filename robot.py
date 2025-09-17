"""

Main TS robot code; I plan to use this file in competitions.

"""

from tof import TOF, TOFChain
from compass import Compass
from motors_i2c import Motor
from cam import AsyncCam, normalise, lerp, angle_lerp, clamp, sign
from solenoid import Solenoid

from gpiozero import Button
from vector import Vector
import math
import asyncio
import json
import os
from time import perf_counter as now, sleep
from pathlib import Path
import cv2
import numpy as np



def smooth_linear(x: float, a = 674.1) -> float:
    return pow(x, 3) / (a + pow(x, 2))

class Goal:
    Yellow = "Yellow"
    Blue = "Blue"
class Mode:
    Update = "Update"
    Idle = "Idle"
    Calibrate = "Calibrate"
class RobotVars:
    ball_angle: float = 0
    normalised_ball_angle: float = 0
    ball_distance: float = 0
    heading: float = 0
    initial_headings: list[float] = None
    last_seen_ball: int = 0
    has_ball: int = 0
    tof_distances: list[float] = []
    frontTofDistance: float = 0
    
    blind_milliseconds: int = 670
    target_goal: int = Goal.Yellow
    drive_speed: float = 0.6
    top_speed: float = 0.8
    dribble_speed: float = 0.5
    maintain_orientation_speed: float = 1 / 67
    mode: int = Mode.Idle
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
class State:
    Chasing = "Chasing"
    Defending = "Defending"
    Shooting = "Shooting"
    Stalled = "Stalled"
    KickOff = "KickOff"
    Blind = "Blind"
class Blackboard:        
    atDefenderGoal: bool = False
    atAttackingGoal: bool = False
    goalTicks: int = 0
    xPosition: float = None
    kickoffDuration: float = 2000
    kickoffTimer: float = 0
    targetDirection: float = None
class Robot:
    def __init__(self, loop, motors, camera, compasses: list, tofs, captureTof):
        self.loop = loop
        
        self.mode: int = Mode.Idle
        self.state: str = State.Chasing
        self.previous_state: str = State.Chasing
        self.bb = Blackboard()
        
        self.future_motor_speeds: list[float] = [0, 0, 0, 0]
        
        self.vars = RobotVars()
        with open(Path(__file__).parent / "config.json", "r") as f:
            self.config = json.load(f)
            f.close()
            
        self.vars.blind_milliseconds = self.config["blindnessTimer"]
        self.vars.drive_speed = self.config["speedDrive"]
        self.vars.top_speed = self.config["speedMax"]
        self.vars.target_goal = Goal.Blue if self.config["targetGoal"] == "blue" else Goal.Yellow
        self.vars.maintain_orientation_speed = self.config["maintainOrientationTurnSpeed"]
        self.vars.dribble_speed = self.config["speedDribble"]
        self.cameraOrientation = self.config["cameraOrientation"]
        
        self.vars.initial_headings = [0 for x in compasses]
        
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

        def f(x, k= 0.8, a = 5, D = 14.2) -> float:
            try:
                return k / (1 + math.exp(-4 + (1 / a) * (x - D)))
            except OverflowError:
                return 0
            
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

    def drive_direction_bias(self, a: float, d: float) -> float:
        return a
    def drive_speed_bias(self, a: float, d: float) -> float:
        return 1
        
    async def update(self):
        "Logic for the robot gameplay"
        
        span = 30 # degrees
        held_distance = 18.0 # cm
        ball_is_at_front = abs(self.vars.normalised_ball_angle) <= span
        view_ball_as_captured = (ball_is_at_front and self.vars.ball_distance < held_distance) if self.vars.ball_distance is not None else 0
        ball_is_close_to_tof = self.vars.frontTofDistance < 90.0
        if (ball_is_close_to_tof and not ball_is_at_front) or view_ball_as_captured:
            self.vars.has_ball = min(400, self.vars.has_ball + self.dt)
        else:
            self.vars.has_ball = max(0, self.vars.has_ball - self.dt)
        
        if self.vars.has_ball > 0:
            await self.enable_dribbler()
        else:
            await self.stop_dribbler()
                
        
        if self.state == State.Blind:
            """
            Centre on the field horizontally
            """
            
            localisation_angle = self.utils.camera.blue_angle
            if localisation_angle is None: localisation_angle = self.utils.camera.yellow_angle
            
            # If goals not seen, spin.
            if localisation_angle is None:
                await self.turn(0.167, 1.0)
                await self.confirm_drive()
                return
                
            await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.4)
            
            # Center yourself
            a = normalise(localisation_angle - self.vars.heading)
            speed = self.vars.center_speed * (1 - math.cos(math.radians(3 * ((a + 90) % 180 - 90))))
            await self.drive_in_direction(self.vars.heading + 90, speed * sign(a), 1.0)
                
            await self.confirm_drive()
            
            self.vars.has_ball = 0
            await self.stop_dribbler()
        if self.state in [State.Chasing, State.Defending]:
            self.previous_state = self.state
        
        if self.state == State.Chasing:
            """
            Do normal chase code
            Follow ball polynomial (with prediction of ball velocity?)
            Face north
            """
            
            if self.vars.has_ball >= 100:
                self.state = State.Shooting
                return
                
            await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.4)
        
            direction = self.calculate_final_direction(self.vars.normalised_ball_angle, self.vars.ball_distance)
            direction = self.drive_direction_bias(direction, self.vars.ball_distance)
            speed = self.drive_speed_bias(direction, self.vars.ball_distance) * self.vars.drive_speed
            
            await self.drive_in_direction(direction + self.vars.heading, speed, contribution = 1.0)
            await self.confirm_drive()
        
        elif self.state == State.Shooting:
            """
            Don't maintain heading.
            update targetDirection to be left or right depending on self.xPosition
            Turn to face targetDirection
            THEN, drive slowly backwards to the goal until self.atAttackingGoal
            Shooting with solenoid
            """
            
            # Go back to chasing if the ball is too far away
            # ~ if not (ball_is_at_front or self.utils.camera.angle is None) or self.vars.ball_distance >= 30.0:
            if self.vars.has_ball < 100:
                self.state = State.Chasing
            
            if self.vars.target_goal == Goal.Yellow: target_angle = self.utils.camera.yellow_angle
            else:                                    target_angle = self.utils.camera.blue_angle
                
            # If the goals are not visible, just drive forward
            if target_angle is None:
                await self.drive_in_direction(0, self.vars.dribble_speed, contribution = 1.0)
                await self.confirm_drive()
                return
            
            # Turn and drive to the goal.
            await self.turn(target_angle * .0067, contribution=0.8)
            if abs(target_angle) < 25:
                await self.drive_in_direction(0, self.vars.dribble_speed, contribution = 1.0)
            await self.confirm_drive()
                
        elif self.state == State.Defending:
            """
            Go to goal until atDefenderGoal is true
            oscillate on the goalline (proportional to self.goalTicks).
            Face towards ball/northward.
            """
            
            self.bb.goalTicks += 1
            
            if self.vars.target_goal == Goal.Blue:
                backing_distance = self.utils.camera.yellow_distance
                localisation_angle = self.utils.camera.yellow_angle
            else:
                backing_distance = self.utils.camera.blue_distance
                localisation_angle = self.utils.camera.blue_angle
            
            # ~ # Back up to the goal.
            if backing_distance is not None:
                dist_from_goal = 30.0 #cm
                await self.drive_in_direction(180, 0.5 * smooth_linear(backing_distance - dist_from_goal), 1.0)
        
            # If goals can't be found, just spin.
            if localisation_angle is None:
                await self.turn(0.1, 1.0)
                await self.confirm_drive()
                return
                
            await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.67)
            
            # Center yourself
            offset = 30 * math.sin(self.bb.goalTicks * .024)
            a = normalise(localisation_angle + offset - self.vars.heading)
            speed = self.vars.center_speed * (1 - math.cos(math.radians(3 * ((a + 90) % 180 - 90))))
            await self.drive_in_direction(self.vars.heading + 90, speed * sign(a), 1.0)
                
            await self.confirm_drive()
            
        elif self.state == State.Stalled:
            """
            TO FIND STALLED STATE: Read encoders to see if things are thingy
            do things 
            """
            ...
        elif self.state == State.KickOff:
            """
            Drive directly to the ball until self.kickoffTimer <= 0.
            update self.kickoffTimer
            """
            ...
            
        
    async def idle(self):
        "Robot is not moving or sensing"
        await self.brake()
        await self.stop_dribbler()
        await asyncio.sleep(2 * self.update_interval)
                
        # Attempt to reset things here because the thing kinda tweaks out after
        # ~ self.vars.ball_angle = 0
        # ~ self.vars.ball_distance = 1000
        # ~ self.vars.last_seen_ball = 0
        # ~ self.vars.has_ball = 0
    
    def get_orientation(self) -> float:
        angles = [normalise(compass.read() - self.vars.initial_headings[i]) for i, compass in enumerate(self.utils.compasses) if compass.enabled]
        if len(angles) == 0: return None
        return sum(angles) / len(angles)
        
    async def calibrate(self):
        "Set initial heading"
        await self.brake()
        await self.stop_dribbler()
        
        for i, compass in enumerate(self.utils.compasses):
            if not compass.enabled: continue
            self.vars.initial_headings[i] = compass.read()
        
        if self.utils.camera.yellow_angle is not None:
            if abs(self.utils.camera.yellow_angle) > 90:
                 self.vars.target_goal = Goal.Blue
        if self.utils.camera.blue_angle is not None:
            if abs(self.utils.camera.blue_angle) > 90:
                 self.vars.target_goal = Goal.Yellow
        
        await asyncio.sleep(2 * self.update_interval)
        
    async def start(self):
        
        await self.brake()
        await self.stop_dribbler()
        self.utils.camera.debug = True
        
        center = np.array(self.utils.camera.center.copy(), dtype=np.uint32)
        
        def arrow(origin: np.array, degrees: float, radius: float) -> tuple:
            if type(origin) != np.array: origin = np.array(origin, np.uint32)
            
            a = math.radians(degrees)
            return (origin, origin + np.array([radius * math.cos(a), radius * math.sin(a)], dtype=np.int32))
        
        calibration_count = 60
        # Sync up the robot event loop with the camera output 
        async for raw_frame in self.utils.camera.main():
            curr = now()
            self.dt = (curr - self.prev) * 1000
            self.prev = curr
            
            frame = self.utils.camera.process(raw_frame.copy())
            if self.utils.camera.debug and self.utils.camera.ticks % 8 == 0:
                # rotate so ball is up
                # ~ angle = self.utils.camera.angle + 90
                # ~ M = cv2.getRotationMatrix2D(self.utils.camera.center, angle, 1.0)
                # ~ drawn_frame = cv2.warpAffine(drawn_frame, M, self.utils.camera.size)
                
                if self.vars.ball_angle is not None:
                    direction = self.calculate_final_direction(self.vars.normalised_ball_angle, self.vars.ball_distance)
                    cv2.line(frame, *arrow(center, direction, 150), [255, 255, 255], 3)
                cv2.line(frame, *arrow(center, self.vars.heading, 50), [150, 150, 255], 5)
                
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.putText(frame, str(self.state), (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, False)
                cv2.imshow("Camera", frame)
                cv2.setWindowTitle("Camera", f"Camera FPS: {self.utils.camera.fps}")
                cv2.waitKey(1)
            
            self.vars.frontTofDistance = self.utils.captureTof.read()
            self.vars.heading = self.get_orientation()
            
            if None in [self.utils.camera.angle, self.utils.camera.distance]:
                if self.vars.last_seen_ball <= 0 and self.state not in [State.Shooting]:
                    self.state = State.Blind
                    self.vars.ball_angle = None
                    self.vars.ball_distance = None
                else:
                    self.vars.last_seen_ball = max(0, self.vars.last_seen_ball - self.dt) # Milliseconds
            else:
                if self.state == State.Blind:
                    self.state = self.previous_state
                self.vars.last_seen_ball = self.vars.blind_milliseconds
                self.vars.ball_angle = normalise(self.utils.camera.angle + self.cameraOrientation - 180)
                self.vars.ball_distance = self.utils.camera.distance
            
            # Setting blackboard variables
            if self.vars.ball_angle:
                self.vars.normalised_ball_angle = normalise(self.vars.ball_angle - self.vars.heading)
        
            # I do this so that when it starts up it doesnt just sprint away
            if calibration_count > 0:
                await self.calibrate()
                calibration_count -= 1
        
            if self.utils.switch_left.is_pressed:
                mode = Mode.Update
                await self.update()
                
            elif self.utils.switch_right.is_pressed:
                mode = Mode.Calibrate
                await self.calibrate()
                
            else:
                mode = Mode.Idle
                await self.idle()
                
            if self.vars.mode != mode:
                self.vars.mode = mode
                print(f"Switch: {mode}")
                
            await asyncio.sleep(self.update_interval)

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
    camera      = AsyncCam([800, 800])
    compasses   = [Compass(0x4a), Compass(0x4b)]
    tofs        = TOFChain([0x50, 0x53, 0x54, 0x56])
    capture_tof = TOF(0x57)
    solenoid    = Solenoid(17)

    loop = asyncio.get_event_loop()
    for index in motors:
        motor = motors[index]
        loop.create_task(motor.event_loop())
    ts = Robot(loop, motors, camera, compasses, tofs, capture_tof)
    loop.create_task(ts.start())
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        sleep(0.1)
        
        # Kill motors once stopped by user
        for index in motors:
            motor = motors[index]
            motor.set_immediately(0)
            
        print("Robot stopped by user")
    except Exception as e:
        print(e)
    
    
