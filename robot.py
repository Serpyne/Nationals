"""

Main TS robot code; I plan to use this file in competitions.

"""

from tof import TOF, TOFChain
from compass import Compass
from motors_i2c import Motor
from cam import AsyncCam, normalise, lerp, angle_lerp, clamp, sign
from solenoid import Solenoid

from screeninfo import get_monitors

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
from threading import Thread



def clamp_lerp(x, a, b):
    return clamp((x - a) / (b - a), 0, 1)
        
def smooth_linear(x: float, a = 674.1) -> float:
    return pow(x, 3) / (a + pow(x, 2))
    
def in_range(x, x_range) -> bool:
    if x_range[0] > x_range[1]:
        x_range = (x_range[1], x_range[0])
    return x_range[0] <= x and x <= x_range[1]

def calculate_x(y_hit: float, a1: float, a2: float, half_h: float = 183 / 2):
    x_hits = [0, 0]
    a1 -= 90
    a2 -= 90
    if a1 % 180 != 0:
        "y = tan(a1)x - 243/2"
        x_hits[0] = ( y_hit + half_h ) / math.tan(math.radians(a1))
    else:
        x_hits[0] = 0
    if a2 % 180 != 0:
        "y = tan(a2)x + 243/2"
        x_hits[1] = ( y_hit - half_h ) / math.tan(math.radians(a2))
    else:
        x_hits[1] = 0
    return -np.mean(x_hits)
def y_from_distances(d1, d2, half_h = 183 / 2) -> float:
    return half_h * (d1 - d2) / (d1 + d2)
    
def compute_pos_from_corner(corner: dict, front: str, back: str):
    y_hit = y_from_distances(corner[back]["dist"], corner[front]["dist"])
    x_hit = calculate_x(y_hit, corner[back]["angle"], corner[front]["angle"])
    return (x_hit, y_hit)

def collide_point_quadrilateral(point: tuple, quadrilateral: list[tuple]) -> str:
    e1, e2, e3, e4 = quadrilateral
    def collide_line_line(point, direction, start, end):
        if direction == "horizontal":
            x = (point[1] - start[1]) * (start[0] - end[0]) / (start[1] - end[1]) + start[0]
            y = point[1]
            t = (start[1] - y) / (start[1] - end[1])
            return (x, y), t
        elif direction == "vertical":
            y = (point[0] - start[0]) * (start[1] - end[1]) / (start[0] - end[0]) + start[1]
            x = point[0]
            t = (start[0] - x) / (start[0] - end[0])
        return (x, y), t
    # right line
    collision1, t1 = collide_line_line(point, "horizontal", e2, e4)
    # left line
    collision2, t2 = collide_line_line(point, "horizontal", e1, e3)
    # top line
    collision3, t3 = collide_line_line(point, "vertical", e1, e2)
    # bottom line
    collision4, t4 = collide_line_line(point, "vertical", e3, e4)
    
    sides = [point[0] <= collision1[0], point[0] >= collision2[0], point[1] <= collision3[1], point[1] >= collision4[1]]
    delta = [point[0] - collision1[0], collision2[0] - point[0], point[1] - collision3[1], collision4[1] - point[1]]
    for i, side in enumerate(sides):
        if side: delta[i] = 0
    
    return delta, [t1, t2, t3, t4]
    
def determine_position(aB, dB, aY, dY, width = 123.0, length = 183.0):

    dY_squared = pow(dY, 2)
    dB_squared = pow(dB, 2)
    length_squared = pow(length, 2)

    a_Yray = math.radians(90 - aY)
    a_Bray = math.radians(90 - aB)

    a = (dY_squared - dB_squared) * (1 + pow(math.tan(a_Yray), 2))
    b = -2 * length * dY_squared * math.tan(a_Yray)
    c = length_squared * dY_squared
    x_mean = (-b + sign(aY) * math.sqrt(abs(pow(b, 2) - 4 * a * c))) / (2 * a)

    if dY / dB == 1:
        y_approx = 0
    else:
        a = (dY_squared - dB_squared)
        b = -2 * length * dB_squared
        c = (dY_squared - dB_squared) * pow(x_mean, 2) - length_squared * dB_squared
        y_approx = (-b - math.sqrt(pow(b, 2) - 4 * a * c)) / (2 * a)
    
    s = 0.5 * (dY + dB + length)
    if length > s: return
    
    x_3 = - (2 / length) * sign(aY) * math.sqrt(s * (s - dY) * (s - dB) * (s - length))
    y_3 = 0.5 * x_3 * (math.tan(math.radians(90 - aY)) + math.tan(math.radians(90 - aB)))

    t = clamp_lerp(abs(x_3), 0.1 * width, 0.5 * width)
    x_sol = lerp(x_mean, x_3, t)
    if y_3 is None:
        y_sol = y_approx
    else:
        y_sol = lerp(y_approx, y_3, t)
    return (x_sol, y_sol)
    
class Goal:
    Yellow = "Yellow"
    Blue = "Blue"
class Mode:
    Update = "Update"
    Idle = "Idle"
    Calibrate = "Calibrate"
class RobotVars:
    position: list = None
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
    lastSpeed: float = None
    backingDistance: float = 30.0
    outOfBounds: bool = False
    outOfBoundsTicks: int = 0
    lastGlobalYellowAngle: float = None
    lastGlobalYellowDistance: float = None
    lastGlobalBlueAngle: float = None
    lastGlobalBlueDistance: float = None
class Utilities:
    motors = None
    camera = None
    compasses = None
    tofs = None
    switch_left: Button = None
    switch_right: Button = None
    captureTof = None
    solenoid = None
class State:
    Chasing = "Chasing"
    Defending = "Defending"
    Shooting = "Shooting"
    Stalled = "Stalled"
    KickOff = "KickOff"
    Blind = "Blind"
NUM_SAMPLES = 30
class Blackboard:        
    atDefenderGoal: bool = False
    atAttackingGoal: bool = False
    atAttackingGoalTicks: int = 0
    goalTicks: int = 0
    xPosition: float = None
    approxXPosition: float = None
    yPosition: float = None
    kickoffDuration: float = 2000
    kickoffTimer: float = 0
    targetDirection: float = None
    previousTargetDirection: float = 180
    leavingGoal: bool = False
    returnToGoalThreshold: bool = 80.0
    lastAttackingAngle: float = None
    attackingAngle: float = None
    pastAttackingAngles: list[float] = np.zeros(1 + NUM_SAMPLES, dtype=np.int32)
    meanAttackingAngle: float = None
    attackingDistance: float = None
    pastAttackingDistances: list[float] = np.zeros(1 + NUM_SAMPLES, dtype=np.int32)
    meanAttackingDistance: float = None
    defendingAngle: float = None
    lastDefendingAngle: float = None
    defendingDistance: float = None
    lastYellowAngle: float = None
    lastYellowDistance: float = None
    lastBlueAngle: float = None
    lastBlueDistance: float = None
    cameraOrientation: float = None
    capturedSpeed: float = 0
    isKicking: bool = False
    kickedTicks: int = 0
    currTurn: float = 0
    currDrive: list = [0, 0]
    lastNormal = None
class Robot:
    def __init__(self):
        
        with open(Path(__file__).parent / "config.json", "r") as f:
            self.config = json.load(f)
            f.close()

        MOTOR_TOPRIGHT = 0x19
        MOTOR_TOPLEFT = 0x1a
        MOTOR_BOTTOMRIGHT = 0x1c
        MOTOR_BOTTOMLEFT = 0x1b
        DRIBBLER = 0x1e

        self.utils: Utilities = Utilities()
        self.utils.motors = {
            0: Motor(address=MOTOR_TOPRIGHT),
            1: Motor(address=MOTOR_TOPLEFT),
            3: Motor(address=MOTOR_BOTTOMRIGHT),
            2: Motor(address=MOTOR_BOTTOMLEFT),
            "dribbler": Motor(address=DRIBBLER, max_speed=150_000_000)
        }
        
        self.utils.camera      = AsyncCam([800, 800], center=self.config["center"])
        self.utils.compasses   = [Compass(0x4a), Compass(0x4b)]
        self.utils.tofs        = TOFChain([0x50, 0x53, 0x54, 0x56])
        self.utils.captureTof = TOF(0x57)
        self.utils.solenoid    = Solenoid(17)

        self.utils.switch_left = Button(self.config["addresses"]["switchLeft"], pull_up=False)
        self.utils.switch_right = Button(self.config["addresses"]["switchRight"], pull_up=False)
        
        self.mode: int = Mode.Idle
        self.state: str = State.Defending # State.Chasing
        self.previous_state: str = self.state
        self.bb = Blackboard()
        
        self.is_goalie: bool = False#True
        
        self.future_motor_speeds: list[float] = [0, 0, 0, 0]
        
        self.vars = RobotVars()
        self.vars.blind_milliseconds = self.config["blindnessTimer"]
        self.vars.drive_speed = self.config["speedDrive"]
        self.vars.top_speed = self.config["speedMax"]
        self.vars.target_goal = Goal.Blue if self.config["targetGoal"] == "blue" else Goal.Yellow
        self.vars.maintain_orientation_speed = self.config["maintainOrientationTurnSpeed"]
        self.vars.dribble_speed = self.config["speedDribble"]
        self.vars.cameraOrientation = self.config["cameraOrientation"]
        self.vars.backingDistance = self.config["backingDistance"]
        
        self.vars.initial_headings = [0 for x in self.utils.compasses]
        
        self.speedBias = self.config["speedBias"]
        self.angleCoeff = self.config["anglePolyCoefficients"]
        
        self.corners = self.config["corners"]
        self.front = self.corners["TopLeft"]["front"]
        self.back = "blue" if self.front == "yellow" else "yellow"
        
        tl = compute_pos_from_corner(self.corners["TopLeft"], self.front, self.back)
        tr = compute_pos_from_corner(self.corners["TopRight"], self.front, self.back)
        br = compute_pos_from_corner(self.corners["BottomRight"], self.front, self.back)
        bl = compute_pos_from_corner(self.corners["BottomLeft"], self.front, self.back)
        self.quadrilateral = [tl, tr, bl, br]
        
        
        print(*[(int(x[0]), int(x[1])) for x in self.quadrilateral])
        # ~ f = self.corners["TopLeft"]["yellow"]
        # ~ b = self.corners["TopLeft"]["blue"]
        # ~ self.left = max(self.quadrilateral[0][0], self.quadrilateral[2][0])
        # ~ self.right = min(self.quadrilateral[1][0], self.quadrilateral[3][0])
        # ~ self.width = self.right - self.left
        # ~ self.top = min(self.quadrilateral[0][1], self.quadrilateral[1][1])
        # ~ self.bottom = max(self.quadrilateral[2][1], self.quadrilateral[3][1])
        # ~ self.length = self.top - self.bottom
        # ~ print(self.left, self.right, self.top, self.bottom)
        # ~ print(determine_position(f["angle"], f["dist"], b["angle"], b["dist"], width=self.width, length = self.length))
        
        self.utils.camera.set_masks(self.config["cameraMasks"])
        self.prev: float = now()
        self.dt: float = 1/60
        self.update_interval: float = 0.00001
        
    def calculate_final_direction(self, angle: float, distance: float) -> float:
        def angle_poly(x: float) -> float:
            return (self.angleCoeff["x5"] * pow(x, 5)) + (self.angleCoeff["x4"] * pow(x, 4)) + (self.angleCoeff["x3"] * pow(x, 3)) + (self.angleCoeff["x2"] * pow(x, 2)) + (self.angleCoeff["x1"] * x)

        def f(x) -> float:
            return clamp(-(1/30) * (x - 50), 0, 1)
            
        angle = normalise(angle)
        is_negative: bool = angle < 0
        
        mapped_angle: float = angle_poly(angle) if angle > 0 else -angle_poly(abs(angle))

        return normalise(angle_lerp(angle, mapped_angle, f(distance)))

    def compute_new_direction_vector(self, angle: float, speed: float, oob_factor: float = 2.0):
        pos = [self.bb.approxXPosition, self.bb.yPosition]
        if None in pos:
            return
            
        normal = [0, 0]
        
        delta, t = collide_point_quadrilateral(pos, self.quadrilateral)
        # ~ print(delta)

        mag = 2.0
        # Right
        normal[0] -= abs(delta[0])
        # Left
        normal[0] += abs(delta[1])
        # Top
        normal[1] -= abs(delta[2])
        # Bottom
        normal[1] += abs(delta[3])
            
        if normal == [0, 0]:
            if self.vars.outOfBoundsTicks > 0:
                normal = self.bb.lastNormal.copy()
            else:
                return
        self.bb.lastNormal = normal.copy()
        
        normal_direction = math.degrees(math.atan2(normal[0], normal[1]))
        normal_magnitude = math.sqrt(pow(normal[0], 2) + pow(normal[1], 2))
        global_travel_direction = angle - self.vars.heading
        deltaAngle = normalise(global_travel_direction - normal_direction)
        diffAngle = abs(deltaAngle)
        
        # If they're pointing in the same direction. Also if its REALLY out then just push it out
        if diffAngle <= 90 and normal_magnitude < 10.0: return
        
        # Perpendicular = 1 * original speed, Antiparallel = 0
        t = clamp_lerp(normal_magnitude, 3, 25)
        new_speed = (0.05 + 0.95 * math.sin(math.radians(diffAngle))) * abs(speed)
        # ~ new_speed = lerp(new_speed, 1, t)
        da = lerp(0, 90, t)
        if normalise(deltaAngle + self.vars.heading) > 0:
            new_direction = normal_direction + (90 - da) + self.vars.heading
        else:
            new_direction = normal_direction - (90 - da) + self.vars.heading
        
        return (normalise(new_direction), new_speed)
        
    async def drive_in_direction(self, angle: float, speed: float, contribution: float = 1.0, oob_angle: float = 15.0):
        # First pass for out of bounds with angle checking.
        # ~ if self.utils.camera.blue_angle is not None:
            # ~ ba = self.utils.camera.blue_angle - self.vars.heading
            # ~ bd = self.utils.camera.blue_distance
            # ~ if self.vars.target_goal == Goal.Yellow:
                # ~ angle_range = (self.corners["BottomLeft"]["blue"]["angle"], self.corners["BottomRight"]["blue"]["angle"])
                # ~ if in_range(ba, (0, angle_range[0])) or in_range(ba, (angle_range[1], 0)):
                    # ~ if ba < 0:
                        # ~ angle = clamp(self.utils.camera.blue_angle + 90, -90, -oob_angle)
                    # ~ else:
                        # ~ angle = clamp(self.utils.camera.blue_angle - 90, oob_angle, 90)
            # ~ else:
                # ~ angle_range = (normalise(self.corners["BottomLeft"]["blue"]["angle"] + 180), normalise(self.corners["BottomRight"]["blue"]["angle"] + 180))
                # ~ if not (in_range(ba, angle_range)):
                    # ~ if ba < 0:
                        # ~ angle = clamp(self.utils.camera.blue_angle - 90, -180 + oob_angle, -90)
                    # ~ else:
                        # ~ angle = clamp(self.utils.camera.blue_angle + 90, 90, 180 - oob_angle)
        # ~ elif self.utils.camera.yellow_angle is not None: # and self.utils.camera.blue_angle is None:
            # ~ ta = self.utils.camera.yellow_angle - self.vars.heading
            # ~ td = self.utils.camera.yellow_distance
            # ~ if self.vars.target_goal == Goal.Yellow:
                # ~ angle_range = (self.corners["TopRight"]["yellow"]["angle"], self.corners["TopLeft"]["yellow"]["angle"])
                # ~ if not in_range(ta, angle_range):
                    # ~ if ta < 0:
                        # ~ angle = clamp(self.utils.camera.yellow_angle - 90, -180 + oob_angle, -90)
                    # ~ else:
                        # ~ angle = clamp(self.utils.camera.yellow_angle + 90, 90, 180 - oob_angle)
            # ~ else:
                # ~ angle_range = (normalise(self.corners["TopRight"]["yellow"]["angle"] + 180), normalise(self.corners["TopLeft"]["yellow"]["angle"] + 180))
                # ~ if in_range(ta, (0, angle_range[0])) or in_range(ta, (angle_range[1], 0)):
                    # ~ if ta < 0:
                        # ~ angle = clamp(self.utils.camera.yellow_angle + 90, -90, -oob_angle)
                    # ~ else:
                        # ~ angle = clamp(self.utils.camera.yellow_angle - 90, oob_angle, 90)
        
        # Goal distance checking
        # ~ dist_to_goals = 30.0 # 45.0
        # ~ movement_decided = False
        # ~ if self.utils.camera.yellow_angle:
            # ~ if self.utils.camera.yellow_distance <= dist_to_goals:
                # ~ movement_decided = True
                # ~ deltaAngle = normalise(angle - self.utils.camera.yellow_angle)
                # ~ normal_magnitude = dist_to_goals - self.utils.camera.yellow_distance
                # ~ if abs(deltaAngle) <= 90:
                    # ~ da = lerp(0, 90, clamp_lerp(normal_magnitude, 0.0, 7.0))
                    # ~ if deltaAngle > 0:
                        # ~ angle = self.utils.camera.yellow_angle - 180 - (90 - da)
                    # ~ else:
                        # ~ angle = self.utils.camera.yellow_angle - 180 + (90 - da)
        # ~ if not movement_decided and self.utils.camera.blue_angle:
            # ~ if self.utils.camera.blue_distance <= dist_to_goals:
                # ~ deltaAngle = normalise(angle - self.utils.camera.blue_angle)
                # ~ normal_magnitude = dist_to_goals - self.utils.camera.blue_distance
                # ~ if abs(deltaAngle) <= 90:
                    # ~ da = lerp(0, 90, clamp_lerp(normal_magnitude, 0.0, 7.0))
                    # ~ if deltaAngle > 0:
                        # ~ angle = self.utils.camera.blue_angle - 180 - (90 - da)
                    # ~ else:
                        # ~ angle = self.utils.camera.blue_angle - 180 + (90 - da)
        
        # Second pass for position checking
        new_angle_speed_pair = self.compute_new_direction_vector(angle, speed)
        self.vars.outOfBounds = False
        self.vars.outOfBoundsTicks -= 1
        if new_angle_speed_pair is not None:
            self.vars.outOfBounds = True
            self.vars.outOfBoundsTicks = 5
            angle, speed = new_angle_speed_pair
        # ~ print(new_angle_speed_pair is not None, round(angle), round(speed, 2))
        
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
        
        self.vars.lastSpeed = speed
        
    async def turn(self, speed: float, contribution: float = 1.0):
        for i in range(4):
            self.future_motor_speeds[i] += clamp(speed * contribution, -1, 1)
    async def brake(self):
        self.vars.lastSpeed = 0
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
        
    async def reverse_dribbler(self):
        self.utils.motors['dribbler'].set_speed(1)
    async def enable_dribbler(self):
        self.utils.motors['dribbler'].set_speed(-1)
    async def stop_dribbler(self):
        self.utils.motors['dribbler'].set_speed(0)

    def drive_direction_bias(self, a: float, d: float) -> float:
        a = normalise(a)
        def f(x, B = 76):
            return x * (1 - 1 / (1 + pow(abs(x / B), 3)))
        mapped_angle = 180 * f(a) / f(180)
        return angle_lerp(a, mapped_angle, clamp(-(1/30)*(d - 50), 0, 1))
    def drive_speed_bias(self, a: float, d: float, close = 30, far = 41) -> float:
        a = normalise(a)
        f = 1 - self.speedBias["forwardDamping"] / (1 + pow(0.02 * a, 4))
        g = 1 - 0.5 * self.speedBias["sideDamping"] * (1 - math.cos(2 * math.radians(a)))
        mapped_speed = f * g
        return lerp(mapped_speed, 1 - self.speedBias["forwardDamping"], clamp_lerp(d, close, far))
        
    async def update(self):
        "Logic for the robot gameplay"
        
        span = 27.0 # degrees
        held_distance = 16.0 # cm
        ball_is_at_front = abs(self.vars.normalised_ball_angle) <= span
        view_ball_as_captured = (ball_is_at_front and self.vars.ball_distance < held_distance) if self.vars.ball_distance is not None else 0
        ball_is_close_to_tof = self.vars.frontTofDistance < 9.0
        if (ball_is_close_to_tof and not ball_is_at_front) or view_ball_as_captured:
            self.vars.has_ball = min(500, self.vars.has_ball + self.dt)
        else:
            self.vars.has_ball = max(0, self.vars.has_ball - self.dt)
        
        if self.state != State.Shooting:
            self.bb.isKicking = False
            if self.vars.has_ball > 0:
                await self.enable_dribbler()
            else:
                await self.stop_dribbler()
        
        # ~ self.state = State.Defending
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
                
            # Back up to the goal.
            if self.bb.defendingDistance is not None:
                speed = 0.10 * smooth_linear(self.bb.defendingDistance - self.vars.backingDistance)
                await self.drive_in_direction(self.bb.defendingAngle + self.vars.heading, speed, 0.67)
            else:
                speed = 0.7
                await self.drive_in_direction(180 + self.vars.heading, speed, 0.5)
        
            await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.4)
            
            # Center yourself
            a = normalise(localisation_angle - self.vars.heading + 180)
            speed = 0.5 * smooth_linear((a + 90) % 180 - 90, a = 300)
            if abs(a) >= 90: speed *= -1
            await self.drive_in_direction(self.vars.heading - 90, speed, 0.5)
                
            await self.confirm_drive()
            await self.stop_dribbler()
            self.vars.has_ball = 0
            
        if self.state in [State.Chasing, State.Defending]:
            self.previous_state = self.state
            if not self.bb.isKicking and self.bb.kickedTicks > 0:
                self.bb.kickedTicks = self.bb.kickedTicks - 1
        
        if self.state in [State.Chasing, State.Shooting] and self.is_goalie:
            if self.bb.defendingDistance >= self.bb.returnToGoalThreshold: # or received 'I got this' message from other robot
                self.bb.leavingGoal = False
                self.state = State.Defending
        if self.state == State.Chasing:
            """
            Do normal chase code
            Follow ball polynomial (with prediction of ball velocity?)
            Face north
            """
            
            # Transition from chasing to shooting if ball is held long enough.
            if self.vars.has_ball >= 110:
                self.bb.targetDirection = None
                self.state = State.Shooting
                self.bb.leavingGoal = False
                self.bb.currTurn = 0
                self.bb.currDrive = [0, 0]
                return
            
            # If facing roughly forward
            if abs(self.vars.heading) <= 90:
                # Turn to ball if we are on the edge of field. I got rid of this
                # bc turning the robot messed up the camera localisation
                # ~ if self.vars.outOfBoundsTicks <= 0:
                await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.4)
                # ~ else:
                    # ~ await self.turn(-smooth_linear(self.vars.ball_angle) * 0.05, contribution=0.4)
                    
                direction = self.calculate_final_direction(self.vars.normalised_ball_angle, self.vars.ball_distance)
                direction = self.drive_direction_bias(direction, self.vars.ball_distance)
                speed = self.drive_speed_bias(direction, self.vars.ball_distance) * self.vars.drive_speed
                
                await self.drive_in_direction(direction + self.vars.heading, speed, contribution = 1.0)
            # If backward, i.e. if we lost the ball after trying to shoot.
            else:
                # If the ball is behind us
                if abs(self.vars.ball_angle) > 90:
                    await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.4)
                # if the ball is in front, collect it directly without turning first
                else:
                    await self.turn(-self.vars.ball_angle * 0.05, contribution=0.6)
                    direction = self.calculate_final_direction(self.vars.ball_angle, self.vars.ball_distance)
                    direction = self.drive_direction_bias(direction, self.vars.ball_distance)
                    speed = 0.5 + 0.5 * clamp_lerp(self.vars.ball_distance, 30, 45)
                    await self.drive_in_direction(direction, speed, 1.0)
                
            await self.confirm_drive()
        
        elif self.state == State.Shooting:
            """
            Don't maintain heading.
            update targetDirection to be left or right depending on self.xPosition
            Turn to face targetDirection
            THEN, drive slowly backwards to the goal until self.atAttackingGoal
            Shooting with solenoid
            """
            
            if not self.bb.isKicking:
                await self.enable_dribbler()
                
            # Go back to chasing if the ball is too far away
            ca = self.utils.camera.angle if self.utils.camera.angle else 0
            # ~ if not (abs(ca) <= 45 or self.utils.camera.angle is None) or self.vars.ball_distance >= 21.0:
            if self.vars.has_ball < 75 and self.bb.kickedTicks <= 0:
                self.state = State.Chasing
            
            # ~ # If the goals are not visible, just drive forward
            # ~ if self.bb.attackingAngle is None:
                # ~ await self.drive_in_direction(0, self.vars.dribble_speed, contribution = 1.0)
                # ~ await self.confirm_drive()
                # ~ return
            
            # Turn and drive to the goal.
            # ~ await self.turn(-self.bb.attackingAngle * .0067, contribution=0.8)
            # ~ if abs(self.bb.attackingAngle) < 25:
                # ~ await self.drive_in_direction(0, self.vars.dribble_speed, contribution = 1.0)
            # ~ else:
                # ~ await self.drive_in_direction(0, 0.1, contribution = 1.0)
            # ~ await self.confirm_drive()
            # ~ return
            
            if self.bb.capturedSpeed is not None:
                self.bb.capturedSpeed *= .950
                
            # If the robot is facing forward
            if abs(self.vars.heading) <= 90:
                if self.bb.targetDirection is None:
                    if self.bb.xPosition < 0: # left side of field
                        self.bb.targetDirection = 90
                    else: # right side of field
                        self.bb.targetDirection = -90
                    self.bb.previousTargetDirection = self.bb.targetDirection
                    self.bb.capturedSpeed = self.vars.lastSpeed
                    self.bb.atAttackingGoalTicks = 0
                    self.bb.isKicking = False
            # If facing backwards, like when it just collected the ball after it dropped it
            else:
                self.bb.targetDirection = self.bb.previousTargetDirection
                    
            attackingAngle = self.bb.meanAttackingAngle
            attackingDistance = self.bb.meanAttackingDistance
            
            t = clamp_lerp(attackingDistance, 120, 67)
            goalGlobalAngle = attackingAngle - self.vars.heading
            backwardsAngle = normalise(goalGlobalAngle + 100 * sign(self.bb.targetDirection))
            newAngle = angle_lerp(self.bb.targetDirection, backwardsAngle, t * max(0, 0.3 - self.bb.capturedSpeed) / 0.3)
            # ~ newAngle = angle_lerp(newAngle, 0, clamp_lerp(attackingDistance, 100, 67))
            deltaAngle = normalise(self.vars.heading - newAngle)
            
            TF = 0.005
            LF = 0.5
            
            # Drives and lerps the speed down so that the ball can be captured
            if abs(deltaAngle) >= 30 and self.bb.atAttackingGoalTicks <= 0 and self.bb.capturedSpeed >= 0.10:
                self.bb.currTurn = angle_lerp(self.bb.currTurn, -deltaAngle * TF, LF)
                await self.turn(self.bb.currTurn, 0.8)
                self.bb.currDrive[0] = angle_lerp(self.bb.currDrive[0], 0 + self.vars.heading * 1.0, LF)
                await self.drive_in_direction(self.bb.currDrive[0], self.bb.capturedSpeed, contribution = 1.0)
                await self.confirm_drive()
                return
            
            # Drive along the wing
            if not self.bb.atAttackingGoal:
                self.bb.atAttackingGoalTicks -= 1
            else:
                self.bb.atAttackingGoalTicks = 30
            if self.bb.atAttackingGoalTicks <= 0:
                self.bb.currTurn = angle_lerp(self.bb.currTurn, -deltaAngle * TF, LF)
                await self.turn(self.bb.currTurn, 0.8)
                angle = angle_lerp(self.vars.heading + 0, attackingAngle, t = clamp_lerp(attackingDistance, 43, 27))
                self.bb.currDrive[0] = angle_lerp(self.bb.currDrive[0], angle, LF)
                self.bb.currDrive[1] = lerp(self.bb.currDrive[1], 0.5, LF)
                await self.drive_in_direction(*self.bb.currDrive, 0.8)
                await self.confirm_drive()
                return
            
            # Strafe if its outside the angle range.
            # ~ if abs(goalGlobalAngle) > 45:
                # ~ if goalGlobalAngle < 0:
                    # ~ angle = attackingAngle + 90
                # ~ else:
                    # ~ angle = attackingAngle - 90
                # ~ await self.turn(-normalise(attackingAngle + 180) * TF, 0.75)
                # ~ await self.drive_in_direction(angle, 0.35, 1.0)
                # ~ await self.confirm_drive()
                # ~ return
            
            # Drive/kick towards the goal.
            self.bb.currTurn = angle_lerp(self.bb.currTurn, -TF * abs(attackingAngle) * sign(self.bb.targetDirection), LF)
            await self.turn(self.bb.currTurn, 0.67)
            i = self.bb.pastAttackingAngles[-1]
            if self.bb.isKicking or abs(self.bb.pastAttackingAngles[i]) <= 10.0:
                self.bb.isKicking = True
                await self.reverse_dribbler()
                self.bb.kickedTicks += 1
                
                if self.bb.kickedTicks <= 17:
                    await self.drive_in_direction(0, 4.167, 1.0)
                elif self.bb.kickedTicks <= 34:
                    await self.drive_in_direction(180 + self.vars.heading, 0.5, 1.0)
                else:
                    self.state = State.Defending
                
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
            
            # If goals not seen, spin.
            if localisation_angle is None:
                await self.turn(0.167, 1.0)
                await self.confirm_drive()
                return
                
            # Back up to the goal.
            if self.bb.defendingDistance is not None:
                speed = 0.05 * smooth_linear(self.bb.defendingDistance - self.vars.backingDistance, a = 600)
                await self.drive_in_direction(self.bb.defendingAngle + self.vars.heading, speed, 0.5)
            else:
                speed = self.vars.drive_speed
                await self.drive_in_direction(180 + self.vars.heading, speed, 0.5)
            
            
            # If ball not seen
            if self.utils.camera.angle is None:
                # Center yourself
                a = normalise(localisation_angle - self.vars.heading + 180)
                speed = 0.5 * smooth_linear((a + 90) % 180 - 90, a = 670)
                if abs(a) >= 90: speed *= -1
                await self.drive_in_direction(self.vars.heading - 90, speed, 0.5)
                await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.67)
                
            else:
                # Position between ball and goal
                a = normalise(self.utils.camera.angle - self.vars.heading)
                speed = 0.5 * smooth_linear((a + 90) % 180 - 90, a = 670)
                if abs(a) >= 90: speed *= -1
                await self.drive_in_direction(self.vars.heading + 90, speed, 0.5)
                await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.67)
                # ~ await self.turn(-normalise(self.bb.lastDefendingAngle - 180) * 0.04, contribution=0.5)
            
                # This will be for the goalie
                if self.bb.kickedTicks <= 0:
                    if self.bb.leavingGoal:
                        self.state = State.Chasing
                        await self.confirm_drive()
                        return
                    if abs(a) > 110 or (self.vars.ball_distance <= 30.0 and abs(self.vars.ball_angle) <= 25.0):
                        self.bb.leavingGoal = True
                        self.state = State.Chasing
                        await self.confirm_drive()
                        return
                    self.bb.leavingGoal = False
                
            await self.confirm_drive()
            await self.stop_dribbler()
                
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
        if self.vars.lastSpeed == 0: return
        
        await self.brake()
        await self.stop_dribbler()
    
    def get_orientation(self) -> float:
        angles = [normalise(compass.read() - self.vars.initial_headings[i]) for i, compass in enumerate(self.utils.compasses) if compass.enabled]
        if len(angles) == 0: return None
        return sum(angles) / len(angles)
        
    async def calibrate(self):
        "Set initial heading"
        if self.vars.lastSpeed != 0:
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
        
    async def start(self):
        
        await self.brake()
        await self.stop_dribbler()
        self.utils.camera.debug = True
        
        await asyncio.sleep(0.67)
        
        center = np.array(self.utils.camera.center.copy(), dtype=np.uint32)
        
        def arrow(origin: np.array, degrees: float, radius: float) -> tuple:
            if type(origin) != np.array: origin = np.array(origin, np.uint32)
            
            a = math.radians(degrees)
            return (origin, origin + np.array([radius * math.cos(a), radius * math.sin(a)], dtype=np.int32))
        
        def i2c_loop():
            while True:
                self.vars.frontTofDistance = self.utils.captureTof.read()
                self.vars.tof_distances = self.utils.tofs.read()
                self.vars.heading = self.get_orientation()
                
                # Calculate xPosition
                OFFSET = 3.0
                A = math.radians(35)
                H = math.radians(self.vars.heading)
                x1 = -math.cos(H - A) * max(0, self.vars.tof_distances[0] - OFFSET)
                x2 = -math.cos(H + A) * max(0, self.vars.tof_distances[1] - OFFSET)
                x3 = math.cos(H - A) * max(0, self.vars.tof_distances[2] - OFFSET)
                x4 = math.cos(H + A) * max(0, self.vars.tof_distances[3] - OFFSET)
                self.bb.xPosition = -min(x1, x2) - max(x3, x4)
                
                sleep(0.01)
                
        Thread(target=i2c_loop, daemon=True).start()
        
        calibration_count = 41
        
        monitor = None; wx = wy = 0
        monitors = get_monitors()
        if self.utils.camera.debug and len(monitors) > 0:
            monitor = monitors[0]
            wx, wy = monitor.width - self.utils.camera.size[0] - 5, (monitor.height - self.utils.camera.size[1]) // 2
        
        # Sync up the robot event loop with the camera output
        async for raw_frame in self.utils.camera.main():

            curr = now()
            self.dt = (curr - self.prev) * 1000
            self.prev = curr
                
            frame = self.utils.camera.process(raw_frame)
            
            if self.utils.camera.debug and (self.utils.camera.ticks % 8) == 0 and monitor is not None:
                
                if self.vars.ball_angle is not None:
                    direction = self.calculate_final_direction(self.vars.normalised_ball_angle, self.vars.ball_distance)
                    cv2.line(frame, *arrow(center, direction, 150), [255, 255, 255], 3)
                cv2.line(frame, *arrow(center, self.vars.heading, 50), [150, 150, 255], 5)
                
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                y = 0
                v = {
                    "State": self.state,
                    # ~ "atDefenderGoal": self.bb.atDefenderGoal,
                    # ~ "atAttackingGoal": self.bb.atAttackingGoal,
                    "atAttackingGoalTicks": self.bb.atAttackingGoalTicks,
                    # ~ "goalTicks": self.bb.goalTicks,
                    "xPosition": self.bb.approxXPosition,
                    "yPosition": self.bb.yPosition,
                    # ~ "captureTof": self.vars.frontTofDistance,
                    # ~ "has_ball": self.vars.has_ball,
                    # ~ "kickoffTimer": self.bb.kickoffTimer,
                    "targetDirection": self.bb.targetDirection,
                    "targetGoal": self.vars.target_goal,
                    # ~ "ballAngle": self.vars.ball_angle,
                    # ~ "ballDistance": self.vars.ball_distance,
                    # ~ "attackDistance": self.bb.attackingDistance,
                    # ~ "defendDistance": self.bb.defendingDistance,
                    # ~ "global attackGoalAngle": self.bb.attackingAngle - self.vars.heading if self.bb.attackingAngle else None,
                    # ~ "global defendGoalAngle": self.bb.defendingAngle - self.vars.heading if self.bb.defendingAngle else None,
                    # ~ "attackGoalAngle": self.bb.lastAttackingAngle,
                    # ~ "capturedSpeed": self.bb.capturedSpeed
                    "position": self.vars.position,
                    "lastSpeed": self.vars.lastSpeed,
                    # ~ "self.bb.currTurn": self.bb.currTurn,
                    "lastNormal": self.bb.lastNormal,
                    "orientation": self.vars.heading,
                    "currDrive": self.bb.currDrive,
                    "OOB": self.vars.outOfBounds
                }
                for key, val in v.items():
                    cv2.putText(frame, f"{key}: {val}", (10, 30 + y * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA, False)
                    y += 1
                cv2.imshow("Camera", frame)
                cv2.setWindowTitle("Camera", f"Camera FPS: {self.utils.camera.fps}")
                cv2.moveWindow("Camera", wx, wy)
                cv2.waitKey(1)
            
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
                self.vars.ball_angle = normalise(self.utils.camera.angle) #Q+ self.vars.cameraOrientation)
                self.vars.ball_distance = self.utils.camera.distance
            
            # Setting blackboard variables
            if self.vars.ball_angle:
                self.vars.normalised_ball_angle = normalise(self.vars.ball_angle - self.vars.heading)
            if self.utils.camera.yellow_angle is not None:
                self.bb.lastYellowAngle = self.utils.camera.yellow_angle
                self.bb.lastYellowDistance = self.utils.camera.yellow_distance
                self.vars.lastGlobalYellowAngle = normalise(self.utils.camera.yellow_angle - self.vars.heading)
                self.vars.lastGlobalYellowDistance = self.utils.camera.yellow_distance
            if self.utils.camera.blue_angle is not None:
                self.bb.lastBlueAngle = self.utils.camera.blue_angle
                self.bb.lastBlueDistance = self.utils.camera.blue_distance
                self.vars.lastGlobalBlueAngle = normalise(self.utils.camera.blue_angle - self.vars.heading)
                self.vars.lastGlobalBlueDistance = self.utils.camera.blue_distance
            if self.vars.target_goal == Goal.Yellow:
                self.bb.attackingAngle = self.utils.camera.yellow_angle
                self.bb.attackingDistance = self.utils.camera.yellow_distance
                self.bb.defendingAngle = self.utils.camera.blue_angle
                self.bb.defendingDistance = self.utils.camera.blue_distance
            else:
                self.bb.attackingAngle = self.utils.camera.blue_angle
                self.bb.attackingDistance = self.utils.camera.blue_distance
                self.bb.defendingAngle = self.utils.camera.yellow_angle
                self.bb.defendingDistance = self.utils.camera.yellow_distance
            if self.bb.attackingAngle is not None:
                self.bb.lastAttackingAngle = normalise(self.bb.attackingAngle)
                self.bb.atAttackingGoal = bool(abs(normalise(self.bb.attackingAngle - self.vars.heading)) >= 55.0)
                i = self.bb.pastAttackingAngles[-1]
                self.bb.pastAttackingAngles[i] = self.bb.attackingAngle
                self.bb.pastAttackingAngles[-1] = (i + 1) % NUM_SAMPLES
                self.bb.meanAttackingAngle = np.mean(self.bb.pastAttackingAngles[:NUM_SAMPLES])
            if self.bb.attackingDistance is not None:
                self.bb.atAttackingGoal = self.bb.atAttackingGoal or bool(self.bb.attackingDistance < 45.0)
                i = self.bb.pastAttackingDistances[-1]
                self.bb.pastAttackingDistances[i] = self.bb.attackingDistance
                self.bb.pastAttackingDistances[-1] = (i + 1) % NUM_SAMPLES
                self.bb.meanAttackingDistance = np.mean(self.bb.pastAttackingDistances[:NUM_SAMPLES])
            if self.bb.defendingAngle is not None:
                self.bb.lastDefendingAngle = self.bb.defendingAngle
            if self.bb.defendingDistance is not None:
                self.bb.atDefenderGoal = bool(self.bb.defendingDistance <= 45.0)
            
            # Camera localisation
            if None not in [self.bb.lastBlueDistance, self.bb.lastYellowDistance]:
                if self.front == "yellow":
                    a1, a2 = self.bb.lastBlueAngle - self.vars.heading, self.bb.lastYellowAngle - self.vars.heading
                    d1, d2 = self.bb.lastBlueDistance, self.bb.lastYellowDistance
                else:
                    a1, a2 = self.bb.lastYellowAngle - self.vars.heading, self.bb.lastBlueAngle - self.vars.heading
                    d1, d2 = self.bb.lastYellowDistance, self.bb.lastBlueDistance
                self.bb.yPosition = y_from_distances(d1, d2)
                self.bb.approxXPosition = calculate_x(self.bb.yPosition, a1, a2)
            else:
                ...
                # ~ # CHANGE ThIS TO USE THE LAST DISTANCE SEEN
                # ~ if d1 is None and d2 is None:
                    # ~ ...
                # ~ elif d1 is None:
                    # ~ self.bb.yPosition = 67 - d2 * math.cos(math.radians(self.bb.attackingAngle))
                # ~ else:
                    # ~ self.bb.yPosition = -94 - d1 * math.cos(math.radians(self.bb.defendingAngle))
            
                
            # I do this so that when it starts up it doesnt just sprint away
            if calibration_count > 0:
                await self.calibrate()
                calibration_count -= 1
                continue
        
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
            


if __name__ == "__main__":
    
    loop = asyncio.get_event_loop()
    
    ts = Robot()
    loop.create_task(ts.start())
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        sleep(0.1)
        
        # Kill motors once stopped by user
        for index in ts.utils.motors:
            motor = ts.utils.motors[index]
            motor.set_speed(0)
            
        print("Robot stopped by user")
    except Exception as e:
        print(e)
    
    
