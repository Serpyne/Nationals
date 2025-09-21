"""

Main TS robot code; I plan to use this file in competitions.

"""

from tof import TOF, TOFChain
from compass import Compass
from motors_i2c import Motor
from cam import AsyncCam, normalise, lerp, angle_lerp, clamp, sign
from solenoid import Solenoid

from concurrent.futures import ProcessPoolExecutor

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

# ~ def get_goal_vectors(top: list, bottom: list) -> list[float, float]:
    # ~ if type(top) == dict: top = [top["angle"], top["dist"]]
    # ~ if type(bottom) == dict: bottom = [bottom["angle"], bottom["dist"]]
    # ~ ta, td = math.radians(top[0]), top[0]
    # ~ ba, bd = math.radians(bottom[0]), bottom[0]
    
    # ~ x_top = td * math.sin(ta)
    # ~ x_bottom = bd * math.sin(ba)
    # ~ y_top = td * math.cos(ta)
    # ~ y_bottom = bd * math.cos(ba)
    
    # ~ return (x_top, x_bottom, y_top, y_bottom)
    
# ~ def in_cassini_oval(d1: float, d2: float, half_width: float = 90.5):
    # ~ return d1 * d2 < 2 * pow(half_width, 2)

def calculate_x(y_hit: float, a1: float, a2: float):
    x_hits = [0, 0]
    a1 -= 90
    a2 -= 90
    if a1 % 180 != 0:
        "y = tan(a1)x - 243/2"
        x_hits[0] = ( y_hit + 243 / 2 ) / math.tan(math.radians(a1))
    else:
        x_hits[0] = 0
    if a2 % 180 != 0:
        "y = tan(a2)x + 243/2"
        x_hits[1] = ( y_hit - 243 / 2 ) / math.tan(math.radians(a2))
    else:
        x_hits[1] = 0
    return -np.mean(x_hits)
def y_from_distances(d1, d2, half_h = 243 / 2) -> float:
    return half_h * (d1 - d2) / (d1 + d2)

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
    defendingDistance: float = None
    cameraOrientation: float = None
    capturedSpeed: float = 0
    isKicking: bool = False
    kickedTicks: int = 0
    currTurn: float = 0
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
        self.state: str = State.Chasing
        self.previous_state: str = State.Chasing
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
        front = self.corners["TopLeft"]["front"]
        back = "blue" if front == "yellow" else "yellow"
        self.cornerFrontDirection = Goal.Yellow if front == "yellow" else Goal.Blue
        
        # ~ lengths = []
        # ~ x_top, x_bottom, y_top, y_bottom = get_goal_vectors(top=self.corners["TopLeft"]["yellow"], bottom=self.corners["TopLeft"]["blue"])
        # ~ self.left = -0.5 * abs(x_top + x_bottom)
        # ~ print(x_top, x_bottom, y_top, y_bottom)
        # ~ lengths.append(abs(y_top - y_bottom))
        # ~ x_top, x_bottom, y_top, y_bottom = get_goal_vectors(top=self.corners["TopRight"]["yellow"], bottom=self.corners["TopRight"]["blue"])
        # ~ self.right = 0.5 * abs(x_top + x_bottom)
        # ~ print(x_top, x_bottom, y_top, y_bottom)
        # ~ lengths.append(abs(y_top - y_bottom))
        # ~ x_top, x_bottom, y_top, y_bottom = get_goal_vectors(top=self.corners["BottomLeft"]["yellow"], bottom=self.corners["BottomLeft"]["blue"])
        # ~ self.left = max(self.left, -0.5 * abs(x_top + x_bottom))
        # ~ print(x_top, x_bottom, y_top, y_bottom)
        # ~ lengths.append(abs(y_top - y_bottom))
        # ~ x_top, x_bottom, y_top, y_bottom = get_goal_vectors(top=self.corners["BottomRight"]["yellow"], bottom=self.corners["BottomRight"]["blue"])
        # ~ self.right = min(self.right, 0.5 * abs(x_top + x_bottom))
        # ~ print(x_top, x_bottom, y_top, y_bottom)
        # ~ lengths.append(abs(y_top - y_bottom))
        
        # ~ self.width = self.right - self.left
        # ~ self.length = np.mean(lengths)
        # ~ print(self.left, self.right, self.width, self.length)
        
        tl = self.corners["TopLeft"]
        y_hit = y_from_distances(tl[back]["dist"], tl[front]["dist"])
        x_hit = calculate_x(y_hit, tl[back]["angle"], tl[front]["angle"])
        print(x_hit, y_hit)
        
        self.utils.camera.set_masks(self.config["cameraMasks"])
        self.prev: float = now()
        self.dt: float = 1/60
        self.update_interval: float = 0.00001
        
    # ~ def determine_position(self, overcompensation_factor = 1.1) -> tuple:
        # ~ if None in [self.vars.lastGlobalYellowAngle, self.vars.lastGlobalBlueAngle]: return None
        # ~ ta = self.vars.lastGlobalYellowAngle + self.vars.heading
        # ~ td = self.vars.lastGlobalYellowDistance
        # ~ ba = self.vars.lastGlobalBlueAngle + self.vars.heading
        # ~ bd = self.vars.lastGlobalBlueDistance
        # ~ if ta is None and ba is None: return None
        # ~ if ta is None:
            # ~ a = math.radians(ba)
            # ~ x = bd * math.sin(a)
            # ~ y = 0.5 * self.length - bd * math.cos(a)
            # ~ if x < 0: rx = x / (-self.left)
            # ~ else: rx = x / self.right
            # ~ return (overcompensation_factor * rx, overcompensation_factor * 2 * y / self.length)
        # ~ if ba is None:
            # ~ a = math.radians(ta)
            # ~ x = td * math.sin(a)
            # ~ y =  0.5 * self.length - td * math.cos(a)
            # ~ if x < 0: rx = x / (-self.left)
            # ~ else: rx = x / self.right
            # ~ return (overcompensation_factor * rx, overcompensation_factor * 2 * y / self.length)
        # ~ x_top, x_bottom, y_top, y_bottom = get_goal_vectors(top=(ta - self.vars.heading, td), bottom=(ba - self.vars.heading, bd))
        # ~ x = (x_top + x_bottom)
        # ~ y = -(y_top + y_bottom)
        # ~ print((ta - self.vars.heading, td), (ba - self.vars.heading, bd), *[int(x) for x in (x_top, x_bottom, y_top, y_bottom)])
        # ~ return (x / self.width, y / self.length)
        
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
        if self.vars.position is None:
            return
            
        normal_direction = 0
        normal_magnitude = 0
        
        if self.vars.position[0] <= -1.0:
            normal_direction = 90
            normal_magnitude = -self.vars.position[0] - 1.0
        elif self.vars.position[0] >= 1.0:
            normal_direction = -90
            normal_magnitude = self.vars.position[0] - 1.0
        # ~ if self.vars.position[1] >= 1.0:
            # ~ normal_direction = 180
            # ~ normal_magnitude = self.vars.position[1] - 1.0
        # ~ elif self.vars.position[1] <= -1.0:
            # ~ normal_direction = 180
            # ~ normal_magnitude = -self.vars.position[1] - 1.0
            
        if normal_direction == 0: return
        
        normal_magnitude *= oob_factor
        global_travel_direction = angle - self.vars.heading
        deltaAngle = normalise(global_travel_direction - normal_direction)
        diffAngle = abs(deltaAngle)
        
        # If they're pointing in the same direction
        if diffAngle <= 90 and abs(normal_magnitude) < 0.25: return
        
        # Perpendicular = 1 * original speed, Antiparallel = 0
        new_speed = (0.15 + 0.85 * math.sin(math.radians(diffAngle))) * speed
        da = lerp(0, 85, clamp(normal_magnitude, 0, 1))
        if normalise(deltaAngle + self.vars.heading) > 0:
            new_direction = normal_direction + (90 - da) + self.vars.heading
        else:
            new_direction = normal_direction - (90 - da) + self.vars.heading
        
        return (new_direction, new_speed)
        
    async def drive_in_direction(self, angle: float, speed: float, contribution: float = 1.0, oob_angle: float = 15.0):
        # First pass for out of bounds with angle checking.
        if self.utils.camera.blue_angle is not None:
            ba = self.utils.camera.blue_angle - self.vars.heading
            bd = self.utils.camera.blue_distance
            if self.vars.target_goal == Goal.Yellow:
                angle_range = (self.corners["BottomLeft"]["blue"]["angle"], self.corners["BottomRight"]["blue"]["angle"])
                if in_range(ba, (0, angle_range[0])) or in_range(ba, (angle_range[1], 0)):
                    if ba < 0:
                        angle = clamp(self.utils.camera.blue_angle + 90, -90, -oob_angle)
                    else:
                        angle = clamp(self.utils.camera.blue_angle - 90, oob_angle, 90)
            else:
                angle_range = (normalise(self.corners["BottomLeft"]["blue"]["angle"] + 180), normalise(self.corners["BottomRight"]["blue"]["angle"] + 180))
                if not (in_range(ba, angle_range)):
                    if ba < 0:
                        angle = clamp(self.utils.camera.blue_angle - 90, -180 + oob_angle, -90)
                    else:
                        angle = clamp(self.utils.camera.blue_angle + 90, 90, 180 - oob_angle)
        elif self.utils.camera.yellow_angle is not None: # and self.utils.camera.blue_angle is None:
            ta = self.utils.camera.yellow_angle - self.vars.heading
            td = self.utils.camera.yellow_distance
            if self.vars.target_goal == Goal.Yellow:
                angle_range = (self.corners["TopRight"]["yellow"]["angle"], self.corners["TopLeft"]["yellow"]["angle"])
                if not in_range(ta, angle_range):
                    if ta < 0:
                        angle = clamp(self.utils.camera.yellow_angle - 90, -180 + oob_angle, -90)
                    else:
                        angle = clamp(self.utils.camera.yellow_angle + 90, 90, 180 - oob_angle)
            else:
                angle_range = (normalise(self.corners["TopRight"]["yellow"]["angle"] + 180), normalise(self.corners["TopLeft"]["yellow"]["angle"] + 180))
                if in_range(ta, (0, angle_range[0])) or in_range(ta, (angle_range[1], 0)):
                    if ta < 0:
                        angle = clamp(self.utils.camera.yellow_angle + 90, -90, -oob_angle)
                    else:
                        angle = clamp(self.utils.camera.yellow_angle - 90, oob_angle, 90)
        
        # Second pass for position checking
        new_angle_speed_pair = self.compute_new_direction_vector(angle, speed)
        self.vars.outOfBounds = False
        self.vars.outOfBoundsTicks -= 1
        if new_angle_speed_pair is not None:
            self.vars.outOfBounds = True
            self.vars.outOfBoundsTicks = 30
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
            
        # ~ self.future_motor_speeds[0] += FL * contribution
        # ~ self.future_motor_speeds[1] += FR * contribution
        # ~ self.future_motor_speeds[2] += -FL * contribution
        # ~ self.future_motor_speeds[3] += -FR * contribution
        
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
        # ~ print([int(100 * x) for x in self.future_motor_speeds])
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
                speed = 0.05 * smooth_linear(self.bb.defendingDistance - self.vars.backingDistance)
                await self.drive_in_direction(self.bb.defendingAngle + self.vars.heading, speed, 0.5)
            else:
                # ~ speed = 0.25 * smooth_linear(self.bb.yPosition)
                speed = 0.7
                await self.drive_in_direction(180 + self.vars.heading, speed, 0.9)
        
            await self.turn(-self.vars.heading * self.vars.maintain_orientation_speed, contribution=0.4)
            
            # Center yourself
            a = normalise(localisation_angle - self.vars.heading + 180)
            speed = smooth_linear((a + 90) % 180 - 90, 1000)
            if abs(a) >= 90: speed *= -1
            await self.drive_in_direction(self.vars.heading - 90, 0.1 * speed, 0.2)
                
            await self.confirm_drive()
            await self.stop_dribbler()
            self.vars.has_ball = 0
            
        if self.state in [State.Chasing, State.Defending]:
            self.previous_state = self.state
        
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
                self.bb.currTurn = 0
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
            if self.vars.has_ball < 75:
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
                    self.bb.kickedTicks = 0
            # If facing backwards, like when it just collected the ball after it dropped it
            else:
                self.bb.targetDirection = self.bb.previousTargetDirection
                    
            attackingAngle = self.bb.meanAttackingAngle
            attackingDistance = self.bb.meanAttackingDistance
            
            t = clamp_lerp(attackingDistance, 120, 67)
            goalGlobalAngle = attackingAngle - self.vars.heading
            backwardsAngle = normalise(goalGlobalAngle + 100 * sign(self.bb.targetDirection))
            newAngle = angle_lerp(self.bb.targetDirection, backwardsAngle, t * max(0, 0.3 - self.bb.capturedSpeed) / 0.3)
            newAngle = angle_lerp(newAngle, 0, clamp_lerp(attackingDistance, 100, 67))
            deltaAngle = normalise(self.vars.heading - newAngle)
            
            TF = 0.005
            LF = 0.3
            
            # Drives and lerps the speed down so that the ball can be captured
            if abs(deltaAngle) >= 30 and self.bb.atAttackingGoalTicks <= 0 and self.bb.capturedSpeed >= 0.10:
                self.bb.currTurn = angle_lerp(self.bb.currTurn, -deltaAngle * TF, LF)
                await self.turn(self.bb.currTurn, 0.8)
                await self.drive_in_direction(0 + self.vars.heading * 1.0, self.bb.capturedSpeed, contribution = 1.0)
                await self.confirm_drive()
                return
            
            # Drive along the wing
            if not self.bb.atAttackingGoal:
                self.bb.atAttackingGoalTicks -= 1
            else:
                self.bb.atAttackingGoalTicks = 30
            if self.bb.atAttackingGoalTicks <= 0:
                self.bb.currTurn = angle_lerp(self.bb.currTurn, -deltaAngle * TF, LF)
                await self.turn(self.bb.currTurn, 0.3)
                angle = angle_lerp(self.vars.heading + 0, attackingAngle, t = clamp_lerp(attackingDistance, 43, 27))
                await self.drive_in_direction(angle, 0.5, 0.8)
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
            await self.turn(self.bb.currTurn, 0.65)
            i = self.bb.pastAttackingAngles[-1]
            if abs(self.bb.pastAttackingAngles[i]) <= 15:
                self.bb.isKicking = True
                await self.reverse_dribbler()
                self.bb.kickedTicks += 1
                
                if self.bb.kickedTicks <= 17:
                    await self.drive_in_direction(0, 4.1, 1.0)
                else:
                    await self.drive_in_direction(0, -0.5, 1.0)
                
            await self.confirm_drive()
                
        elif self.state == State.Defending:
            """
            Go to goal until atDefenderGoal is true
            oscillate on the goalline (proportional to self.goalTicks).
            Face towards ball/northward.
            """
            
            # This will be for the goalie
            if self.is_goalie:
                if self.bb.leavingGoal:
                    self.state == State.Chasing
                    return
                if self.vars.ball_distance <= 40.0 and abs(self.vars.ball_angle) <= 50.0 and self.atDefenderGoal:
                    self.bb.leavingGoal = True
                    self.state == State.Chasing
                    return
                self.bb.leavingGoal = False
                
                
            self.bb.goalTicks += 1
            
            if self.vars.target_goal == Goal.Blue:
                backing_distance = self.utils.camera.yellow_distance
                localisation_angle = self.utils.camera.yellow_angle
            else:
                backing_distance = self.utils.camera.blue_distance
                localisation_angle = self.utils.camera.blue_angle
            
            # Back up to the goal.
            if backing_distance is not None:
                await self.drive_in_direction(180, 0.5 * smooth_linear(backing_distance - self.vars.backingDistance), 1.0)
        
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
        
        # Sync up the robot event loop with the camera output
        async for raw_frame in self.utils.camera.main():

            curr = now()
            self.dt = (curr - self.prev) * 1000
            self.prev = curr
                
            frame = self.utils.camera.process(raw_frame)
            
            if self.utils.camera.debug and self.utils.camera.ticks % 8 == 0:
                
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
                    # ~ "cornerFrontDirection": self.cornerFrontDirection,
                    "position": self.vars.position,
                    "lastSpeed": self.vars.lastSpeed,
                    "OOB": self.vars.outOfBounds
                }
                for key, val in v.items():
                    cv2.putText(frame, f"{key}: {val}", (10, 30 + y * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA, False)
                    y += 1
                cv2.imshow("Camera", frame)
                cv2.setWindowTitle("Camera", f"Camera FPS: {self.utils.camera.fps}")
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
                self.vars.lastGlobalYellowAngle = normalise(self.utils.camera.yellow_angle - self.vars.heading)
                self.vars.lastGlobalYellowDistance = self.utils.camera.yellow_distance
            if self.utils.camera.blue_angle is not None:
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
                self.bb.atAttackingGoal = self.bb.atAttackingGoal or bool(self.bb.attackingDistance < 32.0)
                i = self.bb.pastAttackingDistances[-1]
                self.bb.pastAttackingDistances[i] = self.bb.attackingDistance
                self.bb.pastAttackingDistances[-1] = (i + 1) % NUM_SAMPLES
                self.bb.meanAttackingDistance = np.mean(self.bb.pastAttackingDistances[:NUM_SAMPLES])
            if self.bb.defendingDistance is not None:
                self.bb.atDefenderGoal = bool(self.bb.defendingDistance <= 45.0)
            
            # Camera localisation
            d1, d2 = self.bb.defendingDistance, self.bb.attackingDistance
            if None not in [d1, d2]:
                self.bb.yPosition = y_from_distances(d1, d2)
                a1, a2 = self.bb.defendingAngle - self.vars.heading, self.bb.attackingAngle - self.vars.heading
                self.bb.approxXPosition = calculate_x(self.bb.yPosition, a1, a2)
            else:
                if d1 is None and d2 is None:
                    ...
                elif d1 is None:
                    self.bb.yPosition = 67 - d2 * math.cos(math.radians(self.bb.attackingAngle))
                else:
                    self.bb.yPosition = -94 - d1 * math.cos(math.radians(self.bb.defendingAngle))
            # ~ self.vars.position = self.determine_position()
            
                
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
    
    
