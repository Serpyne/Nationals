"""

Main TS robot code; I plan to use this file in competitions.

"""

from tof import TOF, TOFChain
from compass import Compass
from motors_i2c import Motor
from cam import AsyncCam, normalise, lerp, angle_lerp, clamp, sign
from solenoid import Solenoid
from bt import BT, Message, STATES

from screeninfo import get_monitors

from gpiozero import Button, LED
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

MAXROT = 0.9

def clamp_lerp(x, a, b):
    return clamp((x - a) / (b - a), 0, 1)
        
def smooth_linear(x: float, a = 674.1) -> float:
    return pow(x, 3) / (a + pow(x, 2))
    
def in_range(x, x_range) -> bool:
    if x_range[0] > x_range[1]:
        x_range = (x_range[1], x_range[0])
    return x_range[0] <= x and x <= x_range[1]

# ~ a_g = pow(10, -30.5)
# ~ b_g = pow(10, 28.7)
def out_of_bounds_function(d_back, d_front, box_width = 182.0, field_length = 183.0):
    # ~ def g(x):
        # ~ return a_g * (pow(x, 16) + b_g)
    return max(0, d_back * d_front - (pow(box_width / 2, 2) + pow(field_length / 2, 2)))

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
    tof_distances: list[float] = [float("inf") for _ in range(4)]
    frontTofDistance: float = float("inf")
    
    damaged: bool = False
    
    blind_milliseconds: int = 670
    target_goal: int = Goal.Yellow
    drive_speed: float = 0.6
    top_speed: float = 0.8
    dribble_speed: float = 0.5
    maintain_orientation_speed: float = 1 / 67
    mode: int = Mode.Idle
    center_speed: float = 0.41
    lastSpeed: float = 0
    backingDistance: float = 30.0
    outOfBounds: bool = False
    outOfBoundsTicks: int = 0
    hasUnstalled: bool = False
class Utilities:
    motors = None
    camera = None
    compasses = None
    tofs = None
    switch_left: Button = None
    switch_right: Button = None
    captureTof = None
    solenoid = None
    bt = None
class State:
    Chasing = "Chasing"
    Defending = "Defending"
    Shooting = "Shooting"
    Stalled = "Stalled"
    KickOff = "KickOff"
    Blind = "Blind"
NUM_SAMPLES = 2
class ShootingStyle:
    HideBall = 0
    Clear = 1
    Flick = 2
    MoveToSide = 3
class Blackboard:        
    atDefenderGoal: bool = False
    atAttackingGoal: bool = False
    atAttackingGoalTicks: int = 0
    goalTicks: int = 0
    xPositionTOF: float = None
    xPosition: float = None
    yPosition: float = None
    kickoffDuration: float = 2000
    kickoffTimer: float = 0
    targetDirection: float = None
    previousTargetDirection: float = 180
    leavingGoal: bool = False
    returnToGoalThreshold: bool = 80.0
    
    inFrontOfBallTicks: int = 0
    
    attackingAngle: float = None
    attackingDistance: float = None
    pastAttackingAngles: list[float] = np.zeros(1 + NUM_SAMPLES, dtype=np.int32)
    pastAttackingDistances: list[float] = np.zeros(1 + NUM_SAMPLES, dtype=np.int32)
    meanAttackingAngle: float = None
    meanAttackingDistance: float = None
    pastDefendingDistances: list[float] = np.zeros(1 + NUM_SAMPLES, dtype=np.int32)
    meanDefendingDistance: float = None
    defendingAngle: float = None
    defendingDistance: float = None
    
    attackingWidth: float = None
    
    pastGlobalAttackingAngles: list[float] = np.zeros(1 + NUM_SAMPLES, dtype=np.int32)
    meanGlobalAttackingAngle: float = None
    
    lastYellowAngle: float = None
    lastYellowDistance: float = None
    lastBlueAngle: float = None
    lastBlueDistance: float = None
    
    cameraOrientation: float = None
    capturedSpeed: float = 0
    isKicking: bool = False
    kickedTicks: int = 0
    currTurn: float = 0
    currDrive = [0, 0]
    lastNormal = None
    normal: list = [0, 0]
    shootingStyle: int = ShootingStyle.Clear
    atSideTicks: int = 0
    hideBallThreshold = 0.80
    collectingBallTicks: int = 0
    lastDriveDir: float = None
    lastDriveSpeed: float = None
    isDribbling: bool = False
    strafeDirection: float = 90
    isStrafing: bool = False
    flickDirection = 1
    isFlicking: bool = False
    isTurning: bool = False
    
    waitingForBlackToDefend: bool = False
    
    ballSpeed: float = 0
    ballStoppedTicks: int = 0
class Robot:
    def __init__(self):
        
        with open(Path(__file__).parent / "config.json", "r") as f:
            self.config = json.load(f)
            f.close()

        MOTOR_TOPRIGHT = 25
        MOTOR_TOPLEFT = 26
        MOTOR_BOTTOMRIGHT = 27
        MOTOR_BOTTOMLEFT = 28
        DRIBBLER = 0x1F

        self.utils: Utilities = Utilities()
        self.utils.motors = {
            0: Motor(address=MOTOR_TOPRIGHT),
            1: Motor(address=MOTOR_TOPLEFT),
            2: Motor(address=MOTOR_BOTTOMRIGHT),
            3: Motor(address=MOTOR_BOTTOMLEFT),
            "dribbler": Motor(address=DRIBBLER, max_speed=200_000_000) #245_000_000 for ballhiding
        }
        
        self.utils.camera      = AsyncCam([600, 600], center=self.config["center"])
        self.utils.compasses   = [Compass(0x4a), Compass(0x4b)]
        # ~ self.utils.tofs        = TOFChain([0x50, 0x53, 0x54, 0x56])
        # ~ self.utils.captureTof = TOF(0x57)
        self.utils.solenoid    = Solenoid(23)
        self.utils.bt          = BT(mode="server")

        self.middle = LED(self.config["addresses"]["switchMiddle"])
        self.middle.on()
        
        self.utils.switch_left = Button(self.config["addresses"]["switchLeft"], pull_up=False)
        self.utils.switch_right = Button(self.config["addresses"]["switchRight"], pull_up=False)
        
        self.mode: int = Mode.Idle
        self.state: str = State.KickOff # State.Chasing
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
        self.front = self.corners["TopLeft"]["front"]
        self.back = "blue" if self.front == "yellow" else "yellow"
        
        def get_dims_from_corner(corner_name: str):
            corner = self.corners[corner_name]
            front = corner[self.front]
            back = corner[self.back]
            a1 = math.radians(front["angle"])
            a2 = math.radians(back["angle"])
            x1 = -front["dist"] * math.sin(a1)
            x2 = -back["dist"] * math.sin(a2)
            y1 = front["dist"] * math.cos(a1)
            y2 = back["dist"] * math.cos(a2)
            return 0.5 * (x1 + x2), abs(y1 - y2)
            
        self.length = float("inf")
        self.left = -float("inf")
        self.right = float("inf")
        
        x, length = get_dims_from_corner("TopLeft")
        self.left = max(x, self.left); self.length = min(length, self.length)
        x, length = get_dims_from_corner("BottomLeft")
        self.left = max(x, self.left); self.length = min(length, self.length)
        x, length = get_dims_from_corner("TopRight")
        self.right = min(x, self.right); self.length = min(length, self.length)
        x, length = get_dims_from_corner("BottomRight")
        self.right = min(x, self.right); self.length = min(length, self.length)
        
        self.width = abs(abs(self.right) + abs(self.left))
        
        self.utils.camera.set_masks(self.config["cameraMasks"])
        self.prev: float = now()
        self.dt: float = 1/40
        self.update_interval: float = 0.00001
        
    def calculate_final_direction(self, angle: float, distance: float) -> float:
        def angle_poly(x: float) -> float:
            return (self.angleCoeff["x6"] * pow(x, 6)) + (self.angleCoeff["x5"] * pow(x, 5)) + (self.angleCoeff["x4"] * pow(x, 4)) + (self.angleCoeff["x3"] * pow(x, 3)) + (self.angleCoeff["x2"] * pow(x, 2)) + (self.angleCoeff["x1"] * x)

        def f(x) -> float:
            return clamp_lerp(x, 90, 20)
            
        angle = normalise(angle)
        is_negative: bool = angle < 0
        
        mapped_angle: float = angle_poly(angle) if angle > 0 else -angle_poly(abs(angle))

        return normalise(angle_lerp(angle, mapped_angle, f(distance)))

    async def drive_in_direction(self, angle: float, speed: float, contribution: float = 1.0, immediate: bool = False):
        if speed < 0:
            angle = normalise(angle + 180)
            speed = -speed
            
        if self.vars.outOfBounds and not immediate:
            normalDirection, normalMag = self.bb.normal.copy()
            
            globalTravelDirection = angle - self.vars.heading
            deltaAngle = normalise(globalTravelDirection - normalDirection)
            diffAngle = abs(deltaAngle)
            
            if diffAngle > 90 or normalMag > 350.0:
                
                # Perpendicular = 1 * original speed, Antiparallel = 0
                t = clamp_lerp(normalMag, 80, 250)
                speed = (0.1 + 0.9 * math.sin(math.radians(diffAngle))) * abs(speed)
                speed = lerp(speed, self.vars.top_speed, t)
                da = lerp(0, 90, t)
                if normalise(deltaAngle) > 0:
                    angle = normalDirection + (90 - da) + self.vars.heading
                else:
                    angle = normalDirection - (90 - da) + self.vars.heading
                    
        FL = math.sin(math.radians(37.5 - angle))
        FR = math.sin(math.radians(37.5 + angle))

        if abs(FL) >= abs(FR):
            FR = (speed / abs(FL)) * FR
            FL = (speed / FL) * abs(FL)
        elif abs(FL) < abs(FR):
            FL = (speed / abs(FR)) * FL
            FR = (speed / FR) * abs(FR)
            
        if not immediate:
                
            self.future_motor_speeds[0] += FL * contribution
            self.future_motor_speeds[1] += FR * contribution
            self.future_motor_speeds[2] += -FL * contribution
            self.future_motor_speeds[3] += -FR * contribution
            
        else:
            
            self.utils.motors[0].set_speed(FL * contribution, immediate=True)
            self.utils.motors[1].set_speed(FR * contribution, immediate=True)
            self.utils.motors[2].set_speed(-FL * contribution, immediate=True)
            self.utils.motors[3].set_speed(-FR * contribution, immediate=True)
        
        self.vars.lastSpeed = abs(speed)
        self.bb.lastDriveDir = normalise(angle)
    
    async def lerp_drive(self, angle, speed, contribution=1.0):
        if speed < 0:
            angle = normalise(angle + 180)
            speed = -speed
            
        self.bb.currDrive[0] = angle_lerp(self.bb.currDrive[0], angle, 0.1)
        self.bb.currDrive[1] = lerp(self.bb.currDrive[1], speed, 0.1)
        
        await self.drive_in_direction(*self.bb.currDrive, contribution)
        
    async def lerp_turn(self, angle, contribution=0.5):
        if abs(angle) > 180: angle = normalise(angle)
        
        self.bb.currTurn = angle_lerp(self.bb.currTurn, angle, 0.05)
        
        await self.turn(self.bb.currTurn, contribution)
    
    async def turn(self, speed: float, contribution: float = 1.0, immediate: bool = False):
        for i in range(4):
            if not immediate:
                self.future_motor_speeds[i] += clamp(speed * contribution, -1, 1)
            else:
                self.utils.motors[i].set_speed(speed * contribution, immediate=True)
    async def brake(self):
        self.vars.lastSpeed = 0
        self.utils.motors[0].set_speed(0, immediate=True)
        self.utils.motors[1].set_speed(0, immediate=True)
        self.utils.motors[2].set_speed(0, immediate=True)
        self.utils.motors[3].set_speed(0, immediate=True)

    async def confirm_drive(self):
        self.utils.motors[0].set_speed(self.future_motor_speeds[0])
        self.utils.motors[1].set_speed(self.future_motor_speeds[1])
        self.utils.motors[2].set_speed(self.future_motor_speeds[2])
        self.utils.motors[3].set_speed(self.future_motor_speeds[3])
        self.future_motor_speeds = [0, 0, 0, 0]
        
    async def reverse_dribbler(self):
        self.utils.motors['dribbler'].set_speed(1, immediate=True)
    async def enable_dribbler(self, speed_percent = 1):
        self.utils.motors['dribbler'].set_speed(-1 * speed_percent, immediate=True)
    async def stop_dribbler(self):
        self.utils.motors['dribbler'].set_speed(0, immediate=True)

    def drive_direction_bias(self, a: float, d: float) -> float:
        a = normalise(a)
        def f(x, B = 76):
            return x * (1 - 1 / (1 + pow(abs(x / B), 3)))
        mapped_angle = 180 * f(a) / f(180)
        return angle_lerp(a, mapped_angle, clamp_lerp(d, 50, 20))
    def drive_speed_bias(self, a: float, d: float, close = 25, far = 50) -> float:
        a = normalise(a)
        f = 1 - self.speedBias["forwardDamping"] / (1 + pow(0.02 * a, 4))
        g = 1 - 0.5 * self.speedBias["sideDamping"] * (1 - math.cos(math.radians(2 * a)))
        mapped_speed = f * g
        return lerp(mapped_speed, 1, clamp_lerp(d, close, far))
        
    async def update(self):
        "Logic for the robot gameplay"
        
        # ~ await self.turn(clamp(-smooth_linear(self.vars.heading, a = 2640) * self.vars.maintain_orientation_speed, -1, 1), 1.0)
        # ~ await self.confirm_drive()
        # ~ return
        
        if self.vars.ball_distance is not None:
            self.vars.drive_speed = 0.8 + 0.2 * clamp_lerp(self.vars.ball_distance, 30, 45)
        else:
            self.vars.drive_speed = 0.8
        if self.vars.ball_angle is not None:
            ga = normalise(self.vars.ball_angle - self.vars.heading)
            factor = 0.5 + 0.5 * math.cos(math.radians(ga))
            self.vars.drive_speed = lerp(1, self.vars.drive_speed, factor)
        
        ball_is_at_front = abs(self.vars.ball_angle) <= 24.0 if self.vars.ball_angle is not None else False
        ball_is_close = self.vars.ball_distance <= 16.5 if self.vars.ball_distance is not None else False
        if ball_is_at_front and ball_is_close:
            self.vars.has_ball = min(500, self.vars.has_ball + self.dt)
        else:
            self.vars.has_ball = max(0, self.vars.has_ball - self.dt)
            
        if self.state != State.Chasing:
            self.bb.inFrontOfBallTicks = 0
        
        if self.state != State.Shooting:
            self.bb.isKicking = False
            
            cond = abs(self.vars.ball_angle) <= 45.0 and self.vars.ball_distance <= 33.0 if None not in [self.vars.ball_angle, self.vars.ball_distance] else False
            if cond:
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
                
            # Back up to the goal.
            if self.bb.defendingDistance is not None:
                speed = 0.10 * smooth_linear(self.bb.defendingDistance - self.vars.backingDistance)
                await self.drive_in_direction(self.bb.defendingAngle + self.vars.heading, speed, 0.67)
            else:
                await self.drive_in_direction(180 + self.vars.heading, self.vars.drive_speed, 1.0)
        
            await self.turn(clamp(-smooth_linear(self.vars.heading, a = 640) * self.vars.maintain_orientation_speed, -MAXROT, MAXROT), 1.0)
            
            # Center yourself
            a = normalise(localisation_angle - self.vars.heading + 180)
            speed = self.vars.center_speed * smooth_linear((a + 90) % 180 - 90, a = 300)
            if abs(a) >= 90: speed *= -1
            await self.drive_in_direction(self.vars.heading - 90, speed, 0.2)
                
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
            
            if self.bb.waitingForBlackToDefend and self.vars.ball_distance > 50:
                self.state = State.Defending
            
            cond = abs(self.vars.ball_angle - self.vars.heading) >= 120 if self.vars.ball_angle is not None else False
            if cond:
                self.bb.inFrontOfBallTicks += 1
            else:
                self.bb.inFrontOfBallTicks = 0
                
            if abs(self.vars.normalised_ball_angle) <= 90:
                self.bb.isTurning = False
            
            # Transition from chasing to shooting if ball is held long enough.
            if self.vars.has_ball >= 75:
                self.bb.targetDirection = None
                self.state = State.Shooting
                self.bb.leavingGoal = False
                self.bb.currTurn = 0
                self.bb.currDrive = [0, self.vars.lastSpeed]
                self.bb.collectingBallTicks = 0
                self.bb.lastDriveSpeed = self.vars.lastSpeed
                self.bb.attackingGoalTicks = 0
                self.bb.shootingStyle = ShootingStyle.Clear
                self.bb.isStrafing = True
                self.bb.flickDirection = -sign(normalise(self.bb.attackingAngle - self.vars.heading)) if self.bb.attackingAngle else 1
                self.bb.isFlicking = False
                return
            
            # If facing roughly forward
            # !!!!!
            # !!!!!
            # !!!!!
            if 1:
            # !!!!!
            # !!!!!
            # !!!!!
            # ~ if abs(normalise(self.vars.heading)) <= 90 and self.vars.ball_distance is not None:
                # If ball is close enough, face to collect it
                
                # ~ cond = self.vars.ball_distance > 45 and abs(normalise(self.vars.ball_angle)) > 125
                cond = self.bb.isTurning = False
                if cond or self.bb.isTurning:
                    await self.turn(-sign(normalise(self.vars.ball_angle)) * 2, 1.0)
                    self.bb.isTurning = True
                elif self.vars.ball_distance <= 55 and abs(self.vars.normalised_ball_angle) <= 60:
                    t = clamp_lerp(self.vars.ball_distance, 22, 50)
                    t *= max(0, (1 - 1.5 * abs(self.vars.normalised_ball_angle) / 90))
                    turnAngle = self.vars.ball_angle
                    angle = angle_lerp(turnAngle, self.vars.heading, t)
                    await self.turn(clamp(-smooth_linear(angle, a = 800) * 0.004, -0.12, 0.12), 1.0)
                else:
                    await self.turn(clamp(-smooth_linear(self.vars.heading, a = 1000) * self.vars.maintain_orientation_speed, -MAXROT, MAXROT), 1.0)
                    
                direction = self.calculate_final_direction(self.vars.normalised_ball_angle, self.vars.ball_distance)
                direction = self.drive_direction_bias(direction, self.vars.ball_distance)
                # If stuck on the edge of field without capturing, swap directions.
                if self.bb.inFrontOfBallTicks >= 110:
                    speed = self.vars.drive_speed * 0.67
                    direction = -sign(self.bb.xPosition) * abs(direction)
                else:
                    speed = self.drive_speed_bias(direction, self.vars.ball_distance) * self.vars.drive_speed
                
                facing_factor = 1
                if cond or self.bb.isTurning:
                    facing_factor = 0.5 + 0.5 * math.cos(math.radians(abs(normalise(self.vars.ball_angle))))
                    # ~ facing_factor = 0 + 1 * facing_factor
                await self.drive_in_direction(direction + self.vars.heading, speed * facing_factor, contribution = 1.0)
            # If backward, i.e. if we lost the ball after trying to shoot.
            else:
                # If the ball is behind us
                if abs(self.vars.ball_angle) > 90:
                    await self.turn(clamp(-smooth_linear(self.vars.heading, a = 1200) * self.vars.maintain_orientation_speed, -MAXROT, MAXROT), 1.0)
                # if the ball is in front, collect it directly without turning first
                else:
                    await self.turn(clamp(-smooth_linear(self.vars.ball_angle, a = 1200) * 0.02, -MAXROT, MAXROT), contribution=1.0)
                    # ~ direction = self.calculate_final_direction(self.vars.ball_angle, self.vars.ball_distance)
                    # ~ direction = self.drive_direction_bias(direction, self.vars.ball_distance)
                    direction = self.vars.ball_angle
                    speed = 0.23 + 0.77 * clamp_lerp(self.vars.ball_distance, 32, 45)
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
            
            # Go back to chasing if the ball is too far away
            ca = self.utils.camera.angle if self.utils.camera.angle else 0
            # ~ if not (abs(ca) <= 45 or self.utils.camera.angle is None) or self.vars.ball_distance >= 21.0:
            if self.vars.has_ball < 5 and self.bb.kickedTicks <= 0:
                self.state = State.Chasing

            if self.bb.attackingGoalTicks <= 5:
                await self.enable_dribbler(0.60)
            
            attackingAngle = self.bb.attackingAngle if self.bb.attackingAngle else self.bb.meanAttackingAngle
            attackingDistance = self.bb.attackingDistance if self.bb.attackingDistance else 67
            attackingWidth = abs(self.bb.attackingWidth) if self.bb.attackingWidth else 0
            nheading = normalise(self.vars.heading)
            
            self.bb.collectingBallTicks += 1
            
            self.bb.isFlicking = False
            if 1:
            # ~ if abs(nheading) <= 115 and self.bb.shootingStyle != ShootingStyle.Flick:
                # Shooting style is 'clear' here initially
                
                t = self.bb.collectingBallTicks * 0.018
                s = clamp(t, 0, 0.41)
                
                await self.turn(clamp(smooth_linear(-attackingAngle * 0.05, a=0.05), -s, s), 1.0)
                if attackingDistance > 70:
                    
                    s = sign(normalise(attackingAngle - self.vars.heading))
                    
                    # ~ if attackingWidth <= 30 and self.bb.isStrafing:
                        # ~ strafeDirection = -35 * s
                        # ~ a = angle_lerp(0, strafeDirection + self.vars.heading, t)
                        # ~ await self.drive_in_direction(a, self.vars.drive_speed, 1.0)
                    # ~ else:
                    self.bb.isStrafing = False
                    a = angle_lerp(-15 * sign(attackingAngle), attackingAngle, min(t, 1))
                    await self.drive_in_direction(a, self.vars.drive_speed, 1.0)
                    
                    await self.confirm_drive()
                    return
                if self.vars.lastSpeed > 0.01:
                    await self.drive_in_direction(0, self.vars.lastSpeed * 0.95, 1.0)
                await self.confirm_drive()
                
                if abs(attackingAngle) <= 13:# or abs(normalise(attackingAngle - self.vars.heading)) > 45:
                    await self.drive_in_direction(0, 10, 1.0, immediate=True)
                    await asyncio.sleep(0.05)
                    await self.utils.solenoid.shoot()
                    await asyncio.sleep(0.05)
                    await self.drive_in_direction(180 + self.vars.heading, 10, 1.0, immediate=True)
                    await asyncio.sleep(0.1)
                    self.state = State.Chasing
                        
            else:
                self.bb.lastDriveSpeed -= 0.03
                self.bb.shootingStyle = ShootingStyle.Flick
                
                if self.bb.lastDriveSpeed >= -self.vars.dribble_speed and self.bb.attackingGoalTicks <= 5:
                    await self.drive_in_direction(0, self.bb.lastDriveSpeed, 1.0)
                    await self.confirm_drive()
                    
                else:
                    attacking_goal_vertical_distance = abs(attackingDistance * math.cos(math.radians(attackingAngle - self.vars.heading)))
                    if attacking_goal_vertical_distance <= 34.0:
                        self.bb.attackingGoalTicks += 1
                    else:
                        self.bb.attackingGoalTicks = 0
                    
                    ga = normalise(attackingAngle - self.vars.heading)
                    if self.bb.attackingGoalTicks <= 5:
                        await self.enable_dribbler(1)
                        await self.turn(clamp(-normalise(self.vars.heading + 180) * 0.005, -MAXROT, MAXROT), 0.5)
                        await self.drive_in_direction(angle_lerp(0 + self.vars.heading, attackingAngle, 0.3), self.vars.dribble_speed, 1.0)
                        await self.confirm_drive()
                    else:
                        await self.enable_dribbler(1)
                        if self.bb.lastDriveSpeed >= 0.01 and not self.bb.isFlicking:
                            self.bb.lastDriveSpeed -= 0.008
                            await self.drive_in_direction(attackingAngle, self.bb.lastDriveSpeed, 1.0)
                        else:
                            self.bb.isFlicking = True
                            
                if self.bb.isFlicking:
                    if abs(normalise(attackingAngle)) >= 23:
                        s = 0.15
                        turnSpeed = clamp(self.bb.flickDirection * abs(normalise(attackingAngle)) * 0.005, -s, s)
                        await self.turn(turnSpeed, 1.0)
                        await self.confirm_drive()
                        return
                    
                    await self.turn(0, 1.0)
                    await self.drive_in_direction(0, 1, 1.0)
                    await self.confirm_drive()
                    await self.reverse_dribbler()
                    await asyncio.sleep(0.1)
                    await self.utils.solenoid.shoot()
                    await asyncio.sleep(0.1)
                    await self.drive_in_direction(180, 1, 1.0)
                    await asyncio.sleep(0.1)
                    self.state = State.Defending
                    
                    
                    # ~ await self.turn(self.bb.flickDirection * 5, 1.0, immediate=True)
                    # ~ await asyncio.sleep(0.3)
                    # ~ await self.turn(-self.bb.flickDirection * 5, 1.0, immediate=True)
                    # ~ await asyncio.sleep(0.4)
                
        elif self.state == State.Defending:
            """
            Go to goal until atDefenderGoal is true
            oscillate on the goalline (proportional to self.goalTicks).
            Face towards ball/northward.
            """
            
            self.bb.goalTicks += 1
            if self.bb.goalTicks >= 50:
                self.state = State.Chasing
                self.bb.goalTicks = 0
                return
            
            # If the ball is stopped and there's lack of progress then switch from goalie to attacker
            if self.bb.ballStoppedTicks >= 180:
                self.state = State.Chasing
            
            if self.vars.target_goal == Goal.Blue:
                backing_distance = self.utils.camera.yellow_distance
                localisation_angle = self.utils.camera.yellow_angle
            else:
                backing_distance = self.utils.camera.blue_distance
                localisation_angle = self.utils.camera.blue_angle
            
            # If goals not seen, spin.
            if localisation_angle is None:
                localisation_angle = 180
                # ~ await self.turn(0.167, 1.0)
                # ~ await self.confirm_drive()
                # ~ return
                
            # Back up to the goal.
            if self.bb.defendingDistance is not None:
                defending_goal_vertical_distance = abs(self.bb.defendingDistance * math.cos(math.radians(self.bb.defendingAngle - self.vars.heading)))
                speed = 0.1 * smooth_linear(defending_goal_vertical_distance - self.vars.backingDistance, a = 300)
                await self.drive_in_direction(self.bb.defendingAngle + self.vars.heading, clamp(speed, -1, 1), 1.0)
            else:
                speed = self.vars.drive_speed
                await self.drive_in_direction(180 + self.vars.heading, speed, 1.0)
            
            
            # If ball not seen
            if self.utils.camera.angle is None:
                # Center yourself
                a = normalise(localisation_angle - self.vars.heading + 180 + 20 * math.sin(self.bb.goalTicks * 0.01))
                speed = self.vars.center_speed * smooth_linear((a + 90) % 180 - 90, a = 670)
                if abs(a) >= 90: speed *= -1
                await self.drive_in_direction(self.vars.heading - 90, speed, 0.2)
                await self.turn(clamp(-smooth_linear(self.vars.heading, a = 640) * self.vars.maintain_orientation_speed, -MAXROT, MAXROT), 1.0)
                
            else:
                # Position between ball and goal
                
                a = normalise(self.utils.camera.angle - self.vars.heading)
                ball_vertical_distance = abs(self.vars.ball_distance * math.cos(math.radians(self.vars.ball_angle - self.vars.heading)))
                
                if self.bb.defendingAngle is not None:
                    # Drive angle is between ball and defending goal
                    sidewaysAngle = angle_lerp(self.utils.camera.angle, self.bb.defendingAngle, 0.5)
                    driveAngle = angle_lerp(sidewaysAngle, sign(sidewaysAngle) * 90, 0.5)
                    driveSpeed = abs(math.sin(math.radians(self.utils.camera.angle - self.bb.defendingAngle)))
                    await self.drive_in_direction(driveAngle, 25 * smooth_linear(driveSpeed, a = 10), 1.0)
                else:
                    # Defending goal not seen so just drive sideways in the direction of the ball
                    speed = 0.05 * smooth_linear((a + 90) % 180 - 90, a = 1100)
                    speed = speed * lerp(1, 10, clamp_lerp(ball_vertical_distance, self.length, 35))
                    if abs(a) >= 90: speed *= -1
                    await self.drive_in_direction(self.vars.heading + 90, clamp(speed, -1, 1), 1.0)
                
                await self.turn(clamp(-smooth_linear(self.vars.heading, a = 410) * self.vars.maintain_orientation_speed, -MAXROT, MAXROT), 1.0)
            
                # This will be for the goalie
                if self.bb.kickedTicks <= 0:
                    if self.bb.leavingGoal:
                        self.state = State.Chasing
                        await self.confirm_drive()
                        self.bb.goalTicks = 0
                        return
                    if abs(a) > 80 or (ball_vertical_distance <= 28.0):
                        self.bb.leavingGoal = True
                        self.state = State.Chasing
                        await self.confirm_drive()
                        self.bb.goalTicks = 0
                        return
                    self.bb.leavingGoal = False
                
            await self.confirm_drive()
            await self.stop_dribbler()
                
        elif self.state == State.Stalled:
            """
            Stalled is when the you have the ball and the enemy robot is pushing up in front of you.
            Just have orientation correction set really high
            and drive really fast forward
            """
            
            angle = angle_lerp(self.vars.ball_angle, 0, 0.5)
            await self.drive_in_direction(angle, 1.0, 1.0)
            await self.turn(-smooth_linear(self.vars.heading, a = 300) * 0.05, 0.5)
            await self.confirm_drive()
            ...
        elif self.state == State.KickOff:
            """
            Drive directly to the ball until self.kickoffTimer <= 0.
            update self.kickoffTimer
            """
            if self.is_goalie:
                self.state = State.Defending
                
            if self.bb.kickoffTimer <= 0:
                self.state = State.Chasing
                return
                
            kickoffTimer -= 1
            
            await self.turn(clamp(-smooth_linear(self.vars.heading, a = 640) * self.vars.maintain_orientation_speed, -MAXROT, MAXROT), 1.0)
                
            if self.vars.ball_angle is not None:
                await self.drive_in_direction(self.vars.ball_angle, self.vars.drive_speed, 1.0)
                
            await self.confirm_drive()
            
    async def idle(self):
        "Robot is not moving or sensing"
        if self.vars.lastSpeed == 0: return
        
        await self.brake()
        await self.stop_dribbler()
    
    def get_orientation(self) -> float:
        angles = [normalise(compass.read() - self.vars.initial_headings[i]) for i, compass in enumerate(self.utils.compasses) if compass.enabled]
        if len(angles) == 0: return None
        if len(angles) == 1: return angles[0]
        return -(angles[0] + 0.5 * normalise(angles[1] - angles[0]))
        
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
        self.utils.camera.debug = 1#False
        
        
        
        await asyncio.sleep(0.67)
        
        center = np.array(self.utils.camera.center.copy(), dtype=np.uint32)
        
        def arrow(origin: np.array, degrees: float, radius: float) -> tuple:
            "creates the arrow object for opencv to render"
            if type(origin) != np.array: origin = np.array(origin, np.uint32)
            
            try:
                a = math.radians(degrees)
                return (origin, origin + np.array([radius * math.cos(a), radius * math.sin(a)], dtype=np.int32))
            except ValueError:
                return ((0, 0), (1, 1))
        
        def capture_tof_loop():
            while True:
                try:
                    self.vars.frontTofDistance = self.utils.captureTof.read()
                except OSError:
                    print("CAPTURE TOF DISCONNECTED!!")
                    self.vars.frontTofDistance = 1000
                sleep(0.005)
        def outer_tofs_loop():
            "reading of outer body tofs"
            while True:
                try:
                    self.vars.tof_distances = self.utils.tofs.read()
                except OSError:
                    print("TOFS DISCONNECTED !!!")
                    self.vars.tof_distances = [500, 500, 500, 500]
                
                # Calculate xPosition via tofs.
                OFFSET = 3.0
                A = math.radians(35)
                H = math.radians(self.vars.heading)
                x1 = -math.cos(H - A) * max(0, self.vars.tof_distances[0] - OFFSET)
                x2 = -math.cos(H + A) * max(0, self.vars.tof_distances[1] - OFFSET)
                x3 = math.cos(H - A) * max(0, self.vars.tof_distances[2] - OFFSET)
                x4 = math.cos(H + A) * max(0, self.vars.tof_distances[3] - OFFSET)
                self.bb.xPositionTOF = -min(x1, x2) - max(x3, x4)
                
                sleep(0.005)
        def imu_loop():
            "reading the imus on its on thread for performance involving i2c blocking"
            while True:
                self.vars.heading = self.get_orientation()
                sleep(0.000001)
        def bt_loop():
            async def _bt():
                while True:
                    async for data in self.utils.bt.start_server():
                        # Receive teammate position
                        print(data)
                        if data is not None:
                            state = data["state"]
                            # ~ print(STATES[state])
                            if STATES[state] == "Chasing":
                                self.bb.waitingForBlackToDefend = True
                            elif STATES[state] == "Defending":
                                self.bb.waitingForBlackToDefend = False
                        
                        if self.vars.damaged:
                            self.utils.bt.msg_type = Message.Solo
                        else:
                            # Send your own position
                            if self.utils.bt.packet_ready: continue
                        
                            if None in [self.bb.xPosition, self.bb.yPosition, self.vars.ball_angle, self.vars.ball_distance, self.state]:
                                continue
                        
                            if self.mode != Mode.Update:
                                self.utils.bt.msg_type = Message.Solo
                            else:
                                
                                self.utils.bt.msg_type = Message.Update
                            if self.vars.ball_distance is not None:
                                if self.bb.waitingForBlackToDefend and self.vars.ball_distance <= 50:
                                    self.utils.bt.msg_type = Message.Command
                            
                            self.utils.bt.state = STATES.index(self.state)
                            
                            self.utils.bt.x = clamp(self.bb.xPosition * 2 / self.width, -1., 1.)
                            self.utils.bt.y = clamp(self.bb.yPosition * 2 / self.length, -1., 1.)
                            
                            rad = math.radians(self.vars.normalised_ball_angle)
                            bx = self.bb.xPosition + self.vars.ball_distance * math.sin(rad)
                            self.utils.bt.ball_x = clamp(bx * 2 / self.width, -1., 1.)
                            by = self.bb.yPosition + 0.5 * self.vars.ball_distance * math.cos(rad)
                            self.utils.bt.ball_y = clamp(by * 2 / self.length, -1., 1.)
                        
                        self.utils.bt.packet_ready = True
                    
            try:
                asyncio.run(_bt())
            except KeyboardInterrupt:
                pass
                
        threads = [
            Thread(target=imu_loop, daemon=True),
            Thread(target=bt_loop, daemon=True)
        ]
        for index in self.utils.motors:
            threads.append(Thread(target=self.utils.motors[index].event_loop, daemon=True))
        
        for thread in threads:
            thread.start()
        
        calibration_count = max(NUM_SAMPLES + 15, 41)
        
        scared_factor = 1.5
        
        window_scale = 1
        monitor = None; wx = wy = 0
        try:
            monitors = get_monitors()
            if self.utils.camera.debug and len(monitors) > 0:
                monitor = monitors[0]
                wx = int(monitor.width - self.utils.camera.size[0] * window_scale - 5)
                wy = int((monitor.height - self.utils.camera.size[1] * window_scale) // 2)
        except: pass
        
        # Sync up the robot event loop with the camera output
        async for raw_frame in self.utils.camera.main():

            curr = now()
            self.dt = (curr - self.prev) * 1000
            self.prev = curr
                
            frame = self.utils.camera.process(raw_frame)
        
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
                if None not in [self.vars.ball_angle, self.vars.ball_distance, self.vars.heading]:
                    a_prev = math.radians(self.vars.ball_angle - self.vars.heading)
                    a_curr = math.radians(self.utils.camera.angle - self.vars.heading)
                    dx = self.vars.ball_distance * math.sin(a_prev) - self.utils.camera.distance * math.sin(a_curr)
                    dy = self.vars.ball_distance * math.cos(a_prev) - self.utils.camera.distance * math.cos(a_curr)
                    self.bb.ballSpeed = math.sqrt(pow(dx, 2) + pow(dy, 2))
                else:
                    self.bb.ballSpeed = None
                self.vars.ball_angle = normalise(self.utils.camera.angle) #Q+ self.vars.cameraOrientation)
                self.vars.ball_distance = self.utils.camera.distance
            
            # Setting blackboard variables
            if self.vars.ball_angle:
                self.vars.normalised_ball_angle = normalise(self.vars.ball_angle - self.vars.heading)
            if self.utils.camera.yellow_angle is not None:
                self.bb.lastYellowAngle = self.utils.camera.yellow_angle
                self.bb.lastYellowDistance = self.utils.camera.yellow_distance
            if self.utils.camera.blue_angle is not None:
                self.bb.lastBlueAngle = self.utils.camera.blue_angle
                self.bb.lastBlueDistance = self.utils.camera.blue_distance
            if self.vars.target_goal == Goal.Yellow:
                self.bb.attackingAngle = self.utils.camera.yellow_angle
                self.bb.attackingDistance = self.utils.camera.yellow_distance
                self.bb.defendingAngle = self.utils.camera.blue_angle
                self.bb.defendingDistance = self.utils.camera.blue_distance
                self.bb.attackingWidth = self.utils.camera.yellow_width
            else:
                self.bb.attackingAngle = self.utils.camera.blue_angle
                self.bb.attackingDistance = self.utils.camera.blue_distance
                self.bb.defendingAngle = self.utils.camera.yellow_angle
                self.bb.defendingDistance = self.utils.camera.yellow_distance
                self.bb.attackingWidth = self.utils.camera.blue_width
            if self.bb.attackingAngle is not None:
                self.bb.atAttackingGoal = bool(abs(normalise(self.bb.attackingAngle - self.vars.heading)) >= 60)
                i = self.bb.pastAttackingAngles[-1]
                if i is not None:
                    self.bb.pastAttackingAngles[i] = self.bb.attackingAngle
                    self.bb.pastAttackingAngles[-1] = (i + 1) % NUM_SAMPLES
                    self.bb.meanAttackingAngle = np.mean(self.bb.pastAttackingAngles[:NUM_SAMPLES])
                i = self.bb.pastGlobalAttackingAngles[-1]
                if i is not None:
                    self.bb.pastGlobalAttackingAngles[i] = normalise(self.bb.attackingAngle - self.vars.heading)
                    self.bb.pastGlobalAttackingAngles[-1] = (i + 1) % NUM_SAMPLES
                    self.bb.meanGlobalAttackingAngle = np.mean(self.bb.pastGlobalAttackingAngles[:NUM_SAMPLES])
            if self.bb.attackingDistance is not None:
                attacking_goal_vertical_distance = abs(self.bb.attackingDistance * math.cos(math.radians(self.bb.attackingAngle - self.vars.heading)))
                self.bb.atAttackingGoal = self.bb.atAttackingGoal or bool(attacking_goal_vertical_distance <= 47.0)
            if self.bb.defendingDistance is not None:
                self.bb.atDefenderGoal = bool(self.bb.defendingDistance <= 45.0)
                
            self.bb.normal = [0.1, 0]
            self.vars.outOfBounds = False
                
            try:
                if None not in [self.bb.attackingAngle, self.bb.defendingAngle]:
                    # Goal distance normals
                    dist = 27
                    mag = 25
                    max_normal = 100
                    if self.bb.attackingDistance <= dist:
                        self.vars.outOfBounds = True
                        angle = normalise(self.bb.meanGlobalAttackingAngle + 180)
                        if angle > 0 and angle < max_normal:
                            angle = max_normal
                        elif angle < 0 and angle > -max_normal:
                            angle = -max_normal
                        self.bb.normal = [angle,
                                            (dist - self.bb.attackingDistance) * mag]
                    elif self.bb.defendingDistance <= dist:
                        self.vars.outOfBounds = True
                        self.bb.normal = [clamp(normalise(self.bb.defendingAngle + 180), -max_normal, max_normal),
                                            (dist - self.bb.defendingDistance) * mag]
                    
                    # Localisation with both goals seen
                    a1 = math.radians(self.bb.meanGlobalAttackingAngle)
                    a2 = math.radians(self.bb.defendingAngle)
                    x1 = -self.bb.attackingDistance * math.sin(a1)
                    x2 = -self.bb.defendingDistance * math.sin(a2)
                    self.bb.xPosition = 0.5 * (x1 + x2)
                    y1 = self.length / 2 - self.bb.attackingDistance * math.cos(a1)
                    y2 = -self.length / 2 - self.bb.defendingDistance * math.cos(a2)
                    self.bb.yPosition = 0.5 * (y1 + y2)
                    cv2.line(frame, *arrow((center[0] + x1, center[1] + y1), 90, 15), [255, 0, 0], 15)
                    cv2.line(frame, *arrow((center[0] + x2, center[1] + y2), 90, 15), [0, 0, 255], 15)
                
                # Localisation with the attacking goal
                elif self.bb.attackingAngle is not None and self.bb.defendingAngle is None:
                    a1 = math.radians(self.bb.meanGlobalAttackingAngle)
                    self.bb.xPosition = -self.bb.attackingDistance * math.sin(a1) * scared_factor
                    self.bb.yPosition = (self.length / 2 - self.bb.attackingDistance * math.cos(a1)) * scared_factor
                    
                # Localisation with the defending goal
                elif self.bb.attackingAngle is None and self.bb.defendingAngle is not None:
                    a2 = math.radians(self.bb.defendingAngle)
                    self.bb.xPosition = -self.bb.defendingDistance * math.sin(a2) * scared_factor
                    self.bb.yPosition = (-self.length / 2 - self.bb.defendingDistance * math.cos(a2)) * scared_factor
                    
                else:
                    if None not in [self.bb.xPosition, self.bb.yPosition]:
                        self.bb.xPosition *= 1.25
                        self.bb.yPosition *= 1.25
                    
                # Check stalled state
                if self.vars.ball_angle is not None:
                    if abs(self.vars.ball_angle) <= 30 and abs(self.vars.heading) <= 45:
                        deltaAngle = 360 // self.utils.camera.ray_num
                        ball_i = int(self.vars.ball_angle / deltaAngle - 0.5) % self.utils.camera.ray_num
                        ball_ray_dist = self.utils.camera.hits[ball_i]
                        numHits = 0
                        threshold = 16.5 # cm
                        for x in range(-1, 2):
                            if self.utils.camera.hits[(ball_i + x) % self.utils.camera.ray_num] < threshold:
                                numHits += 1
                        if numHits >= 2:
                            if self.state != State.Stalled:
                                self.vars.hasUnstalled = False
                                self.previous_state = self.state
                            self.state = State.Stalled
                        else:
                            if not self.vars.hasUnstalled:
                                self.state = self.previous_state
                                self.vars.hasUnstalled = True
                    else:
                        if not self.vars.hasUnstalled:
                            self.state = self.previous_state
                            self.vars.hasUnstalled = True
                
                # Determine shoot strafe direction
                if self.bb.attackingAngle is not None:
                    goal_i = int(self.bb.attackingAngle // (360 // self.utils.camera.ray_num)) % self.utils.camera.ray_num
                    goal_ray_dist = self.bb.attackingDistance if self.bb.attackingDistance else 40
                    if self.utils.camera.hits[(goal_i + 1) % self.utils.camera.ray_num] > goal_ray_dist + 20:
                        self.bb.strafeDirection = 90
                    elif self.utils.camera.hits[(goal_i - 1) % self.utils.camera.ray_num] > goal_ray_dist + 20:
                        self.bb.strafeDirection = -90
                    else:
                        self.bb.strafeDirection = -90 * sign(self.bb.xPosition)
                    
                # Check ball stopped
                if self.vars.lastSpeed < self.vars.drive_speed * 0.4:
                    if self.bb.ballSpeed is None:
                        pass
                        # ~ self.bb.balStoppedTicks = 0
                    else:
                        if self.bb.ballSpeed < 0.25:
                            self.bb.ballStoppedTicks += 1
                else:
                    self.bb.ballStoppedTicks = 0
                    
                min_i = None
                min_dist = float("inf")
                min_dist_allowed = 42.5
                for i, dist in enumerate(self.utils.camera.hits):
                    if 0 < dist < min_dist_allowed and dist < min_dist:
                        min_i = i
                        min_dist = dist
            
                if min_i is not None:
                    
                    checkNum = 2
                    surrounding = np.zeros(1 + 2 * checkNum, dtype=np.float32)
                    surrounding[checkNum] = min_dist
                    for j in range(1, 1 + checkNum):
                        c = math.cos(j * 2 * math.pi / self.utils.camera.ray_num)
                        surrounding[checkNum - j] = self.utils.camera.hits[(min_i - j) % self.utils.camera.ray_num] * c
                        surrounding[checkNum + j] = self.utils.camera.hits[(min_i + j) % self.utils.camera.ray_num] * c
                    
                    averageSurroundingDistance = np.mean(surrounding)
                    # ~ print([int(x) for x in surrounding], averageSurroundingDistance)
                    if averageSurroundingDistance < min_dist_allowed:
                    
                        angle = (360 // self.utils.camera.ray_num) * min_i
                        self.vars.outOfBounds = True
                        normal_angle = normalise(angle + 180 - self.vars.heading)
                        normal_angle = 45 * round(normal_angle / 45)
                        if self.bb.normal[0] != 0.1:
                            if abs(normalise(self.bb.normal[0])) <= 90:
                                normal_angle = 180
                            else:
                                normal_angle = 0
                        self.bb.normal = [normal_angle, (min_dist_allowed - min_dist) * 23]
            except Exception as e:
                pass
                # ~ print(e)
                
            normalMag = 670
            BACKTHRESHOLD = 38
            FRONTTHRESHOLD = 38
            
            defendingAngle = normalise(self.bb.defendingAngle) if self.bb.defendingAngle else 45
            if abs(defendingAngle) < 125:
                behind_i = int(normalise(180 - self.vars.heading) // (360 // self.utils.camera.ray_num)) % self.utils.camera.ray_num
                if self.utils.camera.hits[behind_i] <= BACKTHRESHOLD:
                    self.bb.normal = [0, normalMag]
            
            attackingAngle = normalise(self.bb.attackingAngle) if self.bb.attackingAngle is not None else 135
            if abs(attackingAngle) > 55:
                front_i = int(normalise(-self.vars.heading) // (360 // self.utils.camera.ray_num)) % self.utils.camera.ray_num
                if self.utils.camera.hits[front_i] <= FRONTTHRESHOLD:
                    self.bb.normal = [180, normalMag]
                                
                    
            if self.utils.camera.debug and (self.utils.camera.ticks % 8) == 0 and monitor is not None:
                
                if self.vars.ball_angle is not None:
                    direction = self.calculate_final_direction(self.vars.normalised_ball_angle, self.vars.ball_distance)
                    cv2.line(frame, *arrow(center, direction, 150), [255, 255, 255], 3)
                angles = [-normalise(compass.read() - self.vars.initial_headings[i]) for i, compass in enumerate(self.utils.compasses) if compass.enabled]
                for a in angles:
                    if a is None: continue
                    cv2.line(frame, *arrow(center, a, 150), (255, 100, 100), 4)
                cv2.line(frame, *arrow(center, -self.vars.heading, 100), [150, 150, 255], 5)
                
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if window_scale != 1:
                    frame = cv2.resize(frame, None, fx=window_scale, fy=window_scale, interpolation=cv2.INTER_AREA)
                
                y = 0
                v = {
                    "State": self.state,
                    "Shooting style": ["Hide ball", "CLEAR the ball", "flick", "MoveToSide"][int(self.bb.shootingStyle)],
                    "xPos": self.bb.xPosition,
                    "yPos": self.bb.yPosition,
                    "left": self.left,
                    "right": self.right,
                    "length": self.length,
                    "normal": self.bb.normal,
                    # ~ "atDefenderGoal": self.bb.atDefenderGoal,
                    # ~ "atAttackingGoal": self.bb.atAttackingGoal,
                    # ~ "atAttackingGoalTicks": self.bb.atAttackingGoalTicks,
                    # ~ "goalTicks": self.bb.goalTicks,
                    # ~ "captureTof": self.vars.frontTofDistance,
                    # ~ "has_ball": self.vars.has_ball,
                    # ~ "kickoffTimer": self.bb.kickoffTimer,
                    # ~ "targetDirection": self.bb.targetDirection,
                    "targetGoal": self.vars.target_goal,
                    # ~ "ballAngle": self.vars.ball_angle,
                    # ~ "ballDistance": self.vars.ball_distance,
                    # ~ "attackDistance": self.bb.attackingDistance,
                    "defendDistance": self.bb.defendingDistance,
                    # ~ "global attackGoalAngle": self.bb.attackingAngle - self.vars.heading if self.bb.attackingAngle else None,
                    # ~ "global defendGoalAngle": self.bb.defendingAngle - self.vars.heading if self.bb.defendingAngle else None,
                    # ~ "attackGoalAngle": self.bb.lastAttackingAngle,
                    # ~ "capturedSpeed": self.bb.capturedSpeed
                    # ~ "lastSpeed": self.vars.lastSpeed,
                    "ballSpeed": self.bb.ballSpeed,
                    "ballStoppedTicks": self.bb.ballStoppedTicks,
                    # ~ "self.bb.currTurn": self.bb.currTurn,
                    "orientation": self.vars.heading,
                    "OOB": self.vars.outOfBounds,
                    # ~ "mean a_Y": self.bb.meanDefendingAngle,
                    # ~ "mean d_Y": self.bb.meanDefendingDistance,
                    # ~ "mean a_B": self.bb.meanAttackingAngle,
                    # ~ "mean d_B": self.bb.meanAttackingDistance,
                    # ~ "in front of ball ticks": self.bb.inFrontOfBallTicks,
                    # ~ "thingy": [self.vars.has_ball < 75, self.bb.kickedTicks <= 0]
                }
                for key, val in v.items():
                    if isinstance(val, float): val = round(val, 2)
                    cv2.putText(frame, f"{key}: {val}", (10, 30 + y * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA, False)
                    y += 1
                cv2.imshow("Camera", frame)
                cv2.setWindowTitle("Camera", f"Camera FPS: {self.utils.camera.fps}")
                cv2.moveWindow("Camera", wx, wy)
                cv2.waitKey(1)
            
            
            
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
                
            if mode == Mode.Update:
                self.vars.damaged = False
            else:
                self.vars.damaged = True
                
            if self.vars.mode != mode:
                self.vars.mode = mode
                self.mode = mode
                print(f"Switch: {mode}")
            


if __name__ == "__main__":
    ts = Robot()
    
    try:
        asyncio.run(ts.start())
    except KeyboardInterrupt:
        sleep(0.1)
        
        # Kill motors once stopped by user
        for index in ts.utils.motors:
            ts.utils.motors[index].reset()
            # ~ ts.utils.motors[index].set_speed(0, immediate=True)
            
        print("Robot stopped by user")
    # ~ except Exception as e:
        # ~ print(e)
    
    

