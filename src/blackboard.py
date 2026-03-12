"""
Blackboard for storing robot state and decision-making data
"""

import numpy as np
from .states import ShootingStyle

NUM_SAMPLES = 2

class Blackboard:        
    """Stores blackboard data for decision making and state tracking"""
    
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
