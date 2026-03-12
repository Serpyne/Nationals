"""
State and mode constants for the robot
"""

class Goal:
    """Goal color constants"""
    Yellow = "Yellow"
    Blue = "Blue"

class Mode:
    """Robot operating modes"""
    Update = "Update"
    Idle = "Idle"
    Calibrate = "Calibrate"

class State:
    """Robot game states"""
    Chasing = "Chasing"
    Defending = "Defending"
    Shooting = "Shooting"
    Stalled = "Stalled"
    KickOff = "KickOff"
    Blind = "Blind"

class ShootingStyle:
    """Shooting style constants"""
    HideBall = 0
    Clear = 1
    Flick = 2
    MoveToSide = 3
