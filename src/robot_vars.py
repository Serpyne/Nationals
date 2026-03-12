"""
Robot variables and state tracking
"""

class RobotVars:
    """Stores robot variables and state information"""
    
    ball_angle: float = 0
    normalised_ball_angle: float = 0
    ball_distance: float = 0
    heading: float = 0
    initial_headings: list[float] | None = None
    last_seen_ball: int = 0
    has_ball: float = 0.0
    tof_distances: list[float] = [float("inf") for _ in range(4)]
    frontTofDistance: float = float("inf")
    
    damaged: bool = False
    
    blind_milliseconds: int = 670
    target_goal: str | None = None
    drive_speed: float = 0.6
    top_speed: float = 0.8
    dribble_speed: float = 0.5
    maintain_orientation_speed: float = 1 / 67
    mode: str | None = None
    center_speed: float = 0.41
    lastSpeed: float = 0
    backingDistance: float = 30.0
    outOfBounds: bool = False
    outOfBoundsTicks: int = 0
    hasUnstalled: bool = False
    cameraOrientation: float | None = None
