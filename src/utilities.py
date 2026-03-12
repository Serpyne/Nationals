"""
Hardware utilities class
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hardware import Motor, Compass, TOF, TOFChain, Solenoid
    from .vision import AsyncCam
    from .communication import BT

class Utilities:
    """Stores hardware utilities and components"""
    
    motors: dict = None
    camera: "AsyncCam | None" = None
    compasses: list["Compass"] | None = None
    tofs = None
    switch_left = None
    switch_right = None
    captureTof = None
    solenoid: "Solenoid | None" = None
    bt: "BT | None" = None
