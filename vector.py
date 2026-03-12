"""

Vector2 class

"""

from math import sqrt, atan2, degrees

class Vector:
    """Magnitude; Direction measured in degrees"""
    def __init__(self, *args, **kwargs):
        self.xy = [0, 0]
        
        def vectorTwoCompError(): raise Exception("Vector declaration requires two components")
        if len(args) == 0:
            if len(kwargs) == 0:
                return
            else:
                args = tuple(kwargs.values())
                
        if len(args) == 1:
            if type(args[0]) in [list, tuple]:
                if len(args[0]) == 2:
                    self.xy = list(args[0])
                else:
                    vectorTwoCompError()
            else:
                vectorTwoCompError()
        elif len(args) >= 2:
            self.xy = list(args)
        else:
            vectorTwoCompError()
    
    def __str__(self) -> str:
        return f"Vector<{self.xy[0]}, {self.xy[1]}>"
    def __repr__(self) -> str:
        return str(self)
        
    def __getitem__(self, i) -> float:
        if i < 0 or i > 1: raise Exception("Vector index must be 0 or 1.")
        return self.xy[i]
        
    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.xy[0] + other.xy[0], self.xy[1] + other.xy[1])
    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.xy[0] - other.xy[0], self.xy[1] - other.xy[1])
    def __mul__(self, scalar: float) -> "Vector":
        return Vector(scalar * self.xy[0], scalar * self.xy[1])
    def __truediv__(self, scalar: float) -> "Vector":
        return Vector(self.xy[0] / scalar, self.xy[1] / scalar)
    def int(self) -> "Vector":
        return Vector(int(self.xy[0]), int(self.xy[1]))
    def __tuple__(self) -> tuple[float, float]:
        return self.xy

    def magnitude(self) -> float:
        return sqrt(self.magnitude_squared())
    def magnitude_squared(self) -> float:
        return self.xy[0]**2 + self.xy[1]**2
    
    def direction(self) -> float:
        return degrees(self.direction_radians)
    def direction_radians(self) -> float:
        return atan2(self.xy[1], self.xy[0])
            
if __name__ == "__main__":
    print(Vector(0, 0))
    print(Vector((0, 0)))
    print(Vector(x=0, y=0))
    print(Vector(pos=(0, 0)))
    
    print(Vector(3,4).magnitude())
