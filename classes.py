"""
- Vector(x, y)
- lerp(a, b, [step])
- dist_squared(a<Vector>, b<Vector>)
"""

from math import sqrt

class Vector:
    """
    A 2D Vector defined by an x and y value.
    -> Vector(x, y)
    -> Vector((x, y))
    -> Vector([x, y])
    """
    _vec = [0, 0]
    def __init__(self, *args):
        if len(args) >= 2:
            self._vec = [args[0], args[1]]
        elif type(args[0]) in (list, tuple):
            self._vec = [args[0][0], args[0][1]]

    def __str__(self):
        return f"Vector<{self._vec[0]}, {self._vec[1]}>"

    def __repr__(self):
        return self._vec
    
    def __iter__(self):
        return iter(self._vec)
    
    def copy(self):
        return Vector(self._vec[0], self._vec[1])
    
    def __add__(self, other):
        return Vector(self._vec[0] + other._vec[0], self._vec[1] + other._vec[1])
    
    def __sub__(self, other):
        return Vector(self._vec[0] - other._vec[0], self._vec[1] - other._vec[1])
    
    @property
    def x(self):
        return self._vec[0]

    @x.setter
    def x(self, value):
        self._vec[0] = value

    @property
    def y(self):
        return self._vec[1]

    @y.setter
    def y(self, value):
        self._vec[1] = value

    def int(self):
        return [int(_) for _ in self._vec]
    
    @property
    def magnitude(self) -> float:
        return sqrt(self._vec[0]**2 + self._vec[1]**2)

def lerp(a: float, b: float, step: float = .1) -> float:
    return a + (b - a) * step

def dist_squared(a: Vector, b: Vector) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
