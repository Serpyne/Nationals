class Circle:
    def __init__(self, center, radius, colour):
        self.center = center
        self.radius = radius
        self.colour = colour
class Rect:
    def __init__(self, x, y, w, h, colour):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.colour = colour
class Sector:
    def __init__(self, center, radius, start_angle, end_angle, colour):
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.colour = colour
