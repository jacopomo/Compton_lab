class Cylinder:
    def __init__(self, center, circle, height):
        self.center = center
        self.radius = circle.radius
        self.height = height
    
    def contains(self, point):
        pass #Return true or false if the point is in the volume

class RectPrism:
    def __init__(self, rectangle, height):
        self.center = rectangle.center
        self.length = rectangle.length
        self.width = rectangle.width
        self.height = height

    def contains(self, point):
        pass #Return true or false if the point is in the volume