class Volume:
    def __init__(self, surf_near, surf_far, geometry, material):
        self.surf_near = surf_near
        self.surf_far = surf_far
        self.geometry = geometry # Either Cylinder or RectPrism
        self.material = material
