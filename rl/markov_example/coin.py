# Taken from https://stackoverflow.com/questions/78564587/manim-cant-animate-a-3d-coin-flip

from manim import *

class Coin(VGroup):

    def __init__(self, radius=1, height=None, **kwargs):
        super().__init__(**kwargs)

        if height is None:
            height = radius / 4

        self.top = Dot(radius=radius, color=BLUE, fill_opacity=1)
        self.top.move_to(OUT*height/2)
        
        self.bottom = self.top.copy()
        self.bottom.set_color(RED)
        self.bottom.move_to(IN*height/2)
        
        self.edge = Cylinder(radius=radius, height=height)
        self.edge.set_fill(GREY, opacity=1)
        
        # rotate the cylinder so that the upper face is towards the camera
        self.edge.rotate(90 * DEGREES, OUT)
        
        self.add(self.edge, self.bottom, self.top)
    
    def add_to_front(self, *items):
        self.remove(*items)
        self.add(*items)