import sys
sys.path.append("./")


import numpy as np

from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app, scene
from vispy.color import get_colormaps
from vispy.visuals.transforms import STTransform


class Visual:
    def __init__(self):
        pass
        # canvas = scene.SceneCanvas(keys='interactive', size=self.visual_config['window_size'], show=True)

        # Create a line object
        # color = next(colormaps)
        # line = scene.Line(pos=pos, color=color, method='gl')
        # line.transform = STTransform(translate=[0, 140])
        # line.parent = canvas.central_widget


    # def update(self):
    #     line.set_data()


