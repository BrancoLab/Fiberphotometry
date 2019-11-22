import sys
sys.path.append("./")


import numpy as np
from random import random

from vispy import scene, app

class Visual:
    def __init__(self):
        self.last_updated_frame = 0
        if __name__ == "__main__": # ? for debug
            self.recording = True
            self.visual_config = dict(
                window_size = [1000, 1000],
                n_display_points = 200,
            )
            self.n_recording_sites = 4

    def get_random_color(self):
        col = [np.random.uniform(.4, 0.8),
                np.random.uniform(.6, 1.0), np.random.uniform(.2, .4), 1.0]
        return col

    def start_gui(self):
        self.N, self.M = self.n_recording_sites, self.visual_config['n_display_points']
        cols = 1

        canvas = scene.SceneCanvas(keys='interactive', show=True, size=self.visual_config['window_size'])
        grid = canvas.central_widget.add_grid()
        view = grid.add_view(0, 0)
        view.camera = scene.PanZoomCamera(rect=(0, -1, 1, 1), aspect=20, flip=(True, True, False))
        view.camera.zoom(-5)

        # compute the position of each line
        w, h = self.visual_config['window_size'][0]-200, self.visual_config['window_size'][1]-200
        pos = np.zeros((self.N, 2))
        pos[:, 0] = 200
        pos[:, 1] = [i*h/self.N for i in range(self.N)]

        self.lines = scene.ScrollingLines(n_lines=self.N, line_size=self.M, columns=cols, dx=2/self.M,
                                    cell_size=(10, 8), parent=view.scene)
        self.lines.transform = scene.STTransform(translate=(-1.8, -40), scale=(2, 2))

        timer = app.Timer(connect=self.update)
        timer.start()
        app.run()

    def update(self, ev):
        try:
            if self.recording:
                data = np.array(list(self.data['update_signal'].values())).reshape(self.N, 1)
                self.lines.roll_data(data)
        except:
            pass
        return

if __name__ == "__main__":
    v = Visual()
    v.start_gui()