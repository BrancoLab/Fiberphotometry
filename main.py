import sys
sys.path.append("./")
import os
import time
import numpy as np
from pypylon import pylon
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

from utils.file_io import check_file_exists, check_create_folder, check_folder_empty, create_csv_file
from utils.settings_parser import SettingsParser
from camera.camera import Camera
from datamanager.img_process import ImgProcess
from utils.NI.boardcontrol import NImanager



# ---------------------------------------------------------------------------- #
#                               SECONDARY WINDOW                               #
# ---------------------------------------------------------------------------- #
class FrameViewer(QtGui.QMainWindow, SettingsParser):
    left = 10
    top = 40
    width = 1200
    height = 1200

    def __init__(self, main, parent=None, **kwargs):
        super(FrameViewer, self).__init__(parent)
        settings_file = kwargs.pop("settings_file", None)
        SettingsParser.__init__(self, settings_file=settings_file)

        self.main = main

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 
                        int(self.main.maxX-self.main.minX), 
                        int(self.main.maxY-self.main.minY)))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        # Set geometry
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle("Frame")

    def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_Q:
                print("Stopping")
                self.main.recording=False
                self.main.close()
                self.close()
            event.accept()




# ---------------------------------------------------------------------------- #
#                                  MAIN CLASS                                  #
# ---------------------------------------------------------------------------- #
class Main( QtGui.QMainWindow, SettingsParser, Camera, ImgProcess, NImanager):
    left = 1220
    top = 40
    width = 1600
    trace_height = 200

    def __init__(self, parent=None, **kwargs):
        # Parse kwargs
        settings_file = kwargs.pop("settings_file", None)

        # Load settings
        SettingsParser.__init__(self, settings_file=settings_file)

        self.height = int(self.trace_height*self.n_recording_sites)

        # Start other parent classes
        Camera.__init__(self)
        ImgProcess.__init__(self)
        NImanager.__init__(self)

        # Variables used elserwhere
        self.recording = False
        self.frame_count = 0
        self.data_dump = {i:{'signal':[], 'motion':[]} for i in range(self.n_recording_sites)}

        # Start stuff
        self.setup_experiment_files()
        self.extract_fibers_contours()
        self.start_cameras()

        super(Main, self).__init__(parent)
        self.make_gui()

    # -------------------------------- SETUP FUNCS ------------------------------- #

    def make_gui(self):
        self.frameview = FrameViewer(self)
        self.frameview.show()

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setWindowTitle(self.exp_dir)
        self.setCentralWidget(self.mainbox)
        
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
    
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.plots={i:{'signal':None, 'motion':None} for i in range(self.n_recording_sites)}
        for i in range(self.n_recording_sites):
            #  line plot
            self.otherplot = self.canvas.addPlot()
            self.plots[i]['signal'] = self.otherplot.plot(pen=self.ROIs_colors[i])
            self.plots[i]['motion'] = self.otherplot.plot(pen='w')
            # if i < self.n_recording_sites-1:
            self.canvas.nextRow()

        #### Set Data  #####################
        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # Set geometry
        self.setGeometry(self.left, self.top, self.width, self.height)

        # print start message
        
        start_msg = """
        Starting recording. 
        #####
            To close application and terminate recording press 'q'
        #####
            Experiment name: {}
            saving_video: {}
            n_recording_sites: {}
        """.format(self.exp_dir, self.save_to_video, self.n_recording_sites)
        print(start_msg)

        #### Start  #####################
        self._update()

    def setup_experiment_files(self):
        """[Takes care of creating folder and files for the experiment]
        """
        self.experiment_folder = os.path.join(self.base_dir, self.exp_dir)
        check_create_folder(self.experiment_folder)
        if not check_folder_empty(self.experiment_folder):
            if self.overwrite_files:
                print("\n\n!!! experiment folder is not empty, might risk overwriting stuff !!!\n\n")
            else:
                raise FileExistsError("Experiment folder is not empty and we cant overwrite files, please change folder name")

        # Create files for videos
        if self.save_to_video:
            self.video_files_names = [os.path.join(self.experiment_folder, self.exp_dir+"_cam{}{}".format(i, self.camera_config["video_format"])) 
                                            for i in np.arange(self.camera_config["n_cameras"])]

            # Check if they exist already
            for vid in self.video_files_names:
                if check_file_exists(vid) and not self.overwrite_files: 
                    raise FileExistsError("Cannot overwrite video file: ", vid)

        self.csv_columns = ["ch_{}_{}".format(n, name) for name in ["signal", "motion"] for n in range(self.n_recording_sites)]
        self.csv_path = os.path.join(self.experiment_folder, "sensors_data.csv")
        if os.path.isfile(self.csv_path):
            if not self.overwrite_files: raise FileExistsError("CSV file exists already")
            else:
                os.remove(self.csv_path)
        create_csv_file(self.csv_path, self.csv_columns)

    def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_Q:
                print("Stopping")
                self.recording=False
                self.frameview.close()
                self.close()
            event.accept()

    # ---------------------------------------------------------------------------- #
    # ------------------------------ UPDATE FUNCTION ----------------------------- #
    # ---------------------------------------------------------------------------- #
    def _update(self):
        if self.recording:
            # NI board interaction
            self.update_triggers()

            # Grab and CROP frame
            frame = self.grab_write_frames()
            frame = self.crop_frame(frame)

            # Extract the signal from the ROIs
            self.extract_signal_from_frame(frame)

            # add ROI to frame
            frame = self.mark_roi_on_frame(frame)

            # Display frame
            self.frameview.img.setImage(frame)

            # Update plots
            for i in range(self.n_recording_sites):
                self.plots[i]['signal'].setData(self.data_dump[i]['signal'][-self.visual_config['n_display_points']:])
                self.plots[i]['motion'].setData(self.data_dump[i]['motion'][-self.visual_config['n_display_points']:])

            # Get FPS and make the clock tick
            now = time.time()
            dt = (now-self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = '{} frames - Mean Frame Rate:  {fps:.3f} FPS'.format(self.frame_count, fps=self.fps )
            self.label.setText(tx)
            QtCore.QTimer.singleShot(1, self._update)
            self.counter += 1


# -------------------------------- START CODE -------------------------------- #

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    thisapp = Main()
    thisapp.show()
    sys.exit(app.exec_())
