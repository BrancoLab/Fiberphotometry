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

from utils.settings_parser import SettingsParser
from camera.camera import Camera
from datamanager.img_process import ImgProcess
from utils.NI.boardcontrol import NImanager
from stimuli import grating

from fcutils.file_io.file_io import create_csv_file, append_csv_file
from fcutils.file_io.utils import check_file_exists, check_create_folder, check_folder


# ---------------------------------------------------------------------------- #
#                               SECONDARY WINDOW                               #
# ---------------------------------------------------------------------------- #
class FrameViewer(QtGui.QMainWindow, SettingsParser):
    left = 400
    top = 40
    width = 400
    height = 400

    def __init__(self, main, parent=None, **kwargs):
        super(FrameViewer, self).__init__(parent)
        settings_file = kwargs.pop("settings_file", None)
        SettingsParser.__init__(self, settings_file=settings_file)

        self.main = main

        self.width = kwargs.pop("width", self.width)
        self.height = kwargs.pop("height", self.height)
        self.top = kwargs.pop("top", self.top)
        x_extension = kwargs.pop("x_extension", self.main.maxX-self.main.minX)
        y_extension = kwargs.pop("y_extension", self.main.maxY-self.main.minY)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 
                        int(x_extension), 
                        int(y_extension)))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        # Set geometry
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle("Frame")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            print("Stopping")
            self.main.shut_down()
        elif event.key() == QtCore.Qt.Key_G:
            grating.run()
            event.accept()




# ---------------------------------------------------------------------------- #
#                                  MAIN CLASS                                  #
# ---------------------------------------------------------------------------- #
class Main( QtGui.QMainWindow, SettingsParser, Camera, ImgProcess, NImanager):
    left = 420
    top = 40
    width = 600
    trace_height = 200

    def __init__(self, parent=None, **kwargs):
        # Parse kwargs
        settings_file = kwargs.pop("settings_file", None)

        # Load settings
        SettingsParser.__init__(self, settings_file=settings_file)

        self.height = int(self.trace_height*self.n_recording_sites)

        # Start other parent classes
        NImanager.__init__(self) # start this before the camera to make sure triggers are on
        Camera.__init__(self)
        ImgProcess.__init__(self)

        # Variables used elserwhere
        self.recording = False
        self.frame_count = 0
        self.data_dump = {i:{'signal':[], 'motion':[]} for i in range(self.n_recording_sites)}
        self.ldr_signal_dump = []
        self.prev_behav_frame = None
        self.movement_data_dump = []
        self.stim_leds_on = {'left':0, 'right':0}

        # Start stuff
        self.setup_experiment_files()
        self.extract_fibers_contours()
        self.start_cameras()

        super(Main, self).__init__(parent)
        self.make_gui()

    # -------------------------------- SETUP FUNCS ------------------------------- #

    def make_gui(self):
        # create windows to visualize frames
        self.frameview = FrameViewer(self)
        self.frameview.show()

        self.behav_frameview = FrameViewer(self, top=480, height=550, width=550, 
                        x_extension=self.behav_crop['x1']-self.behav_crop['x0'],
                        y_extension=self.behav_crop['y1']-self.behav_crop['y0'])
        self.behav_frameview.show()

        self.gui_windows = [self.frameview, self.behav_frameview, self]

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setWindowTitle(self.exp_dir)
        self.setCentralWidget(self.mainbox)
        
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
    
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        # Add plots for LDR and mouse movement
        # otherplot = self.canvas.addPlot()
        # self.mvmt_plot = otherplot.plot(pen='w')
        # otherplot = self.canvas.addPlot()
        # self.ldr_plot = otherplot.plot(pen='y')
        # self.canvas.nextRow()

        self.plots={i:{'signal':None, 'motion':None} for i in range(self.n_recording_sites)}
        for i in range(self.n_recording_sites): # use this for multiple fibers
            #  violet led plot
            otherplot = self.canvas.addPlot()
            self.plots[i]['motion'] = otherplot.plot(pen='m')  #self.ROIs_colors[i])  # replace 0 with i for multiple fibers

        # frame movement plot 
        otherplot = self.canvas.addPlot()
        self.mvmt_plot = otherplot.plot(pen='w')
        self.canvas.nextRow() # 


        for i in range(self.n_recording_sites): # use this for multiple fibers
            # blue led plot
            otherplot = self.canvas.addPlot()
            self.plots[i]['signal'] = otherplot.plot(pen='c') # replace 0 with i for multiple fibers

        #  ldr plot
        otherplot = self.canvas.addPlot()
        self.ldr_plot = otherplot.plot(pen='y')

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
        #####
            Colors:
                cyan    - blue LED signal
                magenta - violet LED signal
                white   - behav frame movement
                yellow  - LDR signal
                
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
        self.csv_columns.extend(['ldr', 'behav_mvmt'])
        self.csv_path = os.path.join(self.experiment_folder, "sensors_data.csv")
        if os.path.isfile(self.csv_path):
            if not self.overwrite_files: raise FileExistsError("CSV file exists already")
            else:
                os.remove(self.csv_path)
        create_csv_file(self.csv_path, self.csv_columns)

    def shut_down(self):
        self.recording=False
        self.close_analog_tasks()
        self.switch_leds_off()
        self.close_ffmpeg_writers()
        for window in self.gui_windows: window.close()

    def keyPressEvent(self, event):
        # CLOSE APPLICATION
        if event.key() == QtCore.Qt.Key_Q:
            self.shut_down()
        elif event.key() == QtCore.Qt.Key_G:
            grating.run()

        event.accept()

    def append_to_csvfilse(self):
        # save to csv file
        csv_row_vals = [self.data_dump[k1][k2][-1] for k2 in ['signal','motion'] for k1 in self.data_dump.keys()]
        csv_row_vals.extend([self.ldr_signal_dump[-1], self.movement_data_dump[-1]])
        csv_row = {k:v for k,v in zip(self.csv_columns, csv_row_vals) if 'led' not in k}
        append_csv_file(self.csv_path, csv_row, self.csv_columns)


    # ---------------------------------------------------------------------------- #
    # ------------------------------ UPDATE FUNCTION ----------------------------- #
    # ---------------------------------------------------------------------------- #
    def _update(self):
        if self.recording:
            # NI board interaction
            self.update_triggers()

            # Grab and CROP frames
            frames = self.grab_write_frames()
            if isinstance(frames, list):
                frame = frames[0]
                behav_frame = frames[1]
            else:
                frame = frames
                behav_frame = None
                
            frame = self.crop_frame(frame)

            # Extract the signal from the ROIs
            self.extract_signal_from_frame(frame)

            # add ROI to frame
            frame = self.mark_roi_on_frame(frame)

            # Display frame
            self.frameview.img.setImage(frame)
            if behav_frame is not None:
                # behav_frame = behav_frame[self.behav_crop['x0']:self.behav_crop['x1'], self.behav_crop['y0']:self.behav_crop['y1']]
                self.behav_frameview.img.setImage(behav_frame)
                if self.prev_behav_frame is None:
                    self.prev_behav_frame = behav_frame
                    diff = 0
                else:
                    diff = np.sum((behav_frame - self.prev_behav_frame)**2)
                    self.prev_behav_frame = behav_frame
            else:
                diff = 0
            self.movement_data_dump.append(diff)

            # Append to csv
            self.append_to_csvfilse()

            # Update plots
            self.ldr_plot.setData(self.ldr_signal_dump[-self.visual_config['n_display_points']:])
            self.mvmt_plot.setData(self.movement_data_dump[-self.visual_config['n_display_points']:])

            if self.frame_count > self.visual_config['n_display_points']:
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
