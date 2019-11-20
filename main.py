import sys
sys.path.append("./")
import os
import time
import numpy as np
from pypylon import pylon

from utils.file_io import check_file_exists, check_create_folder, check_folder_empty
from utils.settings_parser import SettingsParser
from camera.camera import Camera
from utils.img_process import ImgProcess
from camera.visualization import Visual

class Main(SettingsParser, Camera, ImgProcess, Visual):
    def __init__(self, **kwargs):
        # Parse kwargs
        settings_file = kwargs.pop("settings_file", None)

        # Load settings
        SettingsParser.__init__(self, settings_file=settings_file)

        # Start other parent classes
        Camera.__init__(self)
        ImgProcess.__init__(self)
        Visual.__init__(self)


    def setup_experiment_files(self):
        """[Takes care of creating folder and files for the experiment]
        """
        self.experiment_folder = os.path.join(self.base_dir, self.exp_dir)
        check_create_folder(self.experiment_folder)
        if not check_folder_empty(self.experiment_folder):
            print("\n\n!!! experiment folder is not empty, might risk overwriting stuff !!!\n\n")

        # Create files for videos
        if self.save_to_video:
            self.video_files_names = [os.path.join(self.experiment_folder, self.exp_dir+"_cam{}{}".format(i, self.camera_config["video_format"])) 
                                            for i in np.arange(self.camera_config["n_cameras"])]

            # Check if they exist already
            for vid in self.video_files_names:
                if check_file_exists(vid) and not self.debug_mode: raise FileExistsError("Cannot overwrite video file: ", vid)

    def start_experiment(self):
        if self.ROIs is None:
            raise ValueError("You need to extract the ROI locations before starting the experiment, call self.detect_fibers")

        self.setup_experiment_files()

        self.parallel_processes = [] # store all the parallel processes

        # Start cameras and set them up`
        self.start_cameras()

        # Start streaming videos
        self.exp_start_time = time.time() * 1000 #  experiment starting time in milliseconds

        # Set up data storage
        self.data = dict(signal=[[] for i in range(self.n_recording_sites)])

        try:
            self.stream_videos() # <- MAIN LOOP, all the important stuff happens here
        except (KeyboardInterrupt, ValueError) as e:
            print("Acquisition terminted with error: ", e)
            self.terminate_experiment()

    def stream_videos(self):
        """[MAIN LOOP. Keeps grabbing frames from camera and calls the processing and visualization functios to extract 
        and display signal ]"""

        # ? Keep looping to acquire frames
        # self.grab.GrabSucceeded is false when a camera doesnt get a frame -> exit the loop
        while True:
            try:
                if self.frame_count % 100 == 0:  # Print the FPS in the last 100 frames
                    if self.frame_count == 0: start = time.time()
                    else: start = self.print_current_fps(start)

                # ! Loop over each camera and get frames
                frames = self.grab_write_frames()

                # ! Extract the signal from the ROIs
                self.extract_signal_from_frame(frames)

                # Update frame count and terminate
                self.frame_count += 1

            except pylon.TimeoutException as e:
                print("Pylon timeout Exception")
                raise ValueError("Could not grab frame from camera within timeout interval")

        # Close camera
        for cam in self.cameras: cam.Close()



if __name__ == "__main__":
    m = Main()
    m.extract_fibers_contours()
    m.start_experiment()
