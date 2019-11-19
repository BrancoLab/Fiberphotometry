import sys
sys.path.append("./")
import os
import time
import numpy as np

from utils.file_io import check_file_exists, check_create_folder, check_folder_empty
from utils.settings_parser import SettingsParser
from camera.camera import Camera

class Main(SettingsParser, Camera):
    def __init__(self, **kwargs):
        # Parse kwargs
        settings_file = kwargs.pop("settings_file", None)

        # Load settings
        SettingsParser.__init__(self, settings_file=settings_file)

        # Start other parent classes
        Camera.__init__(self)


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
        self.setup_experiment_files()

        self.parallel_processes = [] # store all the parallel processes

        # Start cameras and set them up`
        self.start_cameras()

        # Start streaming videos
        self.exp_start_time = time.time() * 1000 #  experiment starting time in milliseconds

        try:
            self.stream_videos() # <- t
        except (KeyboardInterrupt, ValueError) as e:
            print("Acquisition terminted with error: ", e)
            self.terminate_experiment()



if __name__ == "__main__":
    m = Main()
    m.start_experiment()