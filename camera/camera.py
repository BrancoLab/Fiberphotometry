import sys
sys.path.append("./")

from pypylon import pylon
import skvideo.io
import os
import cv2
import numpy as np
import time

from utils.file_io import *


class Camera():
    def __init__(self):
        self.cameras = None
        self.frame_count = 0
        self.cam_writers = {}
        self.grabs = {}
        self.display_frames = {}

    def start_cameras(self):
        self.get_cameras()  # get the detected cameras
        self.get_camera_writers()   # set up a video grabber for each
        self.setup_cameras()    # set up camera parameters (triggering... )

    def get_cameras(self):
        # Get detected cameras 
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.devices = self.tlFactory.EnumerateDevices()
        if not self.devices: 
            raise ValueError("Could not find any camera")
        else:
            self.cameras = pylon.InstantCameraArray(self.camera_config["n_cameras"])  

    def get_camera_writers(self):
        # Open FFMPEG camera writers if we are saving to video
        if self.save_to_video: 
            for i, file_name in enumerate(self.video_files_names):
                print("Writing to: {}".format(file_name))
                self.cam_writers[i] = skvideo.io.FFmpegWriter(file_name, outputdict=self.camera_config["outputdict"])
        else:
            self.cam_writers = {str(i):None for i in np.arange(self.camera_config["n_cameras"])}

    def adjust_camera_exposure(self, camera, exposure):
        camera.StopGrabbing()
        camera.ExposureTime.FromString(str(exposure))
        camera.Open()
        camera.StartGrabbing()


    def setup_cameras(self):
        # set up cameras
        for i, cam in enumerate(self.cameras):
            cam.Attach(self.tlFactory.CreateDevice(self.devices[i]))
            print("Using camera: ", cam.GetDeviceInfo().GetModelName())
            cam.Open()
            cam.RegisterConfiguration(pylon.ConfigurationEventHandler(), 
                                        pylon.RegistrationMode_ReplaceAll, 
                                        pylon.Cleanup_Delete)

            # Set up Exposure and frame size
            cam.ExposureTime.FromString(self.camera_config["acquisition"]["exposure"])
            cam.Width.FromString(self.camera_config["acquisition"]["frame_width"])
            cam.Height.FromString(self.camera_config["acquisition"]["frame_height"])
            cam.Gain.FromString(self.camera_config["acquisition"]["gain"])
            cam.OffsetY.FromString(self.camera_config["acquisition"]["frame_offset_y"])
            # if int(self.camera_config["acquisition"]["frame_offset_x"]) > 0:
            cam.OffsetX.FromString(self.camera_config["acquisition"]["frame_offset_x"])

            # ? Trigger mode set up
            if self.camera_config["trigger_mode"]:
                # Triggering
                cam.TriggerSelector.FromString('FrameStart')
                cam.TriggerMode.FromString('On')
                cam.LineSelector.FromString('Line4')
                cam.LineMode.FromString('Input')
                cam.TriggerSource.FromString('Line4')
                cam.TriggerActivation.FromString('RisingEdge')

                # ! Settings to make sure framerate is correct
                # https://github.com/basler/pypylon/blob/master/samples/grab.py
                cam.OutputQueueSize = 10
                cam.MaxNumBuffer = 10 # Default is 10
            else:
                cam.TriggerMode.FromString("Off")

            # Start grabbing + GRABBING OPTIONS
            cam.Open()
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # ! if you want to extract timestamps for the frames: https://github.com/basler/pypylon/blob/master/samples/grabchunkimage.py

    def print_current_fps(self, start):
        now = time.time()
        elapsed = now - start
        start = now

        # Given that we did 100 frames in elapsedtime, what was the framerate
        time_per_frame = (elapsed / 100) * 1000
        fps = round(1000  / time_per_frame, 2) 
        
        print("Tot frames: {}, fps: {}.".format(
                    self.frame_count, fps))
        return start


    def grab_single_frame(self):
        """[Grabs a single frame from each camera]
        """
        frames = []
        for i, cam in enumerate(self.cameras): 
            try:
                grab = cam.RetrieveResult(self.camera_config["timeout"])
            except:
                raise ValueError("Grab failed")

            if not grab.GrabSucceeded():
                break
            else:
                frames.append(grab.Array)

        if len(frames) == 1:
            return frames[0]
        else:
            return frames

    def grab_write_frames(self):
        """[Grabs a single frame from each camera and writes it to file]
        """
        frames = []
        for i, (writer, cam) in enumerate(zip(self.cam_writers.values(), self.cameras)): 
            try:
                grab = cam.RetrieveResult(self.camera_config["timeout"])
            except:
                raise ValueError("Grab failed")

            if not grab.GrabSucceeded():
                break
            else:
                if self.save_to_video:
                    writer.writeFrame(grab.Array)
                
                frames.append(grab.Array)
        if len(frames) == 1:
            return frames[0]
        else:
            raise NotImplementedError("Only made stuff to work with one camera sorry")
            return frames

    def close_pylon_windows(self):
        if self.live_display:
            for window in self.pylon_windows:
                window.Close()

    def close_ffmpeg_writers(self):
        if self.save_to_video: 
            for writer in self.cam_writers.values():
                writer.close()


if __name__ == "__main__":
    cam = Camera()



