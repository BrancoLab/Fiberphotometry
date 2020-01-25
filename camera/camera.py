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
            if self.camera_config["n_cameras"] > 2: raise NotImplementedError
            self.cameras = pylon.InstantCameraArray(self.camera_config["n_cameras"])  

    def get_camera_writers(self):
        # Open FFMPEG camera writers if we are saving to video
        if self.save_to_video: 
            outdict = self.camera_config['outputdict'].copy()
            outdict['-r'] = str(self.output_fps)
            for i, file_name in enumerate(self.video_files_names):
                if i == 0:
                    w, h = self.camera_config["acquisition"]['frame_width'], self.camera_config["acquisition"]['frame_height']
                else:
                    w, h = self.camera_config["behaviour_acquisition"]['frame_width'], self.camera_config["behaviour_acquisition"]['frame_height']
                indict = {"-r":str(self.output_fps), '-s':'{}x{}'.format(w,h)}

                print("Writing to: {}".format(file_name))
                self.cam_writers[i] = skvideo.io.FFmpegWriter(file_name, inputdict=indict, outputdict=outdict,
                    verbosity=False)
        else:
            self.cam_writers = {str(i):None for i in np.arange(self.camera_config["n_cameras"])}

    def adjust_camera_exposure(self, camera, exposure):
        camera.StopGrabbing()
        try:
            camera.ExposureTime.FromString(str(exposure))          
        except:
            pass
        camera.Open()
        camera.StartGrabbing()


    def setup_cameras(self):
        # set up cameras
        camera_names = [d.GetUserDefinedName() for d in self.devices]
        for i, cam in enumerate(self.cameras):
            if i == 0:
                camera = self.camera_config["calcium_camera"]
                acquisition = self.camera_config["acquisition"]
            else:
                camera = self.camera_config["behaviour_camera"]
                acquisition = self.camera_config["behaviour_acquisition"]

            try:
                idx = camera_names.index(camera)
            except:
                raise ValueError("\n\nCould not find camera {} among devices: {}".format(camera, camera_names))

            try:
                cam.Attach(self.tlFactory.CreateDevice(self.devices[idx]))
            except Exception as e:
                raise ValueError("\n\nFailed to open camera {} with idx {}. Error message:\n{}".format(camera, idx, e))
            print("Using camera: ", cam.GetDeviceInfo().GetUserDefinedName())
            cam.Open()
            cam.RegisterConfiguration(pylon.ConfigurationEventHandler(), 
                                        pylon.RegistrationMode_ReplaceAll, 
                                        pylon.Cleanup_Delete)

            # Set up Exposure and frame size
            try:
                cam.ExposureTime.FromString(acquisition["exposure"])
            except:
                pass
            cam.Width.FromString(acquisition["frame_width"])
            cam.Height.FromString(acquisition["frame_height"])
            try:
                cam.Gain.FromString(acquisition["gain"])
            except:
                pass
                # cam.RawGain.FromString(acquisition["gain"])
            cam.OffsetY.FromString(acquisition["frame_offset_y"])
            cam.OffsetX.FromString(acquisition["frame_offset_x"])


            # ? Trigger mode set up
            if self.camera_config["trigger_mode"]:
                # Triggering
                try:
                    cam.TriggerSelector.FromString('FrameStart')
                    cam.TriggerMode.FromString('On')
                    cam.LineSelector.FromString('Line4')
                    cam.LineMode.FromString('Input')
                    cam.TriggerSource.FromString('Line4')
                    cam.TriggerActivation.FromString('RisingEdge')
                except:
                    pass

                # ! Settings to make sure framerate is correct
                # https://github.com/basler/pypylon/blob/master/samples/grab.py
                cam.OutputQueueSize = 10
                cam.MaxNumBuffer = 10 # Default is 10
            else:
                cam.TriggerMode.FromString("Off")
                cam.OutputQueueSize = 10
                cam.MaxNumBuffer = 10 # Default is 10

            # Start grabbing + GRABBING OPTIONS
            cam.Open()
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)



    def print_current_fps(self):
        now = time.time()
        elapsed = now - self.exp_start

        # Given that we did 100 frames in elapsedtime, what was the framerate
        time_per_frame = (elapsed / self.frame_count) * 1000
        fps = round(1000  / time_per_frame, 2) 
        
        print("Tot frames: {}, fps: {}.".format(
                    self.frame_count, fps))


    def grab_single_frame(self):
        """[Grabs a single frame from each camera]
        """
        frames = []
        for i, cam in enumerate(self.cameras): 
            try:
                grab = cam.RetrieveResult(self.camera_config["timeout"])
            except Exception as e:
                raise ValueError("Grab failed for camera: {}  with exception: {}".format(cam.GetDeviceInfo().GetUserDefinedName(), e))

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
        for i, ((writerkey, writer), cam) in enumerate(zip(self.cam_writers.items(), self.cameras)): 
            try:
                grab = cam.RetrieveResult(self.camera_config["timeout"])
            except Exception as e:
                raise ValueError("Grab failed for camera: {}  with exception: {}".format(cam.GetDeviceInfo().GetUserDefinedName(), e))

            if not grab.GrabSucceeded():
                raise ValueError("Grab failed for camera: {}".format(cam.GetDeviceInfo().GetUserDefinedName()))
            else:
                if self.save_to_video:
                    try:
                        writer.writeFrame(grab.Array)
                    except Exception as e:
                        raise ValueError("Failed to write to frame for camera {}.\n".format(cam.GetDeviceInfo().GetUserDefinedName()),\
                            "Writer {}: {}\n".format(writerkey, writer),\
                            "Frame size: {}\n".format(grab.Array.shape),\
                            "excpetion: ", e)
                frames.append(grab.Array)
        
        self.frame_count += 1
        if len(frames) == 1:
            return frames[0]
        else:
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



