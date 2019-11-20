import sys
sys.path.append("./")

import cv2
import numpy as np
from pypylon import pylon
import matplotlib.pyplot as plt
from scipy.stats import tmean

class ImgProcess:
    def __init__(self):
        self.ROIs = None

        if self.cameras is None:
            self.get_cameras()  # get the detected cameras
            self.setup_cameras()    # set up camera parameters (triggering... )

    def _draw_circles_on_frame(self, circles, frame):
        """[Draws a circle around each ROIs for visualization porpuses]
        """
        # make frame rgb to display colored circles
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        ROIs = []
        # Draw circles that are detected. 
        if circles is not None: 
            if len(circles.shape) == 2:
                return frame, ROIs

            # Convert the circle parameters a, b and r to integers. 
            circles = np.uint16(np.around(circles)) 
        
            for pt in circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2]-self.single_fiber_radius_titration
                ROIs.append((a, b, r))

                # Draw the circumference of the circle. 
                cv2.circle(frame, (a, b), r, (0, 255, 0), 2) 
        
                # Draw a small circle (of radius 1) to show the center. 
                cv2.circle(frame, (a, b), 1, (0, 0, 255), 3) 
        return frame, ROIs

    def detect_fibers(self, frame):
        """[Uses opencv methods to draw circular ROIs around the fibers]
        """
        # grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        original_frame = frame.copy()

        circles=cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, cv2.HOUGH_GRADIENT, 100, 100,
                                param1 = 50, param2 = 250, 
                                minRadius = self.single_fiber_diameter-30, 
                                maxRadius = self.single_fiber_diameter+20) 
        frame, ROIs = self._draw_circles_on_frame(circles, frame) # circles.shape(1, 15, 3)

        return dict(output=frame), ROIs

    def _define_ROIs_masks(self, ROIs, frame):
        """
            [Creates a masked frame for each ROI to facilitate signal extraction later on]
        """
        masks = []
        for roi in ROIs:
            newframe = np.zeros_like(frame)

            # Draw the circle and set else to nan
            cv2.circle(newframe, (roi[0], roi[1]), roi[2], (1, 1, 1), -1) 
            masks.append(newframe)
        return masks


    def display_results(self, frames):
        print("\n\nDisplaying results from roi detection. Press 'q' to continue")

        # Create opencv windows to show results
        for name,frame in frames.items():
            window = cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 900,900)
            cv2.imshow(name,frame)
            cv2.moveWindow("output",100,100)

        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()

        self.fibers_template = frames['output']  # store this for visualization
        # self.add_template_image()

    def display_frame_opencv(self, frame):
        window = cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 900,900)
        cv2.imshow("frame",frame)
        cv2.moveWindow("output",100,100)
        cv2.waitKey(1)

    def extract_fibers_contours(self):
        """[Detects the location of fibers in the frame and draws an ROI around each]
        """
        # Increase camera exposure
        self.adjust_camera_exposure(self.cameras[0], 18000)

        frame = self.grab_single_frame()
        # If we have multiple cameras we will get a list of frames
        if isinstance(frame, list):
            raise NotImplementedError("Cannot deal with multiple cameras here yet")

        # Process the frame to detect fibers
        frames, ROIs = self.detect_fibers(frame)

        # Display results
        self.display_results(frames)

        # reset camera exposure
        self.adjust_camera_exposure(self.cameras[0], self.camera_config["acquisition"]["exposure"])

        # Check that everything went okay
        if len(ROIs) != self.n_recording_sites:
            raise ValueError("Detected {} when {} were expected!!!".format(len(ROIs), 
                                self.n_recording_sites))
        else:
            self.ROI_masks = self._define_ROIs_masks(ROIs, frame)
            self.ROIs = ROIs
            self.recording = True


    def extract_signal_from_frame(self, frame):
        for i, fiber_mask in enumerate(self.ROI_masks):
            signal = np.mean(np.ma.masked_array(frame, fiber_mask))
            self.data['signal'][i].append(signal)
            self.data['update_signal'][i] = signal
          
if __name__ == "__main__":
    ip = ImgProcess()