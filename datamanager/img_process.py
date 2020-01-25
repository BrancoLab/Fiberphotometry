import sys
sys.path.append("./")

import cv2
import numpy as np
from pypylon import pylon
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

from utils.colors import MplColorHelper
from utils.file_io import append_csv_file

class ImgProcess:
    def __init__(self):
        self.ROIs = None 
        self.ROIs_colors, self.ROI_colors_vispy = self._get_fibers_colors()

        if self.cameras is None:
            self.get_cameras()  # get the detected cameras
            self.setup_cameras()    # set up camera parameters (triggering... )

    # ---------------------------------------------------------------------------- #
    #                                     UTILS                                    #
    # ---------------------------------------------------------------------------- #
    def crop_frame(self, frame):
        return frame[self.minY:self.maxY, self.minX:self.maxX]

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

    @staticmethod
    def add_roi_to_list(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(data[0], (x, y), data[1], (0, 255, 0), 2) 
            cv2.circle(data[0], (x, y), data[1], (0, 0, 255), 3)  
            return data[2].append((x, y, data[1]))

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

    def mark_roi_on_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        for (x,y,r), color in zip(self.ROIs, self.ROIs_colors):
            cv2.circle(frame, (x, y), r, color, 2) 
            cv2.circle(frame, (x, y), 1, (0, 0, 255), 3)
        return frame

    def _get_fibers_colors(self):
        ch = MplColorHelper("tab10", 0, self.n_recording_sites, rgb255=True)
        colors = [ch.get_rgb(i) for i in range(self.n_recording_sites)]

        vispy_colors = []
        for r,g,b in colors:
            vispy_colors.append((r/255, g/255, b/255, 1))

        vispy_colors = np.float32(np.vstack(vispy_colors))
        return colors, vispy_colors


    # ---------------------------------------------------------------------------- #
    #                                 PREPROCESSING                                #
    # ---------------------------------------------------------------------------- #

    def manual_fiber_detection(self, frame, ROIs):
        cv2.startWindowThread()
        cv2.namedWindow('detection')
        cv2.moveWindow("detection",100,100)   
        cv2.imshow('detection', frame)

        # create functions to react to clicked points
        data = [frame, self.single_fiber_diameter-self.single_fiber_radius_titration, ROIs]
        cv2.setMouseCallback('detection', self.add_roi_to_list, data)  

        while len(ROIs) < self.n_recording_sites:
            cv2.imshow('detection', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    def automatic_fiber_detection(self, frame):
        """[Uses opencv methods to draw circular ROIs around the fibers]
        """

        # grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        original_frame = frame.copy()

        # circles=cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, cv2.HOUGH_GRADIENT, 100, 100,
        #                         param1 = 50, param2 = 350, 
        #                         minRadius = self.single_fiber_diameter-50, 
        #                         maxRadius = self.single_fiber_diameter+20) 
        # frame, ROIs = self._draw_circles_on_frame(circles, frame) # circles.shape(1, 15, 3)

        # if len(ROIs) < self.n_recording_sites:
            # print("     found {} out of {} rois, please mark the remaining rois manually".format(len(ROIs), self.n_recording_sites))
            # Manually add rois
        ROIs = []
        self.manual_fiber_detection(frame, ROIs)

        # Display results
        self.display_results(dict(output=frame))
        return ROIs

    def compute_new_frame_size(self, frame, ROIs):
        self.minX, self.minY, self.maxX, self.maxY = frame.shape[0], frame.shape[1], 0, 0
        for (x,y,r) in ROIs:
            if x-r < self.minX and x-r>0: self.minX = x-r
            elif x-r <= 0: self.minX = 0

            if y-r < self.minY and y-r > 0: self.minY = y-r
            elif y-r <= 0: self.minY = 0

            if x+r > self.maxX and x+r < frame.shape[0]: self.maxX = x+r
            elif x+r >= frame.shape[0]: self.maxX = frame.shape[1]

            if y+r > self.maxY and y+r < frame.shape[1]: self.maxY = y+r
            elif y+r >= frame.shape[1]: self.maxY = frame.shape[1]

        return [(x-self.minX, y-self.minY, r) for x,y,r in ROIs]
        


    def extract_fibers_contours(self):
        """[Detects the location of fibers in the frame and draws an ROI around each]
        """
        print("Trying automated fiber detection for {} recording sites".format(self.n_recording_sites))
        #  GET OVER EXPOSED FRAME
        self.adjust_camera_exposure(self.cameras[0], 20000)
        self.switch_leds_on() # switch on the stimulation LEDs to see the fiber

        self.trigger_frame()
        frames = self.grab_single_frame()

        # reset camera and LEDs
        self.switch_leds_off()
        self.adjust_camera_exposure(self.cameras[0], self.camera_config["acquisition"]["exposure"])


        # If we have multiple cameras we will get a list of frames
        if isinstance(frames, list):
            frame = frames[0]
        else:
            frame = frames

        # Process the frame to detect fibers
        ROIs = self.automatic_fiber_detection(frame)

        # Crop frame to ROIs
        ROIs = self.compute_new_frame_size(frame, ROIs)
        frame = self.crop_frame(frame)


        # Check that everything went okay
        if len(ROIs) != self.n_recording_sites:
            raise ValueError("Detected {} when {} were expected!!!".format(len(ROIs), 
                                self.n_recording_sites))
        else:
            self.ROI_masks = self._define_ROIs_masks(ROIs, frame)
            self.ROI_mask_3d = np.dstack(self.ROI_masks)
            self.ROIs = ROIs
            self.recording = True


    # ---------------------------------------------------------------------------- #
    #                               SIGNAL EXTRACTION                              #
    # ---------------------------------------------------------------------------- #

    def extract_signal_from_frame(self, frame):


        # Extract the average intensity within each ROI
        frame3d = np.repeat(frame[:, :, np.newaxis], self.n_recording_sites, axis=2)
        if not self.fast_signal_extraction:
            signal = np.float16(np.nanmean(np.ma.masked_array(frame3d, self.ROI_mask_3d), axis=(0,1)))
        else:
            signal = np.float16(np.mean(frame3d*self.ROI_mask_3d, axis=(0,1)))

        # signal = [0]

        for i,s in enumerate(signal):
            if self.frame_count % 2 == 0: # When the frame number is even, we are acquiring under the blue LED
                self.data_dump[i]['signal'].append(s) # <- signal 
                if self.frame_count > 2:
                    self.data_dump[i]['motion'].append(self.data_dump[i]['motion'][-1])
                else:
                    self.data_dump[i]['motion'].append(0)
            else:  # When the frame number is odd, we are acquiring under the violet LED
                if self.frame_count > 2:
                    self.data_dump[i]['signal'].append(self.data_dump[i]['signal'][-1])
                else:
                    self.data_dump[i]['signal'].append(0)
                self.data_dump[i]['motion'].append(s)  # <- signal 


          
if __name__ == "__main__": 
    ip = ImgProcess()