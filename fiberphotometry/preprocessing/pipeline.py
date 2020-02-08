import os
import logging
import sys

from fancylog import fancylog
import fancylog as package

from behaviour.tdms.mantis_videoframes_test import check_mantis_dropped_frames
from behaviour.tdms.utils import get_analog_inputs_clean_dataframe, get_analog_inputs_clean
from behaviour.utilities.signals import get_frames_times_from_squarewave_signal
from tdmstovideo.converter import convert
from fcutils.file_io.utils import check_file_exists, get_file_name
from fcutils.file_io.io import save_json, load_json

from fiberphotometry.preprocessing.plotting import plot_frame_times


# ? For debugging 
data_folder = "Z:\\swc\\branco\\rig_photometry\\tests\\200205\\200205_mantis_test_longer_exposure_noaudio"
ca_video_tdms = "FP_calcium_and_behaviour(0)-FP_calcium_camera.tdms"
ca_video_path = None # If left as None it will save it the same folder and with the same name as the tdms file
FPS = 100

OVERWRITE = False # If true it will overwrite the results of a previous analysis
RUN_CONVERSION = True # If true it will convert the tdms video to mp4
EXTRACT_FRAME_TIMES = True # If true it will extract the time of blue and violed LEDs frames from analog inputs (slow)

# To extrac frame times
analog_inputs_tdms = "FP_calcium_and_behaviour(0).tdms"
camera_triggers_channel='FP_calcium_camera_triggers_reading'
blue_led_triggers_channel='FP_blue_led_reading'
violet_led_triggers_channel='FP_violet_led_reading'


class Pipeline:
    def __init__(self, data_folder, ca_video_tdms,
                    ca_video_path=None, fps=100,
                    overwrite=False, run_conversion=True,
                    extract_frame_times=True, logging=False, 
                    debug=False, **kwargs):
        """
            Pipeline to preprocess fiberphotometry data. IT:
                1) Converts video.tdms to .mp4
                2) Rotates/aligns the frames to a template # TODO
                3) Extract times of blue led and violet led frames [opetional]

            :param data_folder: path to folder with video.tdms and other data [mantis output]
            :param ca_video_tdms: name of video.tdms file (without the folder)
            :param run_conversion: bool, if true the video is converted to mp4
            :param ca_video_path: str, optional. Path to where the converted .tdms -> .mp4 is saved
            :param fps: int, optional fps that the calcium camera recorded at
            :param overwrite: bool, if true it overwrites the results of previous preprocessing
            :param extract_frame_times: bool, if true it extracts the time of camera, blue led and violet led frames
                        from analog inputs files.
            :param logging: set as true if fancylog logging was started by another application. 
            :param kwargs: used to pass params for extract_frame_times: [analog_inputs_tdms, camera_triggers_channel,
                        blue_led_triggers_channel, violet_led_triggers_channel]
            :param debug: bool, if True extra plots are made
            
        """
        # Params
        self.overwrite = overwrite
        self.run_conversion = run_conversion
        self.data_folder = data_folder
        self.ca_video_tdms = ca_video_tdms
        self.ca_video_path = ca_video_path
        self.fps = fps
        self.extract_frame_times = extract_frame_times
        self.logging = logging
        self.debug = debug

        # extra additional params
        self.analog_inputs_tdms = kwargs.get('analog_inputs_tdms', None)
        self.camera_triggers_channel = kwargs.get('camera_triggers_channel', None)
        self.blue_led_triggers_channel = kwargs.get('blue_led_triggers_channel', None)
        self.violet_led_triggers_channel = kwargs.get('violet_led_triggers_channel', None)

        self.setup()

    def setup(self):
        """
            Gets complete paths to the various files and checks that they exist and starts fancy log logger
        """
        # Get complete file paths
        if self.ca_video_path is None:
            self.ca_video_path = os.path.join(self.data_folder, get_file_name(self.ca_video_tdms)+".mp4")
        self.ca_video_tdms = os.path.join(self.data_folder, self.ca_video_tdms)
        self.ca_video_metadata_tdms = os.path.join(self.data_folder, get_file_name(self.ca_video_tdms)+"meta.tdms")

        if self.extract_frame_times:
            try:
                self.analog_inputs_tdms = os.path.join(self.data_folder, self.analog_inputs_tdms)
            except Exception as e:
                raise ValueError("Could not get filepath for analong inputs tdms: \n{}".format(e))
            check_file_exists(self.analog_inputs_tdms, raise_error=True)

        # Start logging
        if not self.logging:
            fancylog.start_logging(self.data_folder, package, verbose=True, filename="fp_preprocessing")
            logging.info("Starting to preprocess files:")
            logging.info("{}\n{}\n{}\n{}\n".format(self.data_folder, self.ca_video_tdms, self.ca_video_metadata_tdms, self.ca_video_path))

        # check files exist
        video_exists = check_file_exists(self.ca_video_tdms)
        if not video_exists:
            logging.warning("Couldn't find calcium video tdms at: " + self.ca_video_tdms)
            raise FileNotFoundError("Couldn't find calcium video tdms at: " + self.ca_video_tdms)

        metadata_exists = check_file_exists(self.ca_video_metadata_tdms)
        if not metadata_exists:
            logging.warning("Couldn't find calcium metadata tdms at: " + self.ca_video_metadata_tdms)
            raise FileNotFoundError("Couldn't find calcium metadata tdms at: " + self.ca_video_metadata_tdms+
                    "\nPipeline assumes that metadata and video files are saved in the same data_folder")

    def run(self):
        """
            Runs the pipeline
        """
        cameraname, experiment_name = self.check_for_dropped_frames()

        # Convert to mp4
        if self.run_conversion:
            self.convert_video_tdms()

        # If the converted video doesn't exist, we can't correct the rotation
        if not check_file_exists(self.ca_video_path):
            logging.warning("No converted video path exists, terminating analysis")
            raise FileExistsError("No converted video analysis exists, please convert video \
                            or set run_conversion=True")

        # Extract timing of blue and violet led frames from the analog inputs
        if self.extract_frame_times:
            frame_starts = self.extract_frames(experiment_name)
        else:
            frame_starts = None

        if self.debug and frame_starts is not None:
            logging.disable(sys.maxsize)
            inputs = get_analog_inputs_clean_dataframe(self.analog_inputs_tdms, is_opened=False)
            plot_frame_times(inputs[self.camera_triggers_channel],
                            inputs[self.blue_led_triggers_channel], 
                            inputs[self.violet_led_triggers_channel], frame_starts)


    def check_for_dropped_frames(self):
        """
            Checks if dropped frames were dropped during the recording
        """
        logging.info("Checking dropped frames...")
        camera_name = self.ca_video_tdms.split("(0)-")[-1].split(".tdms")[0]
        experiment_name = self.ca_video_tdms.split("(0)-")[0]
        notdropped = check_mantis_dropped_frames(self.data_folder, camera_name, experiment_name, 
                                skip_analog_inputs=True, verbose=False)
        if notdropped:
            logging.info("No frames were dropped")

        return camera_name, experiment_name

                
    def convert_video_tdms(self):
        """
            Converts video .tdms to mp4.
        """
        if check_file_exists(self.ca_video_path):
            logging.info("Converted video file exists already, skipping conversion")
        else:
            logging.info("Converting .tdms video to .mp4")
            convert(self.ca_video_tdms, self.ca_video_metadata_tdms, fps=self.fps, output_path=self.ca_video_path)
            logging.info("Video converted, saved at: {}".format(self.ca_video_path))

    def correct_fiberbundle_rotation(self):
        # TODO make code to correct the rotation of the fiber bundle. 
        # Waiting for final cables before testing this. 

        """
            Likely we will use either bespoke code or DLC to find a number of points in the image, 
            then use the common coordinates behaviour to rotate to a template fiber.

            Then from the registered image just use pre-determined ROIs to extract fluorescence.

            Need to find ways to validate this, e.g. generate N random frames with template and frame overlayed
        """

    def extract_frames(self, experiment_name):
        self.frames_times_file = os.path.join(self.data_folder, experiment_name+"frame_times.json")

        if check_file_exists(self.frames_times_file) and not self.overwrite:
            logging.info("Frames times file exists already, skipping. ")
            return load_json(self.frames_times_file)
        else:
            logging.info("Extracting frame times.")


        # Load analog inputs
        logging.info("Loading analog inputs tdms, might take a while...")
        inputs = get_analog_inputs_clean_dataframe(self.analog_inputs_tdms, is_opened=False)

        # Extract frame start time
        frame_starts = {}
        for name, channel in zip(['calcium_camera', 'blue_led', 'violet_led'], 
                        [self.camera_triggers_channel, self.blue_led_triggers_channel, self.violet_led_triggers_channel]):
            frame_starts[name] = [int(x) for x in list(get_frames_times_from_squarewave_signal(inputs[channel].values))]

        # Save
        save_json(self.frames_times_file, frame_starts, append=False)
        logging.info("Saved frame times at self.frames_times_file")
        return frame_starts



if __name__ == '__main__':
    pipe = Pipeline(data_folder, ca_video_tdms,
                    ca_video_path=None, fps=100,
                    overwrite=True, run_conversion=True,
                    extract_frame_times=True, logging=False, 
                    analog_inputs_tdms=analog_inputs_tdms,
                    camera_triggers_channel=camera_triggers_channel,
                    blue_led_triggers_channel=blue_led_triggers_channel,
                    violet_led_triggers_channel=violet_led_triggers_channel,
                    debug=True)

    pipe.run()
