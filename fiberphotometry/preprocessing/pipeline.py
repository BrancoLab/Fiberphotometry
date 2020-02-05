import os
import logging

from fancylog import fancylog
import fancylog as package

from behaviour.tdms.mantis_videoframes_test import check_mantis_dropped_frames
from behaviour.tdms.utils import get_analog_inputs_clean_dataframe, get_analog_inputs_clean
from behaviour.utilities.signals import get_frames_times_from_squarewave_signal
from tdmstovideo.converter import convert
from fcutils.file_io.utils import check_file_exists, get_file_name
from fcutils.file_io.io import save_yaml







class Pipeline:
    def __init__(self, data_folder, ca_video_tdms,
                    ca_video_path=None, fps=100,
                    overwrite=False, run_conversion=True,
                    extract_frame_times=True, **kwargs):
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
            :param kwargs: used to pass params for extract_frame_times: [analog_inputs_tdms, camera_triggers_channel,
                        bue_led_triggers_channel, violet_led_triggers_channel]
            
        """

    



# ---------------------------------------------------------------------------- #
#                                    PARAMS                                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Variables -------------------------------- #

data_folder = "Z:\\swc\\branco\\rig_photometry\\tests\\200205\\200205_mantis_test_longer_exposure_noaudio"
ca_video_tdms = "FP_calcium_and_behaviour(0)-FP_calcium_camera.tdms"
ca_video_path = None # If left as None it will save it the same folder and with the same name as the tdms file
FPS = 100
# ---------------------------------- Options --------------------------------- #

OVERWRITE = False # If true it will overwrite the results of a previous analysis
RUN_CONVERSION = True # If true it will convert the tdms video to mp4
EXTRACT_FRAME_TIMES = True # If true it will extract the time of blue and violed LEDs frames from analog inputs (slow)


# ------------------------------ Extra variables ----------------------------- #
# ? To extrac frame times
analog_inputs_tdms = "FP_calcium_and_behaviour(0).tdms"
camera_triggers_channel='FP_calciumcamera_triggers_reading'
bue_led_triggers_channel='camera_triggers_channel'
violet_led_triggers_channel='camera_triggers_channel'



def setup
# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #
# Get complete file paths
if ca_video_path is None:
    ca_video_path = os.path.join(data_folder, get_file_name(ca_video_tdms)+".mp4")
ca_video_tdms = os.path.join(data_folder, ca_video_tdms)
ca_video_metadata_tdms = os.path.join(data_folder, get_file_name(ca_video_tdms)+"meta.tdms")

if EXTRACT_FRAME_TIMES:
    analog_inputs_tdms = os.path.join(data_folder, analog_inputs_tdms)
    check_file_exists(analog_inputs_tdms, raise_error=True)

# Start logging

fancylog.start_logging(data_folder, package, verbose=True, filename="fp_preprocessing",
        multiprocessing_aware=False, write_cli_args=False)
logging.info("Starting to preprocess files:")
logging.info(data_folder, ca_video_tdms, ca_video_metadata_tdms, ca_video_path)






# ---------------------------------------------------------------------------- #
#                           CHECK EVERYTHING IS OKAY                           #
# ---------------------------------------------------------------------------- #

# ----------------------------- Check files exist ---------------------------- #
video_exists = check_file_exists(ca_video_tdms)
if not video_exists:
    raise FileNotFoundError("Couldn't find calcium video tdms at: " + ca_video_tdms)

metadata_exists = check_file_exists(ca_video_metadata_tdms)
if not metadata_exists:
    raise FileNotFoundError("Couldn't find calcium metadata tdms at: " + ca_video_metadata_tdms+
            "\nPipeline assumes that metadata and video files are saved in the same data_folder")


# --------------------------- Check dropped frames --------------------------- #
if check_file_exists(ca_video_path) and not overwrite:
    logging.info("Found a converted video file, skipping dropped frames checked and conversion")
else:
    logging.info("Checking dropped frames...")
    camera_name = ca_video_tdms.split("(0)-")[-1].split(".tdms")[0]
    experiment_name = ca_video_tdms.split("(0)-")[0]
    notdropped = check_mantis_dropped_frames(data_folder, camera_name, experiment_name, 
                            skip_analog_inputs=True, verbose=False)
    if notdropped:
        logging.info("No frames were dropped")

    # ? tdms -> conversion
    if RUN_CONVERSION:
        logging.info("Converting .tdms video to .mp4")
        convert(ca_video_tdms, ca_video_metadata_tdms, fps=FPS, output_path=ca_video_path)
        logging.info("Video converted")

    


# ---------------------------------------------------------------------------- #
#                               CORRECT ROTATION                               #
# ---------------------------------------------------------------------------- #

# TODO make code to correct the rotation of the fiber bundle. 
# Waiting for final cables before testing this. 

"""
    Likely we will use either bespoke code or DLC to find a number of points in the image, 
    then use the common coordinates behaviour to rotate to a template fiber.

    Then from the registered image just use pre-determined ROIs to extract fluorescence.

    Need to find ways to validate this, e.g. generate N random frames with template and frame overlayed
"""




# ---------------------------------------------------------------------------- #
#                              EXTRACT FRAME TIMES                             #
# ---------------------------------------------------------------------------- #
if EXTRACT_FRAME_TIMES:
        logging.info("Extracting frame times.")

        frames_times_file = os.path.join(data_folder, experiment_name+"frame_times.yml")

        if check_file_exists(frames_times_file) and not overwrite:
            logging.info("Frames times file exists already, skipping. ")
        else:
            logging.info("No frames times file found, running analysis")

        # Load analog inputs
        inputs = get_analog_inputs_clean_dataframe(analog_inputs_tdms, is_opened=False)

        # Extract frame start time
        frame_starts = {}
        for name, channel in zip(['calcium_camera', 'blue_led', 'violet_led'], [camera_triggers_channel, bue_led_triggers_channel, violet_led_triggers_channel]):
            frame_starts[name] = get_frames_times_from_squarewave_signal(inputs[channel].values, debug=False)

        # Save
        save_yaml(frames_times_file, content, append=False)


