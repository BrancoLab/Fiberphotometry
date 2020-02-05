import sys
sys.path.append("./")
import os

from fcutils.file_io.file import load_yaml


class SettingsParser:
    def __init__(self, settings_file=None):
        if settings_file is None:
            settings_file = "config.yaml"
        else: 
            if not os.isfile(settings_file):
                raise FileNotFoundError("Could not find settings file at {}".format(settings_file))

        # Store the parameters
        params = load_yaml(settings_file)
        for name, value in params.items():
            setattr(self, name, value)


