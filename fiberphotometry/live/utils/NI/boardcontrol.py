import numpy as np
import nidaqmx

try:
    import utils.NI.nidaq as iodaq
    ni_ready = True
except Exception as e:
    print("Could not set up NI communication: \n")
    print(e)
    ni_ready = False


class NImanager():
    HIGH = np.array([1]).astype(np.uint8)
    LOW = np.array([0]).astype(np.uint8)
    fHIGH = np.array([1]).astype(np.float64)
    fLOW = np.array([0]).astype(np.float64)

    def __init__(self, **kwargs):
        if ni_ready:
            # Start Analog Output tasks to send triggers to LEDs cubes (for fiber photometry)
            self.blue_do = iodaq.DigitalOutput(self.niboard_config['ni_device'],
                                    self.niboard_config['blue_led_trigger_port'],
                                    self.niboard_config['blue_led_trigger_line'],)
            self.blue_do.StartTask()
            _ = self.blue_do.write(self.LOW)

            self.violet_do = iodaq.DigitalOutput(self.niboard_config['ni_device'],
                                    self.niboard_config['violet_led_trigger_port'],
                                    self.niboard_config['violet_led_trigger_line'],)
            self.violet_do.StartTask()
            _ = self.violet_do.write(self.LOW)

            # Start digital output for camera
            self.camera_do = iodaq.DigitalOutput(self.niboard_config['ni_device'],
                                    self.niboard_config['camera_trigger_port'],
                                    self.niboard_config['camera_trigger_line'],)
            self.camera_do.StartTask()
            _ = self.camera_do.write(self.LOW)

            # Create stuff for the LDR
            # power output
            self.ldr_power = nidaqmx.Task()
            self.ldr_power.ao_channels.add_ao_voltage_chan('Dev3/ao1','mychannel',0,5)
            self.ldr_power.write(5.0)


            # analog input
            self.ldr_ai = nidaqmx.Task()
            self.ldr_ai.ai_channels.add_ai_voltage_chan('Dev3/ai0','ldrinput')

            self._analog_chs = [self.ldr_power, self.ldr_ai]


    # ------------------------------ UPDATE TRIGGERS and ANALOGS ----------------------------- #
    def read_ldr_signal(self):
        self.ldr_power.write(5.0)
        self.ldr_signal_dump.append(self.ldr_ai.read())

    def toggle_leds(self, switch_on=[], switch_off=[]):
        for on in switch_on:
            on.write(self.HIGH)
        for off in switch_off:
            off.write(self.LOW)

    def trigger_frame(self):
        self.camera_do.write(self.HIGH)
        self.camera_do.write(self.LOW)

    def switch_leds_off(self):
        self.blue_do.write(self.LOW)
        self.violet_do.write(self.LOW)

    def switch_leds_on(self):
        self.blue_do.write(self.HIGH)
        self.violet_do.write(self.HIGH)
        
    def update_triggers(self):
        """
            [Sends a trigger to the camera and switches ON and OFF the LEDs]
        """
        if ni_ready:
            if self.frame_count % 2 == 0:
                self.blue_do.write(self.HIGH)
                self.violet_do.write(self.LOW)
            else:
                self.blue_do.write(self.LOW)
                self.violet_do.write(self.HIGH)

            self.trigger_frame()
            self.read_ldr_signal()


    def close_analog_tasks(self):
        for ch in self._analog_chs:
            ch.stop()
