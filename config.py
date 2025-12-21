camera_name = [
    "iPhone15_wide_1",
    "iPhone15_selfie",
    "NEXIGO",
    "A53_front_normal",
    "A53_front_wide", # doesn't calibrate
    "A53_back_ultra_wide"
]



data_settings = {
    "calibration_file": "camera_calibration_iPhone15_wide_1.npz",# is only used in calibration.
    "camera_index": 0, # Select your camera device index, this depends on the computer. The default is 0, if 0 cannot detect the camera, try 1.
    "camera_name": camera_name[5],# is only used in calibration.
    "ppg_input_file": "pulse_data.csv",# modify to your recorded ppg data's name format.
    "record_duration": 60 # record duration time, second.
}