from ..interactor import interactor


class Router:
    def __init__(self):
        self.__interactor=interactor.Interactor()


    def start_recording(self):
        self.__interactor.start_recording()

    def stop_recording(self):
        self.__interactor.stop_recording()

    def get_current_window(self):
        return self.__interactor.get_current_window()

    def get_LSL_clock(self):
        return self.__interactor.get_LSL_clock()

    def get_lsl_offset(self):
        return self.__interactor.get_lsl_offset()

    def get_server_clock(self):
        return self.__interactor.get_server_clock()

    def set_raw_eeg_file_path(self):
        self.__interactor.set_raw_eeg_file_path()

    def start_eye_tracker_recording(self):
        self.__interactor.start_eye_tracker_recording()

    def stop_eye_tracker_recording(self):
        self.__interactor.stop_eye_tracker_recording()