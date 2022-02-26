from http.client import UnimplementedFileMode
import numpy as np

from utils.utils import baseClass

# TODO: trajectory Recorder
class TrajectoryRecorder(baseClass):
    def __init__(self) -> None:
        super().__init__()
        self._current_R = np.zeros((3,3))   # w.r.t world frame
        self._current_T = np.zeros((3,1))   # w.r.t world frame

        self._key_frame_R = np.zeros((3,3)) # w.r.t world frame
        self._key_frame_T = np.zeros((3,1)) # w.r.t world frame

        self._traj_R = []

    def update_key_frame(self, R_relative, T_relative):
        raise NotImplementedError

    def update_pose(self, R_relative, T_relative):
        raise NotImplementedError