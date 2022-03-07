from http.client import UnimplementedFileMode
import numpy as np

from utils.utils import baseClass

# TODO: trajectory Recorder
class TrajectoryRecorder(baseClass):
    def __init__(self) -> None:
        super().__init__()
        self._current_R = np.zeros((3,3))   # w.r.t world frame, R_0W
        self._current_T = np.zeros((3,1))   # w.r.t world frame, T_W_0W

        self._key_frame_R = np.zeros((3,3)) # w.r.t world frame
        self._key_frame_T = np.zeros((3,1)) # w.r.t world frame

        self._traj = []

    def update_key_frame(self, R_10, T_1_01):
        self._key_frame_R, self._key_frame_T = self.update_pose(R_10, T_1_01)

    def update_pose(self, R_10, T_1_01):
        trans_W_0W = self._build_trans_mat(self._current_R, self._current_T)
        self._current_R = R_10 @ self._current_R

        # translation
        T_0_01 = R_10.T @ T_1_01
        self._current_T =self._transform(T_0_01, trans_W_0W)

        self._traj.append(self._build_trans_mat(self._current_R, self._current_T))

        return self._current_R, self._current_T

    def _transform(self, T, trans_mat):
        """
        :param T: np.ndarray(3, 1)
        :param trans_mat: np.ndarray(4, 4)

        :return : np.ndarray(3, 1)
        """
        T = np.concatenate([T, np.zeros((1, 1))], axis=0)
        return (trans_mat @ T)[:3, :]

    def _build_trans_mat(self, R:np.ndarray, T:np.ndarray)->np.ndarray:
        """
        build the 4*4 transform matrix from R and T
        """
        trans_mat = np.concatenate([R, T], axis=-1)
        trans_mat = np.concatenate([trans_mat, np.array([[0,0,0,1]])], axis=0)
        return trans_mat