import pdb
from typing import Tuple
import logging
import json

import numpy as np
import cv2

from utils.utils import baseClass
from utils.Detectors import Detectors
from utils.trajectory import TrajectoryRecorder

class BaseVO(baseClass):
    identity_proj_mat = np.concatenate([np.eye(3), np.zeros([3,1])], axis=1)
    def __init__(self, K:np.ndarray) -> None:
        super().__init__()
        self._K = K
        self._trajectory = TrajectoryRecorder()
        self._args = self._parse_args()

    def _parse_args(self):
        with open("./configure.json") as fp:
            args = json.load(fp)
        return args

    def _KLT_track(
        self, 
        img0:np.ndarray, img1:np.ndarray, 
        ref_kps:Tuple[cv2.KeyPoint], 
        backtrack_check:bool=True, 
        **KLT_args
    ) -> Tuple[cv2.KeyPoint]:
        """
        :param img0: np.ndarray, original image
        :param img1: np.ndarray, new image to track
        :param ref_kps: keypoints from img0
        :param backtrack_check: bool=True, does the tracker back track the keypoints back to img0 to check its validation 

        :return: n_kp1, selected keypoints for img0
        :return: n_kp2, selected keypoints for img1
        :return: mask, match mask, 
        """
        # forward tracking
        kp1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, ref_kps, None, **KLT_args)
        mask = self._get_in_img_mask(img1, kp1)

        # back tracking
        if backtrack_check:
            kp0, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, kp1, None, **KLT_args)
            dist = abs(ref_kps - kp0).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
            back_track_mask = dist < self._extract_param(KLT_args, "back_track_error", 5)
            mask = np.logical_and(mask, back_track_mask)

        valid_kp0 = ref_kps[mask]
        valid_kp1 = kp1[mask]

        return valid_kp0, valid_kp1, mask

    def _get_in_img_mask(self, img1:np.ndarray, kps):
        height, width = img1.shape
        left_mask = kps[:, 0] >= 0
        right_mask = kps[:, 0] <= width-1
        up_mask = kps[:, 1] >= 0
        bottom_mask = kps[:, 1] <= height-1
        width_mask = np.logical_and(left_mask, right_mask)
        height_mask = np.logical_and(up_mask, bottom_mask)
        mask = np.logical_and(width_mask, height_mask)
        return mask

    def _build_trans_mat(self, R:np.ndarray, T:np.ndarray)->np.ndarray:
        """
        build the 4*4 transform matrix from R and T
        """
        trans_mat = np.concatenate([R, T], axis=1)
        trans_mat = np.concatenate([trans_mat, np.array([[0,0,0,1]])], axis=0)
        return trans_mat

    def _triangulate(
        self, extrinsic_0W, extrinsic_1W, 
        kps0, kp1, threshold=None
    )->Tuple[np.ndarray, np.ndarray]:
        
        points3D = cv2.triangulatePoints(
            self._K@extrinsic_0W, self._K@extrinsic_1W, 
            kps0, kp1
        )
        points3D = points3D / points3D[3]

        trans_0W = np.concatenate((extrinsic_0W, np.array([[0,0,0,1]])), axis=0)
        trans_1W = np.concatenate((extrinsic_1W, np.array([[0,0,0,1]])), axis=0)
        points3D_0 = (trans_0W @ points3D)
        points3D_1 = (trans_1W @ points3D)

        # Find points in front of the camera
        judge1=np.where(points3D_0[2]>0) and np.where(points3D_1[2]>0)[0]
        points3D =points3D[:, judge1][:3]

        # Calculate points angle between two camera poses
        T_W_0W = np.linalg.inv(trans_0W)[:3, 3:]
        T_W_1W = np.linalg.inv(trans_1W)[:3, 3:]
        T_W_P0 = points3D - T_W_0W
        T_W_P1 = points3D - T_W_1W
        inner_product = np.sum(T_W_P0*T_W_P1, axis=0)
        T_W_P0_norm = np.linalg.norm(T_W_P0, axis=0)
        T_W_P1_norm = np.linalg.norm(T_W_P1, axis=0)
        angle = inner_product / (T_W_P0_norm * T_W_P1_norm)

        # filtering out points with big enough angle
        # with self adjustment
        if threshold is None:
            threshold=8*np.pi/180
            judge2 = None
            while threshold>=self._args.get("triangulate_agnle_lower_bound")*np.pi/180:
                judge2 = np.where(angle<np.cos(threshold))[0]
                if judge2.size==0:
                    threshold=threshold/2
                    continue 
                if judge2.size > 100:
                    break
                threshold=threshold/2
        else:
            threshold=threshold*np.pi/180
            judge2 = np.where(angle<np.cos(threshold))[0]

        logging.info('final threshold: {}'.format(threshold*180/np.pi))
        return points3D[:, judge2], judge1[judge2]


class VisualOdometry(BaseVO):
    def __init__(self, K:np.ndarray) -> None:
        super(VisualOdometry, self).__init__(K)

    # TODO: 0.0 bootstrap_KLT
    def bootstrap_KLT(
        self, 
        img0: np.ndarray, img1:np.ndarray, 
        kps_detector: Detectors, 
        **KLT_args
    ) -> bool:
        """
        :param img0: 
        :param img1:
        :param kps_detector: Detectors
        :param KLT_args: dict, parameters for KLT tracker
        """
        kps0 = kps_detector.detect(img0, 5, 3, 20, **KLT_args)
        _, kps1, mask = self._KLT_track(img0, img1, kps0)

        # calculate Essential Matrix
        E, ransac_mask = cv2.findEssentialMat(
            kps0, kps1, self._K, cv2.RANSAC, 0.999, 1
        )
        _, R_10, T_1_01, _ = cv2.recoverPose(E, kps0, kps1, self._K)
        ransac_mask = ransac_mask[:, 0] == 1

        trans_mat_10 = self._build_trans_mat(R_10, T_1_01)
        matched_pts_0 = kps0[ransac_mask]
        matched_pts_1 = kps1[ransac_mask]
        cloud_0, triangulate_indices = self._triangulate(
            self.identity_proj_mat, trans_mat_10, matched_pts_0.T, matched_pts_1.T, 
            threshold=self._args.get("triangulate_angle_init_thresh")
        )

        if triangulate_indices.size > 100:
            self._update_key_frame()
        else:
            self._update_pose()




    # TODO: 0.1 bootstrap_Matching
    def bootstrap_Matching(self):
        raise NotImplementedError

    # TODO: 0.0 Update KLT
    def update_KLT(self):
        raise NotImplementedError

    # TODO: 0.1 update matching
    def update_matching(self):
        raise NotImplementedError

    

