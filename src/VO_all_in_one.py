import pdb
from typing import Tuple
import logging
import json
import copy
import matplotlib

import numpy as np
import cv2

import utils
from utils.utils import baseClass
from utils.Detectors import Detectors
from utils.trajectory import TrajectoryRecorder
from utils.KLTTracker import KLTTracker

color = np.random.randint(0, 255, (100, 3))

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
        return args["VO"]

    def _build_trans_mat(self, R:np.ndarray, T:np.ndarray)->np.ndarray:
        """
        build the 4*4 transform matrix from R and T
        """
        trans_mat = np.concatenate([R, T], axis=1)
        trans_mat = np.concatenate([trans_mat, np.array([[0,0,0,1]])], axis=0)
        return trans_mat

    def _triangulate(
        self, trans_0W, trans_1W, 
        kps0, kps1, threshold=None
    )->Tuple[np.ndarray, np.ndarray]:
        extrinsic_0W = trans_0W[:3]
        extrinsic_1W = trans_1W[:3]
        points3D = cv2.triangulatePoints(
            self._K@extrinsic_0W, self._K@extrinsic_1W, 
            kps0, kps1
        )
        points3D = points3D / points3D[3]

        points3D_0 = (trans_0W @ points3D)
        points3D_1 = (trans_1W @ points3D)

        # Find points in front of the camera
        judge1=np.where(points3D_0[2]>0) and np.where(points3D_1[2]>0)[0]
        points3D =points3D[:, judge1][:3]

        # Calculate points angle between two camera poses
        T_W_W0 = np.linalg.inv(trans_0W)[:3, 3:]
        T_W_W1 = np.linalg.inv(trans_1W)[:3, 3:]
        T_W_0P = points3D - T_W_W0
        T_W_1P = points3D - T_W_W1
        inner_product = np.sum(T_W_0P*T_W_1P, axis=0)
        T_W_0P_norm = np.linalg.norm(T_W_0P, axis=0)
        T_W_1P_norm = np.linalg.norm(T_W_1P, axis=0)
        angle = inner_product / (T_W_0P_norm * T_W_1P_norm)

        # filtering out points with big enough angle
        # with self adjustment
        if threshold is None:
            threshold=8*np.pi/180
            judge2 = None
            while threshold>=self._args.get("triangulate_angle_lower_bound")*np.pi/180:
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

    def _img_preprocess(self, src_img:np.ndarray) -> np.ndarray:
        shape = src_img.shape
        if len(shape)==3 and shape[-1]==3:
            return cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        else:
            return src_img

    def _update_pose(self, R_10, T_1_10):
        # TODO: 0.0.0 update_pose
        pass

    def _update_key_frame(self, R_10, T_1_10):
        # TODO: 0.0.1 update_key_frame
        pass


class VisualOdometry(BaseVO):
    def __init__(self, K:np.ndarray) -> None:
        super(VisualOdometry, self).__init__(K)
        self._KLT_tracker = KLTTracker(K)

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
        canvas = img1.copy()    # Cans for drawing

        img0 = self._img_preprocess(img0)
        img1 = self._img_preprocess(img1)

        kps0 = kps_detector.detect(img0, 5, 3, 50, **KLT_args)
        kps0, kps1, mask = self._KLT_tracker.track(img0, img1, kps0)

        # calculate Essential Matrix
        kps0_np = cv2.KeyPoint_convert(kps0)
        kps1_np = cv2.KeyPoint_convert(kps1)
        
        E, ransac_mask = cv2.findEssentialMat(
            kps0_np,kps1_np, 
            self._K, cv2.RANSAC, 0.999, 1
        )
        # T_1_10 means vector from 1 to 0 in refrence frame 1
        _, R_10, T_1_10, _ = cv2.recoverPose(E, kps0_np, kps1_np, self._K)
        ransac_mask = ransac_mask[:, 0] == 1

        trans_mat_10 = self._build_trans_mat(R_10, T_1_10)
        matched_pts_0_np = kps0_np[ransac_mask]
        matched_pts_1_np = kps1_np[ransac_mask]

        # draw the tracks
        # mask = np.zeros_like(canvas)
        for i, (old, new) in enumerate(zip(matched_pts_0_np, matched_pts_1_np)):
            a, b = new.ravel()
            c, d = old.ravel()
            # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            # canvas = cv2.circle(canvas, (int(a), int(b)), 2, color[i].tolist(), -1)
            mask = cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), color=(0,0,255), thickness=2)
        # img = cv2.add(canvas, mask)
        cv2.imwrite('match.jpg', canvas)
        pdb.set_trace()

        cloud_0, triangulate_indices = self._triangulate(
            self._build_trans_mat(np.eye(3), np.zeros([3,1])), trans_mat_10, 
            matched_pts_0_np.T, matched_pts_1_np.T, 
            threshold=self._args.get("triangulate_angle_init_thresh")
        )
        
        if triangulate_indices.size > 100:
            self._update_key_frame(R_10, T_1_10)
            return True
        else:
            self._update_pose(R_10, T_1_10)
            return False




    # TODO: 0.1 bootstrap_Matching
    def bootstrap_Matching(self):
        raise NotImplementedError

    # TODO: 0.0 Update KLT
    def update_KLT(self):
        raise NotImplementedError

    # TODO: 0.1 update matching
    def update_matching(self):
        raise NotImplementedError

    

