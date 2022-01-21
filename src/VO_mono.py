from typing import Tuple, final
import cv2
import numpy as np
import logging
from enum import Enum, auto
from numpy.linalg.linalg import _tensorsolve_dispatcher
from scipy.spatial.transform import Rotation as R
import pdb
from abc import ABC, abstractclassmethod
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from klt_pnp.code.Initialization import Harris


F_detectors = {'FAST': cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
                'SIFT': cv2.SIFT_create(),
                'ORB': cv2.ORB_create(), 
                } 

match_metrics = {
    'FAST': cv2.NORM_L2,
    'SIFT': cv2.NORM_L2,
    'ORB': cv2.NORM_HAMMING
}

lk_params = dict(winSize=(21, 21),  # Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker)
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

KLT_MATCHING_DIFF = 5
INIT_THRESH = 4
TRIANG_LOWER_THRESH = 0.3

KLT_PTS_LOWER_BOUND = 20
TRIANGULATE_PTS_LOWER_BOUND = 50
# class VO_STAGE(Enum):
#     FIRST_FRAME=auto()
#     SECOND_FRAME=auto()
#     DEFAULT=auto()

class MONO_VO(ABC):
    identity_proj_mat = np.concatenate([np.eye(3), np.zeros([3,1])], axis=1)
    def __init__(self, K:np.ndarray, f_extract_type:str="SIFT", show_traj:bool=False) -> None:
        """
        :param K: internal calibration matrix
        :param f_extract_type: ['SIFT', 'ORB']
        """
        self.K = K
        self._last_frame = None    
        self._R_1W = np.eye(3)
        self._T_1_W1 = np.zeros([3,1])
        self._trans_mat_1W = self._build_trans_mat(self._R_1W, self._T_1_W1) # 4*4 transformation matrix

        self._last_key_pts = None
        self._last_des = None
        self._last_matched_pts_idx = None
        self._last_matched_pts = None

        self._T_vectors = []
        self._R_matrices = []
        self._traj = [np.zeros([3,1])]
        self._point_cloud_all= None
        self._last_cloud_W = None
        self._last_cloud_W_matched = None
        self._feature_extracter = F_detectors[f_extract_type]
        self._f_extract_type = f_extract_type

        match_metric = match_metrics[f_extract_type]
        self._feature_matcher = cv2.BFMatcher(match_metric, crossCheck=True)
        harris_patch_size = 3
        harris_kappa = 0.01
        query_keypoint_num = 8
        nonmaximum_supression_radius = 5
        self._harris_detector = Harris(
            harris_patch_size, harris_kappa, 
            query_keypoint_num, nonmaximum_supression_radius
        )

        self._show_traj_flag = show_traj
        self._img_to_show = None
    
    def _extract_features(self, img:np.ndarray, Convert=False)->Tuple[Tuple, np.ndarray]:
        if self._f_extract_type in ['SIFT', 'ORB']:
            feature_pts, feature_des = self._feature_extracter.detectAndCompute(img, None)
        else: 
            feature_pts = self._feature_extracter.detect(img, None)
            feature_des = None

        if Convert:
            feature_pts = cv2.KeyPoint_convert(feature_pts)
        return feature_pts, feature_des

    def _build_trans_mat(self, R:np.ndarray, T:np.ndarray)->np.ndarray:
        """
        build the 4*4 transform matrix from R and T
        """
        trans_mat = np.concatenate([R, T], axis=1)
        trans_mat = np.concatenate([trans_mat, np.array([[0,0,0,1]])], axis=0)
        return trans_mat

    def _build_homo_points(self, points3D:np.ndarray)->np.ndarray:
        point_num = points3D.shape[1]
        points_homo = np.concatenate(
            [points3D, np.ones([1, point_num])], 
            axis=0
        )
        return points_homo

    def _triangulate(
        self, extrinsic_1W, extrinsic_2W, 
        points_1, points_2, threshold=None
    )->Tuple[np.ndarray, np.ndarray]:
        
        points3D = cv2.triangulatePoints(
            self.K@extrinsic_1W, self.K@extrinsic_2W, 
            points_1, points_2
        )
        points3D = points3D / points3D[3]
        # self.draw_cloud(points3D)
        # self.draw_match(self._last_frame, points_1.T, points_2.T)

        trans_1W = np.concatenate((extrinsic_1W, np.array([[0,0,0,1]])), axis=0)
        trans_2W = np.concatenate((extrinsic_2W, np.array([[0,0,0,1]])), axis=0)
        points3D_1 = (trans_1W @ points3D)
        points3D_2 = (trans_2W @ points3D)

        # pdb.set_trace()

        judge1=np.where(points3D_1[2]>0) and np.where(points3D_2[2]>0)[0]
        # # pdb.set_trace()
        points3D =points3D[:, judge1][:3]

        T_W_1W = np.linalg.inv(trans_1W)[:3, 3:]
        T_W_2W = np.linalg.inv(trans_2W)[:3, 3:]

        T_W_P1 = points3D - T_W_1W
        T_W_P2 = points3D - T_W_2W

        inner_product = np.sum(T_W_P1*T_W_P2, axis=0)
        T_W_P1_norm = np.linalg.norm(T_W_P1, axis=0)
        T_W_P2_norm = np.linalg.norm(T_W_P2, axis=0)
        angle = inner_product / (T_W_P1_norm * T_W_P2_norm)

        if threshold is None:
            threshold=8*np.pi/180
            judge2 = None
            while threshold>=TRIANG_LOWER_THRESH*np.pi/180:
                judge2 = np.where(angle<np.cos(threshold))[0]
                # pdb.set_trace()
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
        # if judge2.size == 0:
        #     pdb.set_trace()
        return points3D[:, judge2], judge1[judge2]
        # return points3D, judge1

    def initialization(
        self, img0:np.ndarray, img1:np.ndarray, 
        rgb_img=None, method:str="MATCHING", 
        **kargs
    )->bool:
        """ 
        
        Initialize the Visual Odometry using the first two images. 
        Extract the R and T of these two images and triangulate 3D points. 
        
        :param img0: first image of the dataset
        :param img1: second image of the dataset
        :param method: choose from ["MATCHING", "KLT"] for init method
        
        :return:
        """
        if img0 is None:
            img0 = self._last_frame

        if method == "MATCHING":
            # key points is a tuple of cv2.keyPoint. 
            # can be convert to ndarray using cv2.KeyPoint_convert(kps)
            key_pts_1, des_1 = self._extract_features(img0, Convert=False)
            key_pts_2, des_2 = self._extract_features(img1, Convert=False)

            # DMatch.distance - Distance between descriptors. The lower, the better it is.
            # DMatch.trainIdx - Index of the descriptor in train descriptors
            # DMatch.queryIdx - Index of the descriptor in query descriptors
            # DMatch.imgIdx - Index of the train image.
            match_res = self._feature_matcher.match(des_1, des_2)

            # convert from tuple of keypoints to ndarray
            key_pts_1_np = cv2.KeyPoint_convert(key_pts_1)
            key_pts_2_np = cv2.KeyPoint_convert(key_pts_2)

            # select chosen keypoints
            matched_pts_idx_1 = np.array([match.queryIdx for match in match_res])
            matched_pts_idx_2 = np.array([match.trainIdx for match in match_res])
            matched_pts_1 = key_pts_1_np[matched_pts_idx_1]
            matched_pts_2 = key_pts_2_np[matched_pts_idx_2]

            E, ransac_mask = cv2.findEssentialMat(
                matched_pts_1, matched_pts_2, 
                self.K, cv2.RANSAC, 0.999, 1
            )
            _, R_21, T_2_12, _ = cv2.recoverPose(E, matched_pts_1, matched_pts_2, self.K)
            ransac_mask = ransac_mask[:,0]==1
            
            trans_mat_21 = self._build_trans_mat(R_21, T_2_12)
            matched_pts_1 = matched_pts_1[ransac_mask]
            matched_pts_2 = matched_pts_2[ransac_mask]
            cloud_1, triangulate_indices = self._triangulate(
                self.identity_proj_mat, trans_mat_21[:3], 
                matched_pts_1.transpose(), 
                matched_pts_2.transpose(), 
                threshold=INIT_THRESH
            )

            if triangulate_indices.size < 100:
                logging.info("continue initialize with {} points".format(triangulate_indices.size))
                return False


        elif method == "KLT":
            # TODO: initialize the system with KLT method
            raise NotImplementedError

        self._last_frame = img1
        self._R_1W = R_21 @ self._R_1W
        self._T_1_W1 = T_2_12 + R_21@self._T_1_W1
        self._R_matrices.append(self._R_1W)
        self._T_vectors.append(self._T_1_W1)
        self._trans_mat_1W = trans_mat_21 @ self._trans_mat_1W
        self._traj.append(-self._R_1W.transpose()@self._T_1_W1)
        # print(self._traj)
        cloud_1_homo = np.concatenate(
            [cloud_1, np.ones([1, cloud_1.shape[1]])],
            axis = 0
        )
        self._last_cloud_W = np.linalg.inv(self._trans_mat_1W) @ cloud_1_homo
        self._last_cloud_W = self._last_cloud_W[:3]
        # self._point_cloud_all = cloud_1
        
        self._last_key_pts = cv2.KeyPoint_convert(
            self._harris_detector.distribute_keypoints(img1)
        )
        self._last_key_pts_sift = key_pts_2

        self._last_matched_pts = matched_pts_2[triangulate_indices]
        self._last_matched_pts_idx = matched_pts_idx_2[triangulate_indices]


        # cloud = np.concatenate([cloud_1, np.ones([1, cloud_1.shape[1]])])
        # reproject_pts = self.K@self._trans_mat_1W[:3]@cloud
        # reproject_pts = reproject_pts / reproject_pts[2]
        # reproject_pts = reproject_pts[:2].T
        # self.draw_match(
        #     img1, reproject_pts,
        #     self._last_matched_pts, rgb_img=rgb_img
        # )
        # pdb.set_trace()


        logging.info("Mono Visual Odometry initilized")
        return True

    def _KLT_featureTracking(
        self, img_0, img_1, key_points_ref
    ):
        """
        :param: img_0: image 0
        :param: img_1: image 1
        :param: px_ref: existing key points

        :return: n_kp1, selected keypoints for img_0
        :return: n_kp2, selected keypoints for img_1
        :return: diff_mean, mean distance the key points move
        :return: match_idx, 
        """
        # Feature Correspondence with Backtracking Check
        # print("key_points_ref: ", key_points_ref.shape)
        kp2, st, err = cv2.calcOpticalFlowPyrLK(img_0, img_1, key_points_ref, None, **lk_params)
        # print("kp2: ", kp2.shape)
        # kp1, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_0, kp2, None, **lk_params)

        # dist = abs(key_points_ref - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
        # match_mask = dist < KLT_MATCHING_DIFF  # Verify which features produced good results by the difference being less
                                # than the fMATCHING_DIFF threshold.
        

        height, width = img_1.shape
        left_mask = kp2[:, 0] >= 0
        right_mask = kp2[:, 0] <= width-1
        up_mask = kp2[:, 1] >= 0
        bottom_mask = kp2[:, 1] <= height-1

        width_mask = np.logical_and(left_mask, right_mask)
        height_mask = np.logical_and(up_mask, bottom_mask)
        mask = np.logical_and(width_mask, height_mask)

        # match_mask = np.logical_and(match_mask, mask)
        match_mask = mask
        n_kp1, n_kp2 = key_points_ref[match_mask], kp2[match_mask]

        # Error Management
        matched_num = match_mask.sum()

        if matched_num <= KLT_PTS_LOWER_BOUND:  # If less than 5 good points, it uses the features obtain without the backtracking check
            print('Warning: No match was good. Returns the list without good point correspondence.')
            return n_kp1, n_kp2, matched_num, None

        # Create new lists with the good features
        return n_kp1, n_kp2, matched_num, match_mask

    @abstractclassmethod
    def update(self, img:np.ndarray, method:str="MATCHING"):
        """
        Update new frame 

        :param img: next frame to recover the pose
        :param method: ['MATCHING', 'KLT']
        """
        pass

    def draw_match(
        self, img:np.ndarray, 
        pts1:np.ndarray, pts2:np.ndarray, 
        rgb_img:np.ndarray=None
    )->np.ndarray:
        pts1 = pts1.astype(int)
        pts2 = pts2.astype(int)
        if img is None:
            return
        if rgb_img is not None:
            img1 = rgb_img.copy()
        else:
            img1 = img.copy()
        # print(pts1)
        for i in range(pts1.shape[0]):
            img1 = cv2.drawMarker(img1, pts1[i].tolist(), color=(255,0,0))    # blue
            img1 = cv2.drawMarker(img1, pts2[i].tolist(), color=(0, 0, 255))
            img1 = cv2.line(img1, pts1[i], pts2[i], color=(0,255,0), thickness=1)
        cv2.imwrite("matching_res.jpg", img1)

        return img1

    def draw_cloud(self, points3D:np.ndarray)->np.ndarray:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D(
            points3D[0],
            points3D[2],
            -points3D[1],
        )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('point_cloud.jpg')
        plt.close()

    @property
    def traj(self)->np.array:
        return np.array(self._traj)[:, :, 0]

    @property
    def point_cloud(self)->np.array:
        return self._point_cloud_all

    @property
    def last_point_cloud(self)->np.array:
        # print(self._last_cloud_W_matched.shape)
        return self._last_cloud_W_matched
        
    @property
    def img_to_show(self)->np.ndarray:
        return self._img_to_show


class MONO_VO_Matching(MONO_VO):
    def __init__(self, K: np.ndarray, f_extract_type: str = "SIFT", show_traj: bool = False) -> None:
        super().__init__(K, f_extract_type=f_extract_type, show_traj=show_traj)

    def _calc_relative_scale(self, last_3d_pts:list, cur_3d_pts:np.ndarray)->float:
        assert last_3d_pts.shape == cur_3d_pts.shape, "points number does not match when recover scale"
        scales = []
        idx_range = last_3d_pts.shape[1]

        for i in range(idx_range):
            last_vec = last_3d_pts[:3, i] - last_3d_pts[:3, i-1]
            cur_vec = cur_3d_pts[:3, i] - cur_3d_pts[:3, i-1]

            if np.linalg.norm(cur_vec)>0:
                scale = np.linalg.norm(last_vec) / np.linalg.norm(cur_vec)
                scales.append(scale)

        scale = np.median(scales)
        return scale

    def update(self, img: np.ndarray, method: str = "MATCHING"):
        logging.info("update new frame")
        if method=='MATCHING':
            # detect and match new features
            key_pts, des = self._extract_features(img, Convert=False)
            match_res = self._feature_matcher.match(self._last_des, des)

            match_img = None
            match_img = cv2.drawMatches(
                self._last_frame, 
                self._last_key_pts,
                img, key_pts, 
                match_res[:100], match_img
            )
            self._img_to_show = match_img

            matched_pts_idx_1 = [match.queryIdx for match in match_res]
            matched_pts_idx_2 = [match.trainIdx for match in match_res]
            last_key_pts_np = cv2.KeyPoint_convert(self._last_key_pts)
            key_pts_np = cv2.KeyPoint_convert(key_pts)
            matched_pts_1 = last_key_pts_np[matched_pts_idx_1]
            matched_pts_2 = key_pts_np[matched_pts_idx_2]

            # TODO: decide if skip or not

            # find E, recover pose, triangulation
            E, _ = cv2.findEssentialMat(
                matched_pts_1, matched_pts_2, 
                self.K, cv2.RANSAC, 0.999, 1
            )
            _, R_21, T_2_12, _ = cv2.recoverPose(E, matched_pts_1, matched_pts_2, self.K)
            
            trans_mat_21 = self._build_trans_mat(R_21, T_2_12)

            cloud_1 = cv2.triangulatePoints(
                self.K@self.identity_proj_mat, self.K@trans_mat_21[:3], 
                matched_pts_1.transpose(), matched_pts_2.transpose()
            )
            cloud_1 /= cloud_1[3]

            # scale the trajectory
            scale_pts_idx, last_scale_pts_idx, cur_scale_pts_idx\
                 = np.intersect1d(self._last_matched_pts_idx, matched_pts_idx_1, return_indices=True)
            last_3d_pts_W = self._last_cloud_W[:, last_scale_pts_idx]
            trans_mat_W1 = np.linalg.inv(self._trans_mat_1W)
            cloud_W = trans_mat_W1 @ cloud_1
            cur_3d_pts_W = cloud_W[:, cur_scale_pts_idx]
            scale = self._calc_relative_scale(
                last_3d_pts_W, 
                cur_3d_pts_W # from current frame to world frame
            )
            # print(scale)

            cloud_1[:3] = scale * cloud_1[:3]
            cloud_W = trans_mat_W1 @ cloud_1

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(
            #     self._last_cloud_W[0], 
            #     self._last_cloud_W[2], 
            #     self._last_cloud_W[1],
            #     c='green'
            # )
            # ax.scatter3D(
            #     cloud_W[0],
            #     cloud_W[1],
            #     cloud_W[2],
            # )
            # plt.savefig('point_cloud.jpg')
            # pdb.set_trace()

            # update information
            self._last_frame = img
            self._R_1W = R_21 @ self._R_1W
            # TODO: add a scale, but now it is not precise
            self._T_1_W1 = T_2_12 + R_21@self._T_1_W1
            self._R_matrices.append(self._R_1W)
            self._T_vectors.append(self._T_1_W1)
            self._trans_mat_1W = trans_mat_21 @ self._trans_mat_1W
            self._traj.append(-self._R_1W.transpose() @ self._T_1_W1)

            self._last_cloud_W = cloud_W
            self._point_cloud_all = np.concatenate(
                [self._point_cloud_all, self._last_cloud_W], axis=1
            )

            self._last_key_pts = key_pts
            self._last_matched_pts = matched_pts_2
            self._last_matched_pts_idx = matched_pts_idx_2
            self._last_des = des
        elif method=='PnP':
            # TODO: implement and try PnP method
            cv2.solvePnPRansac()
            # raise NotImplementedError
        elif method=='KLT':
            # TODO: implement and try KLT tracking method
            last_matched_pts, matched_pts, dist_mean = self._KLT_featureTracking(self._last_frame, img, self._last_key_pts)

            # extract new key points
            key_pts, des = self._extract_features(img, Convert=False)
            # merge old key_pts with new key_pts


class MONO_VO_PnP(MONO_VO):
    def __init__(self, K: np.ndarray, f_extract_type: str = "SIFT", show_traj: bool = False) -> None:
        super().__init__(K, f_extract_type=f_extract_type, show_traj=show_traj)
        
        # cv2.cornerHarris()

    def _find_new_pts(
        self, 
        old_key_pts:np.ndarray, 
        new_key_pts:np.ndarray, 
        img:np.ndarray, 
        nms_rad:int = 4
    ):
        exist_pts_map = np.zeros_like(img)
        old_key_pts = old_key_pts.astype(int)
        exist_pts_map[old_key_pts[1], old_key_pts[0]] = 1
        
        key_pts_list = []
        indice_list = []
        for i in range(new_key_pts.shape[1]):
            x = int(new_key_pts[0, i])
            y = int(new_key_pts[1, i])

            patch = exist_pts_map[
                y-nms_rad:y+nms_rad, 
                x-nms_rad:x+nms_rad
            ]

            if patch.any():
                continue
            else:
                key_pts_list.append(new_key_pts[:, i])
                indice_list.append(i)
        return np.array(key_pts_list).T, np.array(indice_list, dtype=int)


    def _reproject(
        self, 
        points3D_W:np.ndarray, 
        proj_mat_CW:np.ndarray, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        height, width = img.shape
        key_points = self.K @ proj_mat_CW @ points3D_W
        key_points = (key_points / key_points[2])[:2]

        left_mask = key_points[0] >= 0
        right_mask = key_points[0] <= width-1
        up_mask = key_points[1] >= 0
        bottom_mask = key_points[1] <= height-1

        width_mask = np.logical_and(left_mask, right_mask)
        height_mask = np.logical_and(up_mask, bottom_mask)
        mask = np.logical_and(width_mask, height_mask)

        return key_points[:, mask], mask
        
    def _calc_mean_angle(
        self,
        points3D: np.ndarray, 
        trans_1_WC: np.ndarray, 
        trnas_2_WC: np.ndarray
    ) -> float:
        T_W_1W = trans_1_WC[:3, 3:]
        T_W_2W = trnas_2_WC[:3, 3:]

        T_W_P1 = points3D - T_W_1W
        T_W_P2 = points3D - T_W_2W
        
        inner_product = np.sum(T_W_P1*T_W_P2, axis=0)
        T_W_P1_norm = np.linalg.norm(T_W_P1, axis=0)
        T_W_P2_norm = np.linalg.norm(T_W_P2, axis=0)
        cos_value = inner_product / (T_W_P1_norm * T_W_P2_norm)
        angle_degree = np.arccos(cos_value)*180/np.pi

        return angle_degree.mean()

    def update(self, img: np.ndarray, method: str = "MATCHING", rgb_img=None):
        super().update(img, method=method)
        print('--'*20)
        
        logging.info("{} points to be matched".format(self._last_matched_pts.shape))
        last_matched_pts, matched_pts, matched_num, match_mask\
            = self._KLT_featureTracking(
                self._last_frame, img, 
                self._last_matched_pts
            )
        logging.info("matched {} points with KLT".format(matched_num))
        
        # too few points
        if match_mask is None:
            logging.warning("too few matched points, skip this frame")
            return 1
        
        # calculate current pose
        cloud_W_matched = self._last_cloud_W[:, match_mask]
        
        _, r_2W, T_2_W2, inliers_idx = cv2.solvePnPRansac(
            cloud_W_matched.T, matched_pts, self.K, None,
            reprojectionError=8,
            flags=cv2.cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=2000
        )
        # if inliers_idx is None:
        #     pdb.set_trace()
        inliers_idx = inliers_idx[:, 0]
        # pdb.set_trace()
        self._img_to_show = self.draw_match(
            img, last_matched_pts[inliers_idx], 
            matched_pts[inliers_idx], rgb_img
        )
        R_2W, _ = cv2.Rodrigues(r_2W)
        last_trans_mat_CW = self._build_trans_mat(self._R_1W, self._T_1_W1)
        R_W2 = R_2W.T
        T_W_2W = -R_W2 @ T_2_W2
        trans_mat_CW = self._build_trans_mat(R_2W, T_2_W2)

        # skip or not
        mean_angle  = self._calc_mean_angle(
            cloud_W_matched[:, inliers_idx], 
            np.linalg.inv(last_trans_mat_CW), 
            np.linalg.inv(trans_mat_CW)
        )
        print(mean_angle, inliers_idx.size)
        if mean_angle<1.5 or inliers_idx.size<30:
            return 1
        if np.linalg.norm(T_2_W2-self._T_1_W1)>10:
            print(T_2_W2)
            pdb.set_trace()
            return 2
        
        # reproject 3D points to current image
        reproject_key_pts, reproject_mask = self._reproject(
            points3D_W = self._build_homo_points(self._last_cloud_W), 
            proj_mat_CW = trans_mat_CW[:3], 
            img = img
        )
        last_cloud_W = self._last_cloud_W[:, reproject_mask]
        
        # find candidate
        last_pts = cv2.KeyPoint_convert(self._last_key_pts)
        last_candidate_pts, candidate_pts, _, candidate_mask\
            = self._KLT_featureTracking(
                self._last_frame, img, 
                last_pts
            )
        last_candidate_pts = last_candidate_pts.T
        candidate_pts = candidate_pts.T

        # triangulate new candidate 3D points
        
        logging.info("input triangulate pts number: {}".format(candidate_pts.shape[1]))
        candidate_cloud_W, candidate_cloud_idx = self._triangulate(
            last_trans_mat_CW[:3], trans_mat_CW[:3], 
            last_candidate_pts, 
            candidate_pts
        )
        last_candidate_pts = last_candidate_pts[:, candidate_cloud_idx]
        candidate_pts = candidate_pts[:, candidate_cloud_idx]

        # find candidate points different from old ones
        new_key_pts, new_key_pts_idx = self._find_new_pts(
            reproject_key_pts, 
            candidate_pts, 
            img
        )
        candidate_cloud_W = candidate_cloud_W[:, new_key_pts_idx]

        logging.info("remaining pts number: {}".format(reproject_key_pts.shape[1]))
        logging.info("new landmarks number: {}".format(new_key_pts_idx.size))
    
        # if triagulate_indice.size < TRIANGULATE_PTS_LOWER_BOUND:
        #     cloud_pts_num = triagulate_indice.size
        #     logging.warning("Too few 3D points: {}".format(cloud_pts_num))
        #     if cloud_pts_num==0:
        #         self.draw_match(
        #             img, last_candidate_pts, candidate_pts, rgb_img=rgb_img
        #         )
        #         # pdb.set_trace()
        #     return 1

        # merge new points with old points
        # pdb.set_trace()
        cloud_W = np.concatenate(
            [last_cloud_W, candidate_cloud_W], 
            axis=1
        )
        self.draw_cloud(cloud_W)
        
        matched_key_pts = np.concatenate(
            [reproject_key_pts, new_key_pts], 
            axis=1
        )
        self.draw_match(
            img, matched_key_pts.T, matched_key_pts.T, rgb_img
        )

        # record data
        self._last_frame = img
        self._R_1W = R_2W
        self._T_1_W1 = T_2_W2
        # self._R_matrices.append(R_2W)
        # self._T_vectors.append(T_2_W2)

        self._trans_mat_1W = self._build_trans_mat(R_2W, T_2_W2)
        self._traj.append(T_W_2W)
        print(self._traj[-1]-self._traj[-2])

        self._last_cloud_W = cloud_W
        # self._point_cloud_all = np.concatenate(
        #         [self._point_cloud_all, self._last_cloud_W], axis=1
        #     )
        logging.info("landmark number: {}".format(matched_key_pts.shape[1]))
        self._last_matched_pts = matched_key_pts.astype(np.float32).T

        # all_key_pts, _ = self._extract_features(img, Convert=False)
        # logging.info("new key points number: {}".format(len(all_key_pts)))
        # self._last_key_pts_sift = all_key_pts

        all_key_pts = self._harris_detector.distribute_keypoints(img)
        logging.info("new key points number: {}".format(all_key_pts.shape[0]))
        self._last_key_pts = cv2.KeyPoint_convert(all_key_pts)        

        return True
    
