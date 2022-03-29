import cv2
import logging
import matplotlib.pyplot as plt
import pdb

from utils.Descriptors import SIFTDescriptor
from utils.Detectors import HarrisDetector, SIFTDetector, ShiTomasiDetector
from utils.features import SIFT
from utils.utils import get_args
from dataset import datasets, KITTI_dataset
from VO_all_in_one import VisualOdometry


PLOT_TRAJ = True

def main():
    # init VO and dataset
    logging.basicConfig(level=logging.INFO, \
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = get_args()
    # dataset = datasets[args.dataset](img_ptr=0)
    dataset = KITTI_dataset()

    # -----------------------------------------
    # params = {"test":True, "harris_threshold":1e-3}
    # img0, rgb_img, _ = dataset.get_next_image()
    # harris_det = HarrisDetector()

    # detector = SIFT_Detector()
    # kps = harris_det.detect(img0)
    # descriptors = detector.compute(img0, kps)
    # print(kps[0].size, kps[0].angle)
    # cv2.KeyPoint_convert(kps)
    # pdb.set_trace()
    
    # feature_extractor = ShiTomasiDetector(edgeThreshold=3.)

    # kps = detector.detect(img0, x_cell_num=5, y_cell_num=3, cell_keypts_max_num=20)
    # description = descriptor.describe(img0, kps)
    # kps = feature_extractor.detect(img0,x_cell_num=5, y_cell_num=3, cell_keypts_max_num=20)
    # description = feature_extractor.describe(img0, kps)
    # kps, description = feature_extractor.detectAndDescribe(img0,x_cell_num=5, y_cell_num=3, cell_keypts_max_num=20)
    # img_to_show = cv2.drawKeypoints(rgb_img, kps, rgb_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.figure()
    # plt.imshow(img_to_show)
    # plt.show()
    # pdb.set_trace()

    # -----------------------------------------
    # TODO: 0.0 VO framework

    VO = VisualOdometry(dataset.K)
    
    img, img_rgb, img_name = dataset.get_next_image()
    for i in range(5):
        new_img, new_img_rgb, new_img_name = dataset.get_next_image()
    kps_detector = HarrisDetector()
    while not VO.bootstrap_KLT(img_rgb, new_img_rgb, kps_detector):
        img = new_img
        new_img, _, _ = dataset.get_img()

    # VO = VOs.VO_TYPES[args.vo]()
    # VO = MONO_VO_PnP(dataset.K, show_traj=PLOT_TRAJ, f_extract_type="SIFT")
    # if PLOT_TRAJ:
    #     plt.ion()
    #     fig = plt.figure()
    
    # img0, rgb_img, _ = dataset.get_next_image()
    # img1, rgb_img, _ = dataset.get_next_image()
    # while not(VO.initialization(img0, img1, rgb_img=rgb_img)):
    #     img1, rgb_img, _ = dataset.get_next_image()

    # img, rgb_img, _ = dataset.get_next_image()

    # break_point = 0
    # while img is not None:
    #     update_res = VO.update(img, rgb_img)
    #     if update_res==1 and break_point!=0:
    #         # record reintialize start
    #         break_point = dataset.index
    #     if update_res==2:
    #         dataset.set_ptr(break_point)
    #         img0, rgb_img, _ = dataset.get_next_image()
    #         img, rgb_img, _ = dataset.get_next_image()
    #         while not(VO.initialization(img0, img, rgb_img=rgb_img)):
    #             img, rgb_img, _ = dataset.get_next_image()
    #     print(dataset.index)
    #     img, rgb_img, _ = dataset.get_next_image()
        
    #     if PLOT_TRAJ:
    #         fig.clf()
    #         traj_plt = fig.add_subplot(121)
    #         traj = VO.traj
    #         # point_cloud = VO.last_point_cloud
    #         traj_plt.plot(traj[:, 0], traj[:, 2])
    #         # traj_plt.scatter(point_cloud[:, 0], point_cloud[:, 2])
    #         traj_plt.set_xlabel('X')
    #         traj_plt.set_ylabel('Y')
    #         plt.draw()

    #         if VO.img_to_show is not None:
    #             ax = fig.add_subplot(122)
    #             ax.imshow(VO.img_to_show)
            
    #         plt.pause(0.01)


    # while img is not None:
    #     cv2.imshow("img_show_window", img)
    #     cv2.waitKey(20)
    #     img, _,  = dataset.get_next_image()
    # cv2.destroyAllWindows()
    
        # plt.savefig("result.jpg")
    # plt.ioff()
    # plt.show()


if __name__ == "__main__":
    main()