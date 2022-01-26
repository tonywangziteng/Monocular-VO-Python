import cv2
import logging
import matplotlib.pyplot as plt
import pdb


from utils.Detectors import Harris_Detector
from utils.utils import get_args
from dataset import datasets


PLOT_TRAJ = True

def main():
    # init VO and dataset
    logging.basicConfig(level=logging.INFO, \
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = get_args()
    dataset = datasets[args.dataset](img_ptr=0)

    # -----------------------------------------
    detector_args = {
        "block_kp_max_num":20, 
        # "nms_radius":10, 
        # "block_size":7, 
        # "sobel_kernel_size":5
    }
    img0, rgb_img, _ = dataset.get_next_image()
    detector = Harris_Detector()
    kps = detector.detect(img0, x_num=5, y_num=3, harris_threshold=1e-3, **detector_args)
    img_to_show = cv2.drawKeypoints(rgb_img, kps, rgb_img)
    plt.figure()
    plt.imshow(img_to_show)
    plt.show()

    # -----------------------------------------

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