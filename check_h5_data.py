import os
import sys
import h5py
import numpy as  np

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # h5_file = "/data/home/joel/datasets/blocal_data/blocal_odom_h5/COM1_L1/2022-12-30-22-08-54.h5"
    h5_file = "/data/home/joel/datasets/blocal_data/blocal_odom_h5_small/COM1_L1/2022-12-30-19-13-14.h5"
    with h5py.File(h5_file) as hf:
        print(hf.keys())
        print(">>>>>>>>>>>>")

        segment_count = int(np.array(hf['segment_count']))
        print(segment_count)

        depth_imgs = np.array(hf['combined_depth'])
        print(depth_imgs.shape)

        junction_poses = np.array(hf['junction_poses'])
        depth_segment_idxs = np.array(hf['depth_segment_idxs'])
        depth_positions = np.array(hf['depth_positions'])
        print(depth_positions.shape)
        depth_edge_int_labels = np.array(hf['depth_edge_int_labels'])
        print(depth_edge_int_labels.shape)
        changepoint_poses = np.array(hf['changepoint_poses'])
        print(changepoint_poses.shape)
        print(">>>>>>>>>")
        
        # np.set_printoptions(threshold=sys.maxsize)
        # print(depth_edge_int_labels)

        def assign_colour(label):
            if label == 0:
                return 'b'
            elif label == 1:
                return 'k'
            elif label == 2:
                return 'r'
            else:
                return 'y'

        pose_colours = [assign_colour(label) for label in depth_edge_int_labels]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(depth_positions[:, 0], depth_positions[:, 1], c=pose_colours)
        ax.scatter(changepoint_poses[:, 0], changepoint_poses[:, 1], marker='x')
        ax.scatter(junction_poses[:, 0], junction_poses[:, 1], c='c', marker='*')
        ax.scatter(depth_positions[0, 0], depth_positions[0, 1], c='r', marker='o')
        ax.scatter(depth_positions[-1, 0], depth_positions[-1, 1], c='g', marker='o')
        ax.set_aspect('equal')

        plt.savefig('h5.png')

        fig, axs = plt.subplots(1, 3)
        for idx, (ims, seg) in enumerate(zip(depth_imgs, depth_segment_idxs)):
            if idx % 10 == 0:
                for ax in axs:
                    ax.cla()
                axs[0].imshow(ims[0])
                axs[1].imshow(ims[1])
                axs[2].imshow(ims[2])
                plt.savefig('depth_ims/' + str(seg) + '_' + str(idx) + '.png')


    # h5_file = "/data/home/joel/datasets/blocal_data/blocal_h5_smallsize_cps/COM1_L1/2022-10-01-17-08-21.h5"
    # with h5py.File(h5_file) as hf:
    #     print(hf.keys())
    #     print(">>>>>>>>>>>>")

    #     np.set_printoptions(threshold=sys.maxsize)

    #     depth_edge_int_labels = np.array(hf['depth_edge_int_labels'])
    #     print(depth_edge_int_labels)

    #     depth_changepoint_labels = np.array(hf['depth_changepoint_labels'])
    #     print(depth_changepoint_labels)
    #     print("<<<<<<<<<<<<")

    #     depth_segment_idxs = np.array(hf['depth_segment_idxs'])
    #     print(depth_segment_idxs)