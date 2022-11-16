import os
import cv2
import sys
import h5py
import numpy as np

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from structures import Intention, BehaviourGraph

matplotlib.rc('image', cmap='jet')


if __name__ == "__main__":
    file_path = "/Users/joel/Research/data/ros_test/COM1_L1/2022-10-02-18-01-40.h5"
    # metadata_path = "/Users/joel/Research/data/area6"
    show_graph = True

    # graph_path = '/Users/joel/Downloads/catkin_ws/src/graph_localizer/src/COM1_L1_graph.json'
    # map_path = '/Users/joel/Downloads/catkin_ws/src/graph_localizer/src/COM1_L1.jpeg'

    # if show_graph:
    #     # graph_path = os.path.join(metadata_path, "behaviour_graph.json")
    #     # map_path = os.path.join(metadata_path, "map.npz")

    #     map_im = cv2.imread(map_path)
    #     # plt.imshow(map_im, origin='lower')
    #     plt.imshow(map_im)

    #     graph = BehaviourGraph()
    #     graph.loadGraph(graph_path)

    #     graph_nodes_pix = list(map(lambda node: node[0], graph.graph_nodes))
    #     if len(graph_nodes_pix) > 0:
    #         colours = [("#fa8072" if node[2] else "g") for node in graph.graph_nodes]
    #         graph_nodes_pix = np.array(graph_nodes_pix)
    #         plt.scatter(graph_nodes_pix[:,0], graph_nodes_pix[:,1], c=colours, s=25)

    #         for idx in range(len(graph.graph_nodes)):
    #             plt.annotate(idx, (graph_nodes_pix[idx, 0], graph_nodes_pix[idx, 1]), color="green")

    #     with h5py.File(file_path, 'r') as f:
    #         segment_count = int(np.array(f['segment_count']))
    #         for segment_idx in range(segment_count):
    #             edge_idx = int(np.array(f[str(segment_idx) + '/edge_idx']))
    #             src, dst, _, _ = graph.graph_edges[edge_idx]
    #             src_pt, _, _ = graph.graph_nodes[src]
    #             dst_pt, _, _ = graph.graph_nodes[dst]
    #             c = 'r' if segment_idx == 0 else (
    #                 'aqua' if segment_idx == segment_count-1 else ('k' if segment_idx % 2 == 1 else 'peru')
    #             )
    #             plt.arrow(*src_pt, *(dst_pt - src_pt), color=c)

    #     plt.show()

    
    with h5py.File(file_path, 'r') as f:
        prev_len = 0
        segment_count = int(np.array(f['segment_count']))
        segment_cumlen = f['segment_cumlen']
        print(segment_count)
        print(segment_cumlen)
        for idx, clen in enumerate(segment_cumlen):
            segment = f[str(idx)]
            label = np.array(segment['label'])
            segment_int = None if label < 0 else Intention(label)
            print("Segment ", idx, ": ", segment_int, ", ", clen - prev_len)
            prev_len = clen

        depth_ims = f['combined_depth']
        timestamps = f['depth_timestamps']
        labels = f['depth_labels']
        edge_idxs = f['depth_edge_idxs']
        print(depth_ims.shape)

        for segment_idx in range(segment_count):
            label = np.array(f[str(segment_idx) + '/label'])
            segment_int = None if label < 0 else Intention(label)
            edge_idx = int(np.array(f[str(segment_idx) + '/edge_idx']))
            print(">>> Currently on edge: ", edge_idx, segment_int)

        segment_cumlen_padded = np.insert(segment_cumlen, 0, 0)
        for segment_idx in range(segment_count):
            label = np.array(f[str(segment_idx) + '/label'])
            segment_int = None if label < 0 else Intention(label)
            edge_idx = int(np.array(f[str(segment_idx) + '/edge_idx']))
            print(">>> Currently on edge: ", edge_idx)
            for im_idx in range(segment_cumlen_padded[segment_idx], segment_cumlen_padded[segment_idx+1]):
                if labels[im_idx] == -1:
                    continue

                fig, axs = plt.subplots(1, 3)
                ims = depth_ims[im_idx, :, :, :]
                stamp = timestamps[im_idx]
                label = labels[im_idx]
                edge_idx = edge_idxs[im_idx]

                axs[0].imshow(ims[0])
                axs[1].imshow(ims[1])
                axs[2].imshow(ims[2])
                # fig.suptitle(str(im_idx) + ": " + str(segment_int) + " | " + str(stamp))
                fig.suptitle(str(label) + ', ' + str(edge_idx) + ' | ' + str(stamp))
                plt.show()


        # for idx, clen in enumerate(segment_cumlen):
        #     print(">>> Segment ", idx)
        #     segment = f[str(idx)]
        #     segment_int = Intention(np.array(segment['label']))
        #     depth_ims = segment['depth']
        #     print(depth_ims.shape)
        #     for im in depth_ims:
        #         plt.imshow(im)
        #         plt.title(str(segment_int))
        #         plt.show()