import os
import sys
import h5py
import numpy as np

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from structures import Intention, BehaviourGraph


if __name__ == "__main__":
    # file_path = "/Users/joel/Research/data/area1/test/area1/t1.h5"
    # metadata_path = "/Users/joel/Research/data/area1"
    file_path = "/Users/joel/Research/data/area6/test_v2/t0.h5"
    metadata_path = "/Users/joel/Research/data/area6"
    show_graph = True

    if show_graph:
        graph_path = os.path.join(metadata_path, "behaviour_graph.json")
        map_path = os.path.join(metadata_path, "map.npz")

        tmp = np.load(map_path)
        map_im = tmp['map']
        plt.imshow(map_im, origin='lower')

        graph = BehaviourGraph()
        graph.loadGraph(graph_path)

        graph_nodes_pix = list(map(lambda node: node[0], graph.graph_nodes))
        if len(graph_nodes_pix) > 0:
            colours = [("#fa8072" if node[2] else "g") for node in graph.graph_nodes]
            graph_nodes_pix = np.array(graph_nodes_pix)
            plt.scatter(graph_nodes_pix[:,0], graph_nodes_pix[:,1], c=colours, s=25)

            for idx in range(len(graph.graph_nodes)):
                plt.annotate(idx, (graph_nodes_pix[idx, 0], graph_nodes_pix[idx, 1]), color="green")

        with h5py.File(file_path, 'r') as f:
            segment_count = int(np.array(f['segment_count']))
            for segment_idx in range(segment_count):
                edge_idx = int(np.array(f[str(segment_idx) + '/edge_idx']))
                src, dst, _, _ = graph.graph_edges[edge_idx]
                src_pt, _, _ = graph.graph_nodes[src]
                dst_pt, _, _ = graph.graph_nodes[dst]
                c = 'r' if segment_idx == 0 else (
                    'aqua' if segment_idx == segment_count-1 else ('k' if segment_idx % 2 == 1 else 'peru')
                )
                plt.arrow(*src_pt, *(dst_pt - src_pt), color=c)

            plt.show()

    
    with h5py.File(file_path, 'r') as f:
        prev_len = 0
        segment_count = int(np.array(f['segment_count']))
        segment_cumlen = f['segment_cumlen']
        for idx, clen in enumerate(segment_cumlen):
            segment = f[str(idx)]
            print("Segment ", idx, ": ", Intention(np.array(segment['label'])), ", ", clen - prev_len)
            prev_len = clen

        depth_ims = f['combined_depth']
        print(depth_ims.shape)

        segment_cumlen_padded = np.insert(segment_cumlen, 0, 0)
        for segment_idx in range(segment_count):
            segment_int = Intention(np.array(f[str(segment_idx) + '/label']))
            edge_idx = int(np.array(f[str(segment_idx) + '/edge_idx']))
            print(">>> Currently on edge: ", edge_idx)
            for im_idx in range(segment_cumlen_padded[segment_idx], segment_cumlen_padded[segment_idx+1]):
                # im = depth_ims[im_idx, :, :]
                # plt.imshow(im)
                # plt.title(str(im_idx) + ": " + str(segment_int))

                fig, axs = plt.subplots(1, 3)
                ims = depth_ims[im_idx, :, :, :]
                print(ims[0])
                axs[0].imshow(ims[0])
                axs[1].imshow(ims[1])
                axs[2].imshow(ims[2])
                fig.suptitle(str(im_idx) + ": " + str(segment_int))
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