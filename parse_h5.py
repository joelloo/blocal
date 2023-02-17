import os
import sys
import h5py
import numpy as np

from structures import BehaviourGraph

if __name__ == "__main__":
    folder_path = '/data/home/joel/datasets/blocal_data/blocal_odom_h5_train/COM1_L1'
    node_idx = 7

    graph = BehaviourGraph()
    graph.loadGraph(os.path.join(folder_path, 'behaviour_graph.json'))

    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            path = os.path.join(folder_path, file)
            # print(file)

            with h5py.File(path) as hf:
                for idx in np.array(hf['depth_edge_idxs']):
                    src, dst, _, _ = graph.graph_edges[idx]
                    if src == node_idx or dst == node_idx:
                        print("FOUND! ", file)
                        break
                