import os
import h5py
import argparse
import numpy as np
from structures import BehaviourGraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, help="Path to dataset",
        default="/data/home/joel/datasets/")
    args = parser.parse_args()

    graph = BehaviourGraph()
    graph.loadGraph(os.path.join(args.directory, 'behaviour_graph.json'))

    files = [file for file in os.listdir(args.directory) if file.endswith('h5')]
    for file in files:
        with h5py.File(os.path.join(args.directory, file)) as hf:
            print(">>> ", file)
            segment_count = int(np.array(hf['segment_count']))
            for i in range(segment_count):
                edge_idx = int(np.array(hf[str(i) + '/edge_idx']))
                if edge_idx == -1:
                    print("Pre-initialisation")
                else:
                    src_idx, dst_idx, _, _ = graph.graph_edges[edge_idx]
                    print(src_idx, ' -> ', dst_idx)