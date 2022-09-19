import torch
from torch.utils.data import Dataset, DataLoader

import os
import sys
import h5py
import numpy as np
from tqdm import tqdm

from structures import Intention, BehaviourGraph

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def collate_graph_data(batch):
    # Prepare the metadata indices. This consists of the ground
    # edge index. For prediction, the metadata also includes
    # the point and segment idx to identify the data point.
    idxs = [gt for gt, _, _, _, _, _ in batch]
    if len(idxs) > 0 and torch.is_tensor(idxs[0]):
        idxs = torch.stack(idxs)
    else:
        idxs = torch.LongTensor(idxs)

    # Prepare the sensor data history (i.e. depth image stack)
    depth_stacks = torch.stack([ds for _, ds, _, _, _, _ in batch], dim=0)

    # Prepare the subgraphs for processing in the GLN. Since the
    # subgraphs have differing sizes, we do not accumulate them in
    # a batch, but instead unroll all the edges and nodes of all
    # the subgraphs in the batch into a flat tensor. We also compute
    # mappings from the nodes/edges in the flattened tensor to the 
    # subgraph that they belong to in the batch.
    node_counts_cumsum = torch.cumsum(torch.LongTensor([len(ncats) for _, _, ncats, _, _, _ in batch]), dim=0)
    node_lower_bound_idx = torch.cat((torch.LongTensor([0]), node_counts_cumsum[:-1]))
    edge_conns_nested = [lower + conns for lower, (_, _, _, _, conns, _) in zip(node_lower_bound_idx, batch)]
    subgraph_edge_connections = torch.cat(edge_conns_nested, dim=0)

    subgraph_node_categories = torch.LongTensor([nc for _, _, ncats, _, _, _ in batch for nc in ncats])
    subgraph_edge_categories = torch.LongTensor([ec for _, _, _, ecats, _, _ in batch for ec in ecats])

    subgraph_node_idxs = torch.LongTensor([i for i, (_, _, ncats, _, _, _) in enumerate(batch) for _ in range(len(ncats))])
    subgraph_edge_idxs = torch.LongTensor([i for i, (_, _, _, ecats, _, _) in enumerate(batch) for _ in range(len(ecats))])

    # Compute the original graph edge connections. Mainly used for debugging
    # or sanity checking the results.
    graph_edge_conns = [gec for _, _, _, _, _, gec in batch]
    if len(graph_edge_conns) > 0 and graph_edge_conns[0] is not None:
        graph_edge_conns = torch.cat(graph_edge_conns, dim=0)
    else:
        graph_edge_conns = None

    # Compute the number of edges in each subgraph in the batch
    # for efficient computation of edge softmax/cross-entropy loss.
    edge_counts_per_sample = torch.LongTensor([len(ecats) for _, _, _, ecats, _, _ in batch])
    batch_edge_count_summed = torch.cat([torch.LongTensor([0]), torch.cumsum(edge_counts_per_sample, dim=0)])

    return (
        idxs,
        depth_stacks,
        (
            subgraph_node_categories,
            subgraph_edge_categories,
            subgraph_edge_connections,
            subgraph_node_idxs,
            subgraph_edge_idxs
        ),
        graph_edge_conns,
        batch_edge_count_summed
    )

def crop_graph(graph, edge_idx, add_noise, radius, get_orig_graph=False):
    edge_src, edge_dst, _, _ = graph.graph_edges[edge_idx]
    centre_node = edge_src
    # centre_node = edge_dst

    # Sample the centre node from among the neighbouring nodes of the ground truth edge
    if add_noise:
        pass

    gt_edge_idx = None
    while gt_edge_idx is None:
        # Find all the nodes radius-hops away from centre_node
        curr_frontier = [centre_node]
        next_frontier = []
        subgraph_nodes = set()
        for _ in range(radius):
            subgraph_nodes.update(curr_frontier)
            for node in curr_frontier:
                outgoing = graph.vertices[node]['out_edges']
                neighbours = [n for n, _, _ in outgoing if n not in subgraph_nodes]
                next_frontier += neighbours

            # Set neighbouring nodes as next frontier, and refresh the next_frontier list
            curr_frontier, next_frontier = next_frontier, curr_frontier
            next_frontier = []
        
        subgraph_nodes.update(curr_frontier)

        # Get all edges connecting the extracted nodes, and verify the ground truth edge is inside
        subgraph_edges = []
        for node in subgraph_nodes:
            outgoing_edges = [
                (node, dst, edge_int) for dst, _, edge_int in graph.vertices[node]['out_edges']
                if dst in subgraph_nodes
            ]

            if node == edge_src:
                for idx, (_, dst, _) in enumerate(outgoing_edges):
                    if dst == edge_dst:
                        gt_edge_idx = len(subgraph_edges) + idx

            subgraph_edges += outgoing_edges

    node_idx_map = {graph_idx:subgraph_idx for subgraph_idx, graph_idx in enumerate(subgraph_nodes)}

    subgraph_edge_conns = torch.LongTensor([
        [node_idx_map[node], node_idx_map[dst]] for node, dst, _ in subgraph_edges
    ])
    graph_edge_conns = torch.LongTensor([[node, dst] for node, dst, _ in subgraph_edges]) if get_orig_graph else None
    edge_categories = torch.LongTensor([edge_int for _, _, edge_int in subgraph_edges])
    node_categories = torch.LongTensor([int(graph.graph_nodes[node][2]) for node in subgraph_nodes])

    subgraph_edge_conns = subgraph_edge_conns.unsqueeze(0) if len(subgraph_edge_conns.shape) < 2 else subgraph_edge_conns
    graph_edge_conns = (
        graph_edge_conns.unsqueeze(0) 
        if graph_edge_conns is not None and len(graph_edge_conns.shape) < 2 
        else graph_edge_conns
    )
    
    return node_categories, edge_categories, subgraph_edge_conns, graph_edge_conns, gt_edge_idx

class GraphTestDataset(Dataset):
    def __init__(self, file, graph_dir, depth_stack_size=20):
        self.depth_data = []
        self.traj_data = []
        self.depth_stack_size = 20

        # Load the data
        print("Loading data: ", file)
        with h5py.File(file, 'r') as f:
            segment_cumlen = f['segment_cumlen']
            segment_upper_bound = segment_cumlen
            segment_lower_bound = np.insert(segment_cumlen, 0, 0)[:-1]
            self.depth_data = torch.from_numpy(np.array(f['combined_depth']))

            for idx, (lower, upper) in enumerate(zip(segment_lower_bound, segment_upper_bound)):
                if upper >= depth_stack_size:
                    start = max(lower, depth_stack_size - 1)

                    graph_edge_idx = int(np.array(f[str(idx) + '/edge_idx']))
                    for pointer in range(start, upper):
                        point_idx = pointer - lower
                        segment_idx = idx
                        depth_end_idx = pointer + 1
                        depth_start_idx = depth_end_idx - self.depth_stack_size
                        self.traj_data.append([point_idx, depth_start_idx, depth_end_idx, segment_idx, graph_edge_idx])

        # Load behaviour graph
        print("Loading behaviour graph")
        self.graph = BehaviourGraph()
        self.graph.loadGraph(graph_dir)
        self.graph.initialise()

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, index):
        point_idx, depth_start_idx, depth_end_idx, segment_idx, graph_edge_idx = self.traj_data[index]
        depth_stack = self.depth_data[depth_start_idx:depth_end_idx]

        (subgraph_node_cats, 
        subgraph_edge_cats, 
        subgraph_edge_conns, 
        graph_edge_conns, 
        gt_edge_idx) = self.cropGraph(graph_edge_idx)

        gt_datum_idx = torch.LongTensor([point_idx, segment_idx, graph_edge_idx, gt_edge_idx])

        return (
            gt_datum_idx,
            depth_stack,
            subgraph_node_cats,
            subgraph_edge_cats,
            subgraph_edge_conns,
            graph_edge_conns
        )

    def cropGraph(self, edge_idx, add_noise=False, radius=3):
        subgraph = crop_graph(self.graph, edge_idx, add_noise, radius, get_orig_graph=True)
        return subgraph


class GraphDataset(Dataset):
    def __init__(self, data_dir):
        # Load the formatted data and keys
        print("Loading data keys")
        tmp = np.load(os.path.join(data_dir, 'formatted.npz'))
        self.dataset_keys = tmp['dataset_keys']
        self.file_keys = tmp['file_keys']
        self.data_idxs = tmp['data']
        self.depth_stack_size = int(tmp['depth_stack_size'])

        # Load the behaviour graphs
        print("Loading behaviour graphs")
        self.graphs = [BehaviourGraph() for _ in range(len(self.dataset_keys))]
        for dataset_name, graph in zip(self.dataset_keys, self.graphs):
            graph.loadGraph(os.path.join(data_dir, dataset_name, 'behaviour_graph.json'))
            graph.initialise()

        # Load the depth image data and edge indices
        print("Loading depth image data and edge indices")
        self.depth_data = [[] for _ in self.dataset_keys]
        for dataset_idx, (dataset, file_keys) in enumerate(zip(self.dataset_keys, self.file_keys)):
            print("Loading ", dataset_idx, ": ", dataset)
            for filename in tqdm(file_keys):
                pathname = os.path.join(data_dir, dataset, filename)
                with h5py.File(pathname, 'r') as f:
                    depth_ims = torch.from_numpy(np.array(f['combined_depth']))
                    self.depth_data[dataset_idx].append(depth_ims)

        print("Dataset loaded!")

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, index):
        dataset_idx, file_idx, _, _, im_idx, edge_idx = self.data_idxs[index]
        depth_stack = self.depth_data[dataset_idx][file_idx][im_idx:im_idx+self.depth_stack_size, :, :]
        subgraph_node_cats, subgraph_edge_cats, subgraph_edge_conns, graph_edge_conns, gt_edge_idx = self.cropGraph(dataset_idx, edge_idx)

        return (
            gt_edge_idx,
            # torch.LongTensor([dataset_idx, gt_edge_idx]),
            depth_stack,
            subgraph_node_cats,
            subgraph_edge_cats,
            subgraph_edge_conns,
            graph_edge_conns
        )

    def cropGraph(self, dataset_idx, edge_idx, add_noise=True, radius=3):
        graph = self.graphs[dataset_idx]
        subgraph = crop_graph(graph, edge_idx, add_noise, radius)
        return subgraph


# if __name__ == "__main__":
#     data_dir = "/Users/joel/Research/data/test"
#     batch_size = 2
#     n_train_workers = 1

#     train_data = GraphDataset(data_dir)
#     # train_data = GraphTestDataset("/Users/joel/Research/data/test/area1/t0.h5", 
#     #     "/Users/joel/Research/data/test/area1/behaviour_graph.json")
#     train_loader = DataLoader(
#         dataset=train_data, batch_size=batch_size, shuffle=True, 
#         num_workers=n_train_workers, collate_fn=collate_graph_data
#     )

#     print('Calling iter')
#     it = iter(train_loader)
#     print('Calling next')

#     gt_edge_idxs, depth_stacks, (node_cats, edge_cats, edge_conns, node_idxs, edge_idxs), graph_edge_conns, summed = next(it)
#     print(">>>>> gt_edge_idxs")
#     print(gt_edge_idxs)
#     print(">>>>> node_cats")
#     print(node_cats)
#     # print(">>>>> edge_cats")
#     # print(edge_cats)
#     # print(">>>>> edge_conns")
#     # print(edge_conns)
#     print(">>>>> node_idxs")
#     print(node_idxs)
#     # print(">>>>> edge_idxs")
#     # print(edge_idxs)
#     print(">>>>> graph_edges")
#     print(torch.cat((edge_conns, graph_edge_conns), dim=1))
#     print(">>>>> batch_summed")
#     print(summed)

#     print("========")
#     print(">>>>> ground truth edges")
#     print(graph_edge_conns[gt_edge_idxs[:, 1], :])
#     print(gt_edge_idxs[:, 0], train_data.dataset_keys)

#     print('Done')

#     for idx, im in enumerate(depth_stacks[0]):
#         plt.imshow(im)
#         plt.title(str(idx))
#         plt.show()
