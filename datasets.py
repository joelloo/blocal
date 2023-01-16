import torch
from torch.utils.data import Dataset, DataLoader

import os
import sys
import h5py
import numpy as np
from tqdm import tqdm

from structures import Intention, BehaviourGraph


def collate_graph_data(batch):
    # Prepare the metadata indices. This consists of the ground truth
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

def crop_graph(graph, edge_idx, add_noise, radius, get_orig_graph=False, noise_radius=1, rng=None):
    edge_src, edge_dst, _, _ = graph.graph_edges[edge_idx]
    centre_node = edge_src
    # centre_node = edge_dst

    # Sample a new centre node using the noise_radius parameter
    if add_noise:
        if rng is None:
            raise Exception("No RNG provided")
        out_edges = graph.vertices[centre_node]['out_edges']
        sampled_node_idx = rng.integers(len(out_edges) + 1) # Include the current centre_node in the draw
        if sampled_node_idx < len(out_edges):
            centre_node, _, _ = out_edges[sampled_node_idx]


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
        import h5py
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
        depth_stack = depth_stack.view(-1, depth_stack.shape[2], depth_stack.shape[3])


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
        import h5py

        self.depth_data = [[] for _ in self.dataset_keys]
        for dataset_idx, (dataset, file_keys) in enumerate(zip(self.dataset_keys, self.file_keys)):
            print("Loading ", dataset_idx, ": ", dataset)
            for filename in tqdm(file_keys):
                pathname = os.path.join(data_dir, dataset, filename)
                with h5py.File(pathname, 'r') as f:
                    depth_ims = torch.from_numpy(np.array(f['combined_depth']))
                    self.depth_data[dataset_idx].append(depth_ims)

        print("Dataset loaded! Size: ", len(self.data_idxs))

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, index):
        dataset_idx, file_idx, _, _, im_idx, edge_idx = self.data_idxs[index]
        # depth_stack = self.depth_data[dataset_idx][file_idx][im_idx:im_idx+self.depth_stack_size, :, :]
        depth_stack = self.depth_data[dataset_idx][file_idx][im_idx:im_idx+self.depth_stack_size, :, :, :]
        depth_stack = depth_stack.view(-1, depth_stack.shape[2], depth_stack.shape[3])
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

    def cropGraph(self, dataset_idx, edge_idx, add_noise=False, radius=3):
        graph = self.graphs[dataset_idx]
        subgraph = crop_graph(graph, edge_idx, add_noise, radius)
        return subgraph

class RosGraphDataset(Dataset):
    def __init__(self, data_dir, add_noise=False):
        self.add_noise = add_noise
        self.rng = np.random.default_rng() if add_noise else None

        # Load sampled and formatted data
        print("Loading data keys")
        tmp = np.load(os.path.join(data_dir, 'ros_formatted.npz'))
        self.area_keys = tmp['area_keys']
        self.file_keys = tmp['file_keys']
        self.data_img_idxs = tmp['data_img_idxs']
        self.data_edge_int_labels = tmp['data_edge_int_labels']
        self.data_changepoint_labels = tmp['data_changepoint_labels']
        self.data_edge_idxs = tmp['data_edge_idxs']
        self.data_file_descs = tmp['data_file_descs']
        self.depth_stack_size = tmp['depth_stack_size']

        # Load the behaviour graphs
        print("Loading behaviour graphs")
        self.graphs = [BehaviourGraph() for _ in range(len(self.area_keys))]
        for dataset_name, graph in zip(self.area_keys, self.graphs):
            graph.loadGraph(os.path.join(data_dir, dataset_name, 'behaviour_graph.json'))
            graph.initialise()

        # Load the depth image data and edge indices
        print("Loading depth image data and edge indices")
        import h5py

        self.depth_data = [[] for _ in self.area_keys]
        self.depth_timestamps = [[] for _ in self.area_keys]
        for area_idx, (area, file_keys) in enumerate(zip(self.area_keys, self.file_keys)):
            print("Loading ", area_idx, ": ", area)
            for filename in tqdm(file_keys):
                pathname = os.path.join(data_dir, area, filename)
                with h5py.File(pathname, 'r') as f:
                    np_depth_ims = np.array(f['combined_depth']).astype(np.float32) * 1e-3
                    depth_ims = torch.from_numpy(np_depth_ims)
                    self.depth_data[area_idx].append(depth_ims)
                    self.depth_timestamps[area_idx].append(np.array(f['depth_timestamps']))

        print("Dataset loaded! Size: ", len(self.data_edge_int_labels))

    def __len__(self):
        return len(self.data_edge_int_labels)

    def __getitem__(self, index):
        area_idx, file_idx = self.data_file_descs[index]
        img_idxs = self.data_img_idxs[index].tolist()[::-1]
        edge_idx = self.data_edge_idxs[index]
        changepoint_label = self.data_changepoint_labels[index]

        # print(area_idx, file_idx, edge_idx)

        depth_stack = self.depth_data[area_idx][file_idx][img_idxs]
        depth_stack = depth_stack.view(-1, depth_stack.shape[2], depth_stack.shape[3])
        subgraph_node_cats, subgraph_edge_cats, subgraph_edge_conns, graph_edge_conns, gt_edge_idx = self.cropGraph(area_idx, edge_idx, self.add_noise)

        # print(">>>>")
        # np.set_printoptions(formatter={'all':lambda x: str(x)})
        # print(self.depth_timestamps[area_idx][file_idx][img_idxs])
        # print("<<<<")

        # print(gt_edge_idx)
        # print(subgraph_edge_conns)
        # print(graph_edge_conns)

        return (
            # gt_edge_idx,
            # torch.LongTensor([area_idx, gt_edge_idx]),
            torch.LongTensor([changepoint_label, gt_edge_idx]),
            depth_stack,
            subgraph_node_cats,
            subgraph_edge_cats,
            subgraph_edge_conns,
            graph_edge_conns
        )

    def cropGraph(self, area_idx, edge_idx, add_noise=True, radius=3):
        graph = self.graphs[area_idx]
        subgraph = crop_graph(graph, edge_idx, add_noise, radius, rng=self.rng)
        return subgraph

class RosGraphTestDataset(RosGraphDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        area_idx, file_idx = self.data_file_descs[index]
        img_idxs = self.data_img_idxs[index].tolist()[::-1]
        edge_idx = self.data_edge_idxs[index]
        changepoint_label = self.data_changepoint_labels[index]

        depth_stack = self.depth_data[area_idx][file_idx][img_idxs]
        depth_stack = depth_stack.view(-1, depth_stack.shape[2], depth_stack.shape[3])
        subgraph_node_cats, subgraph_edge_cats, subgraph_edge_conns, graph_edge_conns, gt_edge_idx = self.cropGraph(area_idx, edge_idx)

        gt_datum_idx = torch.LongTensor([changepoint_label, edge_idx, gt_edge_idx, index])

        return (
            gt_datum_idx,
            depth_stack,
            subgraph_node_cats,
            subgraph_edge_cats,
            subgraph_edge_conns,
            graph_edge_conns
        )

    def cropGraph(self, area_idx, edge_idx, add_noise=False, radius=3):
        graph = self.graphs[area_idx]
        subgraph = crop_graph(graph, edge_idx, add_noise, radius, get_orig_graph=True)
        return subgraph


# if __name__ == "__main__":
#     # data_dir = "/Users/joel/Research/data/test"
#     data_dir = "/Users/joel/Research/data/ros_test/test_h5_v2"
#     batch_size = 1
#     n_train_workers = 1

#     if sys.platform == "darwin":
#         import matplotlib
#         matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt

#     matplotlib.rc('image', cmap='jet')

#     # train_data = GraphDataset(data_dir)
#     # train_data = GraphTestDataset("/Users/joel/Research/data/test/area1/t0.h5", 
#     #     "/Users/joel/Research/data/test/area1/behaviour_graph.json")
#     train_data = RosGraphDataset(data_dir)
#     train_loader = DataLoader(
#         dataset=train_data, batch_size=batch_size, shuffle=True, 
#         num_workers=n_train_workers, collate_fn=collate_graph_data
#     )

#     print('Calling iter')
#     it = iter(train_loader)
#     print('Calling next')

#     gt_edge_idxs, depth_stacks, (node_cats, edge_cats, edge_conns, node_idxs, edge_idxs), graph_edge_conns, summed = next(it)
# #     print(">>>>> gt_edge_idxs")
# #     print(gt_edge_idxs)
# #     print(">>>>> node_cats")
# #     print(node_cats)
# #     # print(">>>>> edge_cats")
# #     # print(edge_cats)
# #     # print(">>>>> edge_conns")
# #     # print(edge_conns)
# #     print(">>>>> node_idxs")
# #     print(node_idxs)
# #     # print(">>>>> edge_idxs")
# #     # print(edge_idxs)
# #     print(">>>>> graph_edges")
# #     print(torch.cat((edge_conns, graph_edge_conns), dim=1))
# #     print(">>>>> batch_summed")
# #     print(summed)

# #     print("========")
# #     print(">>>>> ground truth edges")
# #     print(graph_edge_conns[gt_edge_idxs[:, 1], :])
# #     print(gt_edge_idxs[:, 0], train_data.dataset_keys)

# #     print('Done')

#     print(depth_stacks.shape)
#     print(gt_edge_idxs[0])
#     print(train_data.graphs[0].graph_edges[gt_edge_idxs[0]])
#     _, num_imgs, _, _ = depth_stacks.shape
#     num_steps = int(num_imgs / 3)
    
#     for idx in range(num_steps):
#         im_left = (depth_stacks[0, idx*3, :, :].numpy() * 1e3).astype(np.uint16)
#         im_mid = (depth_stacks[0, idx*3+1, :, :].numpy() * 1e3).astype(np.uint16)
#         im_right = (depth_stacks[0, idx*3+2, :, :].numpy() * 1e3).astype(np.uint16)

#         fig, axs = plt.subplots(1, 3)
#         fig.suptitle(str(idx))
#         axs[0].imshow(im_left)
#         axs[1].imshow(im_mid)
#         axs[2].imshow(im_right)

#         plt.show()
# # #     print(node_idxs)
# # #     # print(">>>>> edge_idxs")
# # #     # print(edge_idxs)
# # #     print(">>>>> graph_edges")
# # #     print(torch.cat((edge_conns, graph_edge_conns), dim=1))
# # #     print(">>>>> batch_summed")
# # #     print(summed)

# # #     print("========")
# # #     print(">>>>> ground truth edges")
# # #     print(graph_edge_conns[gt_edge_idxs[:, 1], :])
# # #     print(gt_edge_idxs[:, 0], train_data.dataset_keys)

# # #     print('Done')

# #     print(depth_stacks.shape)
# #     _, num_imgs, _, _ = depth_stacks.shape
# #     num_steps = int(num_imgs / 3)
    
# #     for idx in range(num_steps):
# #         im_left = depth_stacks[0, idx*3, :, :]
# #         im_mid = depth_stacks[0, idx*3+1, :, :]
# #         im_right = depth_stacks[0, idx*3+2, :, :]

# #         fig, axs = plt.subplots(1, 3)
# #         fig.suptitle(str(idx))
# #         axs[0].imshow(im_left)
# #         axs[1].imshow(im_mid)
# #         axs[2].imshow(im_right)

# #         plt.show()
# # #     print(node_idxs)
# # #     # print(">>>>> edge_idxs")
# # #     # print(edge_idxs)
# # #     print(">>>>> graph_edges")
# # #     print(torch.cat((edge_conns, graph_edge_conns), dim=1))
# # #     print(">>>>> batch_summed")
# # #     print(summed)

# # #     print("========")
# # #     print(">>>>> ground truth edges")
# # #     print(graph_edge_conns[gt_edge_idxs[:, 1], :])
# # #     print(gt_edge_idxs[:, 0], train_data.dataset_keys)

# # #     print('Done')

# #     print(depth_stacks.shape)
# #     _, num_imgs, _, _ = depth_stacks.shape
# #     num_steps = int(num_imgs / 3)
    
# #     for idx in range(num_steps):
# #         im_left = depth_stacks[0, idx*3, :, :]
# #         im_mid = depth_stacks[0, idx*3+1, :, :]
# #         im_right = depth_stacks[0, idx*3+2, :, :]

# #         fig, axs = plt.subplots(1, 3)
# #         fig.suptitle(str(idx))
# #         axs[0].imshow(im_left)
# #         axs[1].imshow(im_mid)
# #         axs[2].imshow(im_right)

# #         plt.show()
# # #     print(node_idxs)
# # #     # print(">>>>> edge_idxs")
# # #     # print(edge_idxs)
# # #     print(">>>>> graph_edges")
# # #     print(torch.cat((edge_conns, graph_edge_conns), dim=1))
# # #     print(">>>>> batch_summed")
# # #     print(summed)

# # #     print("========")
# # #     print(">>>>> ground truth edges")
# # #     print(graph_edge_conns[gt_edge_idxs[:, 1], :])
# # #     print(gt_edge_idxs[:, 0], train_data.dataset_keys)

# # #     print('Done')

# #     print(depth_stacks.shape)
# #     _, num_imgs, _, _ = depth_stacks.shape
# #     num_steps = int(num_imgs / 3)
    
# #     for idx in range(num_steps):
# #         im_left = depth_stacks[0, idx*3, :, :]
# #         im_mid = depth_stacks[0, idx*3+1, :, :]
# #         im_right = depth_stacks[0, idx*3+2, :, :]

# #         fig, axs = plt.subplots(1, 3)
# #         fig.suptitle(str(idx))
# #         axs[0].imshow(im_left)
# #         axs[1].imshow(im_mid)
# #         axs[2].imshow(im_right)

# #         plt.show()