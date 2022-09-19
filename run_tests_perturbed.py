from torch.utils.data import DataLoader

import os
import sys
import json
import argparse
import numpy as np
from functools import reduce

from models import *
from datasets import GraphTestDataset, collate_graph_data
from structures import BehaviourGraph


def diff_graphs(original: BehaviourGraph, perturbed: BehaviourGraph):
    original_edge_set = set(original.graph_edges)
    perturbed_edge_set = set(perturbed.graph_edges)
    removed_edges = list(original_edge_set - perturbed_edge_set)
    added_edges = list(perturbed_edge_set - original_edge_set)

    # TODO: Add support for adding/removing nodes

    removed_edges_str = reduce(lambda x, y: x + "\n" + y, map(str, removed_edges), "")
    added_edges_str = reduce(lambda x, y: x + "\n" + y, map(str, added_edges), "")
    return removed_edges_str, added_edges_str

def parse_predictions(predictions):
    accum_targets = []
    accum_edge_output = []
    for batch in predictions:
        targets, graph_edge_conns, edge_prob_list, edge_count_summed_list = batch
        targets = targets.numpy()
        edge_count_summed_list = edge_count_summed_list.numpy()
        uppers = edge_count_summed_list[1:]
        lowers = edge_count_summed_list[:-1]
        edge_data = torch.cat((graph_edge_conns, edge_prob_list), dim=1).numpy()

        for target, lower, upper in zip(targets, lowers, uppers):
            subgraph_edge_data = edge_data[lower:upper, :]
            accum_targets.append(target.tolist())
            accum_edge_output.append(subgraph_edge_data.tolist())

    file_data_map = {
        'targets': accum_targets,
        'subgraph_edge_output': accum_edge_output
        }

    return file_data_map

def test_model(args):
    model = GLN.load_from_checkpoint(args.model_dir)
    tester = pl.Trainer(
        accelerator='gpu',
        gpus=1
    )

    data_map = {}
    paths = os.listdir(args.test_dir)
    for path in paths:
        traj_file = [f for f in os.listdir(os.path.join(args.test_dir, path)) if f.endswith('.h5')]
        assert(len(traj_file) == 1)
        traj_file = traj_file[0]

        original_graph = BehaviourGraph()
        original_graph.loadGraph(os.path.join(args.test_dir, path, 'behaviour_graph.json'))
        directory_data_map = {}

        for graph_file in [f for f in os.listdir(os.path.join(args.test_dir, path)) if f.endswith('.json')]:
            traj_file = os.path.join(args.test_dir, path, traj_file)
            graph_file = os.path.join(args.test_dir, path, graph_file)
            test_data = GraphTestDataset(traj_file, graph_file, args.depth_stack_size)
            test_loader = DataLoader(
                test_data, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=1, 
                collate_fn=collate_graph_data
                )
            
            predictions = tester.predict(model, test_loader)
            file_data_map = parse_predictions(predictions)
            removed_edges_str, added_edges_str = diff_graphs(original_graph, test_data.graph)
            perturb_description = ">>> Removed edges\n" + removed_edges_str + \
                "\n>>> Added edges\n" + added_edges_str
            file_data_map['description'] = perturb_description
            directory_data_map[graph_file] = file_data_map

        data_map[path] = directory_data_map
    
    with open('results.json', 'w') as f:
        json.dump(data_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, help="Path to dataset",
        default="/data/home/joel/datasets/blocal_testing/blocal_test_area6_only")
    parser.add_argument("--depth_stack_size", type=int, help="Size of depth stack",
        default=20)
    parser.add_argument("--model_dir", type=str, help="Path to model to be tested",
        default="/data/home/joel/datasets/models/gln/e54_220831_gln.ckpt")
    parser.add_argument("--batch_size", type=str, help="Batch size for inference",
        default=8)
    args = parser.parse_args()

    test_model(args)