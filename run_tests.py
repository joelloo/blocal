from ast import parse
from torch.utils.data import DataLoader

import os
import sys
import json
import argparse
import numpy as np

from models import *
from datasets import GraphTestDataset, collate_graph_data

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
    directories = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]
    for directory in directories:
        dir_path = os.path.join(args.test_dir, directory)
        directory_data_map = {}

        for filename in [f for f in os.listdir(dir_path) if f.endswith(".h5")]:
            print(">>> ", directory, filename)

            file_path = os.path.join(dir_path, filename)
            graph_path = os.path.join(dir_path, "behaviour_graph.json")

            test_data = GraphTestDataset(file_path, graph_path, depth_stack_size=args.depth_stack_size)
            test_loader = DataLoader(
                test_data, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=1, 
                collate_fn=collate_graph_data
            )

            predictions = tester.predict(model, test_loader)
            file_data_map = parse_predictions(predictions)
            directory_data_map[filename] = file_data_map

        data_map[directory] = directory_data_map
    
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
