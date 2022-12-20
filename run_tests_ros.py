from ast import parse
from torch.utils.data import DataLoader

import os
import sys
import json
import argparse
import numpy as np

from models import *
from datasets import RosGraphTestDataset, collate_graph_data

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

    test_data = RosGraphTestDataset(args.test_dir)
    test_loader = DataLoader(
            test_data, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=1, 
            collate_fn=collate_graph_data
        )

    predictions = tester.predict(model, test_loader)
    parsed = parse_predictions(predictions)
    with open('results.json', 'w') as f:
        json.dump(parsed, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, help="Path to dataset",
        #default="/data/home/joel/datasets/blocal_data/blocal_h5_smallsize_val")
        default="/data/home/joel/datasets/blocal_data/com1_basement_test")
    parser.add_argument("--depth_stack_size", type=int, help="Size of depth stack",
        default=10)
    parser.add_argument("--model_dir", type=str, help="Path to model to be tested",
        default="/data/home/joel/datasets/models/gln_rw/e24_nonoise.ckpt")
    parser.add_argument("--batch_size", type=str, help="Batch size for inference",
        default=8)
    args = parser.parse_args()

    test_model(args)
