from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

import os
import argparse

from models import *
from datasets import GraphDataset, collate_graph_data

def train_model(args):
    print("=== Set up trainer")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir)
    ckpt_callback = pl_callbacks.ModelCheckpoint(
        os.path.join(args.model_dir, 'ongoing'), 
        save_top_k=-1,
        every_n_epochs=5
    )
    trainer = pl.Trainer(
        accelerator='gpu', 
        gpus=1, 
        max_epochs=args.max_epochs,
        default_root_dir=args.model_dir,
        logger=tb_logger,
        callbacks=[ckpt_callback]
    )

    print("=== Load data")
    train_data = GraphDataset(args.dataset_dir)
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_graph_data
    )

    val_data = GraphDataset(args.val_dataset_dir)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_graph_data
    )

    print("=== Instantiate model and train")
    pl.seed_everything(args.random_seed)
    model = GLN(
        args.n_graph_layers, 
        args.size_gn_feat, 
        args.size_init_global_feat,
        input_image_stack_depth=20,
        num_input_image_sensors=args.num_sensors,
        lr=args.lr
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
        default="/data/home/joel/datasets/blocal_data_x3")
    parser.add_argument("--val_dataset_dir", type=str, help="Path to validation dataset",
        default="/data/home/joel/datasets/blocal_testing/blocal_val_data_x3")
    parser.add_argument("--model_dir", type=str, help="Path to save models to",
        default="/data/home/joel/datasets/models")
    parser.add_argument("--log_dir", type=str, help="Path for logging",
        default="/data/home/joel/datasets/logs")
    parser.add_argument("--batch_size", type=int, help="Training batch size",
        default=32)
    parser.add_argument("--max_epochs", type=int, help="Training max epochs",
        default=200)
    parser.add_argument("--lr", type=float, help="Learning rate",
        default=1e-4)
    parser.add_argument("--n_graph_layers", type=int, help="Number of GN blocks",
        default=2)
    parser.add_argument("--size_gn_feat", type=int, help="Size of GN features",
        default=256)
    parser.add_argument("--size_init_global_feat", type=int, help="Size of encoded visual features",
        default=512)
    parser.add_argument("--random_seed", type=int, help="Random seed",
        default=0)
    parser.add_argument("--num_workers", type=int, help="Number of workers",
        default=8)
    parser.add_argument("--num_sensors", type=int, help="Number of sensors used at each timestep",
        default=3)
    args = parser.parse_args()

    train_model(args)
