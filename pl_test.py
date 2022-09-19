import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

class TestModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(1, 1)

    def forward(self, batch):
        dims = batch.shape
        out = torch.zeros(batch.shape)
        for i in range(dims[0]):
            for j in range(dims[1]):
                out[i, j] = self.layer(batch[i, j].unsqueeze(0).unsqueeze(0))[0, 0]
        return out

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def training_step(self, batch, batch_idx):
        return 0.0

    def predict_step(self, batch, batch_idx):
        return self.forward(batch), batch.shape[0]

class TestDataset(Dataset):
    def __init__(self):
        self.data = [torch.rand(i) for i in range(1, 5)]
        # self.data = [torch.rand(4) for i in range(1, 5)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch):
    if len(batch) == 1:
        return batch[0].unsqueeze(1)
    else:
        raise Exception("Not implemented")


if __name__ == "__main__":
    dataset = TestDataset()
    loader =  DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = TestModel()
    trainer = pl.Trainer(accelerator='cpu')
    predictions = trainer.predict(model, loader)

    print(predictions)