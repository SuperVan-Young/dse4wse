import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import dgl
import torch.nn.functional as F

from dse4wse.gnn.dataloader import LinkUtilDataset
from dse4wse.gnn.model import HeteroNet
from dse4wse.utils import logger

def get_dataset():
    dataset = LinkUtilDataset(save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    return dataset

def get_model():
    model = HeteroNet(h_dim=64)
    return model

def run_model():
    test_data = get_dataset()[0]
    model = get_model()
    model(test_data)

def train_model(model, dataset):
    NUM_EPOCH = 30

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=1e-5)

    for epoch in range(NUM_EPOCH):
        total_loss = 0

        for data in dataset:
            logits = model(data)
            label = data.nodes['link'].data['label'].reshape(-1, 1).float()
            # loss = F.mse_loss(logits, label)
            loss = F.smooth_l1_loss(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        lr_schduler.step()

        logger.info(f"Epoch {epoch}:")
        logger.debug(f"learning rate: {lr_schduler.get_lr()}")
        logger.debug(f"average loss: {total_loss / len(dataset)}")

def test_model(model, dataset):
    with torch.no_grad():
        for data in dataset:
            logits = model(data)
            label = data.nodes['link'].data['label'].reshape(-1, 1).float()
            mae = torch.abs(logits - label).mean()
            logger.debug(f"MAE: {mae}")

def main():
    dataset = get_dataset()
    model = get_model()
    train_model(model, dataset)
    test_model(model, dataset)

if __name__ == "__main__":
    main()