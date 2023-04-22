import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from dse4wse.gnn.dataloader import NoCeptionDataset
from dse4wse.gnn.model import NoCeptionNet
from dse4wse.utils import logger

def get_dataset():
    dataset = NoCeptionDataset(save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    return dataset

def get_model():
    model = NoCeptionNet(h_dim=64, n_layer=2, act_func='elu')
    return model

def run_model():
    test_data = get_dataset()[0]
    model = get_model()
    model(test_data)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint')

def train_model(model, dataset, batch_size=32):
    NUM_EPOCH = 50
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{timestamp}.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=1e-5)

    for epoch in range(NUM_EPOCH):
        total_loss = 0
        i = 0

        tqdm_bar = tqdm(dataset)
        for data in tqdm_bar:
            i += 1
            logits = model(data['graph'], data['graph_feat'])
            # loss = F.mse_loss(logits, label)
            loss = F.smooth_l1_loss(logits, data['label'])
            loss.backward()
            total_loss += loss.item()
            tqdm_bar.set_description(f"avg loss: {total_loss / i}")

            if i % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                for param in model.parameters():
                    param.grad /= batch_size
                optimizer.step()
                optimizer.zero_grad()

        lr_schduler.step()

        logger.info(f"Epoch {epoch}:")
        logger.debug(f"learning rate: {lr_schduler.get_last_lr()}")
        logger.debug(f"average loss: {total_loss / len(dataset)}")

    torch.save(model.state_dict(), checkpoint_path)

def test_model(model, dataset):
    total_mae = 0

    with torch.no_grad():
        for data in tqdm(dataset):
            logits = model(data['graph'], data['graph_feat'])
            label = data['label']
            mae = torch.abs(logits - label).mean()
            total_mae += mae.item()
            # logger.debug(f"MAE: {mae}")
    
    avg_mae = total_mae / len(dataset)
    logger.debug(f"Overall MAE: {avg_mae}")

def main():
    dataset = get_dataset()
    model = get_model()
    train_model(model, dataset)
    test_model(model, dataset)

if __name__ == "__main__":
    main()