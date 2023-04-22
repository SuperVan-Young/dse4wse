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

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint')
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

def get_dataset(training=True):
    dataset = NoCeptionDataset(save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train' if training else 'test'))
    return dataset

def get_model(save_path=None):
    model = NoCeptionNet(h_dim=64, n_layer=2, act_func='elu')
    if save_path:
        model.load_state_dict(torch.load(save_path))
    return model

def train_model(model, dataset, batch_size=32):
    NUM_EPOCH = 50
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{timestamp}.pth")
    logger.info(f"Model checkpoint path: {checkpoint_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_schduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, threshold=1e-3)

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
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                for param in model.parameters():
                    param.grad /= batch_size
                optimizer.step()
                optimizer.zero_grad()

        lr_schduler.step(total_loss / len(dataset))

        logger.info(f"Epoch {epoch}:")
        logger.info(f"learning rate: {lr_schduler._last_lr}")
        logger.info(f"average loss: {total_loss / len(dataset)}")

        torch.save(model.state_dict(), checkpoint_path)

def test_model(model, dataset):
    total_mae = 0
    total_mape = 0

    with torch.no_grad():
        for data in tqdm(dataset):
            logits = model(data['graph'], data['graph_feat'])
            label = data['label']
            mae = torch.abs(logits - label).mean()
            mape = mae / label
            total_mae += mae.item()
            total_mape += mape.item()
            # logger.debug(f"MAE: {mae}")
    
    avg_mae = total_mae / len(dataset)
    avg_mape = total_mape / len(dataset)

    logger.info(f"Overall MAE: {avg_mae}")
    logger.info(f"Overall MAPE: {avg_mape}")

def main():
    model = get_model()
    # model = get_model(os.path.join(CHECKPOINT_DIR, "model_2023-04-22-15-00-11-015813.pth"))
    train_model(model, get_dataset(training=True))

    model.eval()
    test_model(model, get_dataset(training=True))
    test_model(model, get_dataset(training=False))

if __name__ == "__main__":
    main()