import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from dataset import BabyDataset
from model import resnet101
from util import EarlyStopping
import config as cfg



def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError

    # define dataset & dataloader
    dataset_train = BabyDataset(cfg.path.train_filelist)
    dataset_valid = BabyDataset(cfg.path.valid_filelist)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = cfg.train.batch_size,
                                               shuffle = True, drop_last = True,
                                               num_workers = cfg.train.num_worker)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size = cfg.train.batch_size,
                                               shuffle = False, drop_last = False,
                                               num_workers = cfg.train.num_worker)

    # initialize model
    #model = resnet101()
    #model.load_state_dict(torch.load(cfg.path.pretrained_model, map_location = lambda loc, storage: loc))
    model = torchvision.models.resnet101(pretrained = True)
    model.fc = nn.Linear(model.fc.in_features, cfg.train.num_class)
    model.to(device)

    # define optimizer
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr = cfg.train.learning_rate, betas = cfg.train.betas)

    # define logger
    logger = SummaryWriter(log_dir = cfg.path.log_dir)

    # define early stoping checker
    early_stopping = EarlyStopping(tolerance = cfg.train.tolerance, min_delta = cfg.train.min_delta)

    valid_loss_best = 1e5
    for epoch in range(1, cfg.train.num_epoch + 1):
        # train
        model.train()
        losses_train = []
        train_bar = tqdm(loader_train)
        for data in train_bar:
            images, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = cfg.train.loss_function(logits, labels, reduction = 'mean')
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())
            train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch, cfg.train.num_epoch, np.mean(losses_train))

        # validate
        model.eval()
        losses_valid, accs_valid = [], []
        with torch.no_grad():
            for i, data in enumerate(loader_valid):
                images, labels = data[0].to(device), data[1].to(device)
                logits = model(images)
                predict = torch.max(logits, dim = 1)[1]
                loss = cfg.train.loss_function(logits, labels, reduction = 'sum')
                acc = torch.eq(predict, labels).sum().item()
                losses_valid.append(loss.item())
                accs_valid.append(acc)
        print('valid epoch[{}/{}] loss:{:.5f} acc:{:.3f}'.format(epoch, cfg.train.num_epoch,
                                        np.sum(losses_valid) / len(dataset_valid), np.sum(accs_valid) / len(dataset_valid)))

        # logging
        logger.add_scalar('train_loss', np.mean(losses_train), global_step = epoch)
        logger.add_scalar('valid_loss', np.sum(losses_valid) / len(dataset_valid), global_step = epoch)
        logger.add_scalar('valid_acc', np.sum(accs_valid) / len(dataset_valid), global_step = epoch)

        if np.sum(losses_valid) / len(dataset_valid) < valid_loss_best:
            print('save model...')
            if epoch > 1:
                os.remove(os.path.join(cfg.path.log_dir, best_model))
            valid_loss_best = np.sum(losses_valid) / len(dataset_valid)
            torch.save(model.state_dict(), os.path.join(cfg.path.log_dir, f'baby_{epoch}.pt'))
            best_model = f'baby_{epoch}.pt'

        # early stopping
        early_stopping(np.mean(losses_train), np.sum(losses_valid) / len(dataset_valid))
        if early_stopping.early_stop:
            print("Early stopping. We are at epoch:", epoch)
            break
    print('Finished Training')



if __name__ == '__main__':
    main()
