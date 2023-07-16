import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np

import torch
import torch.nn as nn
import torchvision

from dataset import BabyDataset
from model import resnet101
import config as cfg



def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError

    # define dataset & dataloader
    dataset_test = BabyDataset(cfg.path.test_filelist)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = cfg.train.batch_size,
                                              shuffle = False, drop_last = False,
                                              num_workers = cfg.train.num_worker)

    # initialize model
    #model = resnet101()
    #model.load_state_dict(torch.load(cfg.path.pretrained_model, map_location = lambda loc, storage: loc))
    model = torchvision.models.resnet101(pretrained = True)
    model.fc = nn.Linear(model.fc.in_features, cfg.train.num_class)
    model.load_state_dict(torch.load(cfg.path.trained_model))
    model.to(device)

    # evaluation
    model.eval()
    accs = []
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            images, labels = data[0].to(device), data[1].to(device)
            logits = model(images)
            predict = torch.max(logits, dim = 1)[1]
            acc = torch.eq(predict, labels).sum().item()
            accs.append(acc)

    # print
    acc = np.sum(accs) / len(dataset_test)
    print('Accuracy: {:.2f}'.format(acc * 100))
    with open(cfg.path.final_result, 'w') as f:
        f.write('Accuracy: {:.2f}'.format(acc * 100))
    print('Finished evaluation')



if __name__ == '__main__':
    main()
