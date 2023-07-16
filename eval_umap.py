import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from tqdm import tqdm
from PIL import Image
import clip
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import umap

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



def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    convertor_test = Baby_test()
    test_num = len(convertor_test)

    batch_size = 10
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 5])  # 8 number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(convertor_test,
                                               batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=nw)

    print("using {} images for testing.".format(test_num))
    net = resnet101()

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)  # classification amount
    net.to(device)

    # load pretrain weights
    model_weight_path = "./baby_resnet101_2.pth"  # save the weight
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), False)

    # test
    with torch.no_grad():
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        predicted_final = np.zeros(shape=(0))
        labels_final = np.zeros(shape=(0))
        with torch.no_grad():
            test_bar = tqdm(test_loader)

            data = torch.zeros(1, 2).cuda()
            a = torch.zeros(1)
            for test_data in test_bar:
                test_images, test_labels = test_data
                outputs = net(test_images.to(device))

                #
                data = torch.cat((data, outputs), 0)

                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                predicted_np = predict_y.cpu().numpy()
                labels_np = test_labels.cpu().numpy()
                labels_np = labels_np.astype(np.int)
                predicted_final = np.append(predicted_final, predicted_np)
                labels_final = np.append(labels_final, labels_np).astype(np.int)

                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

            # draw umap
            labels_final = torch.from_numpy(labels_final)
            labels_final = torch.cat((a, labels_final))
            label = labels_final.detach().numpy()
            print(label.shape)
            print(label)

            data = data.cpu().numpy()
            reducer = umap.UMAP(min_dist=0.5, n_neighbors=300, n_components=2,
                                random_state=42)  # min_dist: julong chengdu
            embedding = reducer.fit_transform(data)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)
            plt.gca().set_aspect('equal', 'datalim')
            plt.colorbar(boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))
            plt.title('UMAP projection of ResNet34')
            plt.show()

        test_accurate = acc / test_num
        print('test_accuracy: %.3f' %
              (test_accurate))

        print(metrics.confusion_matrix(labels_final, predicted_final))
        print(metrics.classification_report(labels_final, predicted_final, digits=2))

        # get confusion matrix
        cm = confusion_matrix(labels_final, predicted_final)
        # Normalize by row
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(labels_final,predicted_final)
        plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')
    print('Finished Testing')



if __name__ == '__main__':
    main()
