import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.optim as optim


from tqdm.auto import tqdm, trange

import numpy as np

import sklearn.metrics as metrics
from dgcnn_model import DGCNN, cal_loss
LOG_INTERVAL = 5

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = train_data, train_labels
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
def train():
    ##args
    batch_size = 8
    test_batch_size = 16
    num_points = 2048
    k = 20
    emb_dims = 1024
    dropout_p = 0.5
    lr = 0.001
    momentum = 0.9
    use_sgd = True
    epochs = 250
    ####
    
    train_loader = DataLoader(ModelNet40(partition='train', num_points= num_points),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points= num_points),
                             batch_size=test_batch_size, shuffle=True, drop_last=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running on ", device)

    model = DGCNN(k, emb_dims, dropout_p, output_channels=40).to(device)

    if use_sgd:
        print("Using SGD")
        opt = optim.SGD(model.parameters(), lr=lr*100, momentum=momentum, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, epochs, eta_min=lr)
    else:
        print("Using Adam")
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)
        
    criterion = cal_loss
    best_test_acc = 0
    
    
    for epoch in trange(1, epochs+1, desc='Epochs', leave=True):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        
        for i, data in enumerate(tqdm(train_loader, desc='Batches', leave=False)):
            data, label = data[0].to(device), data[1].to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
            scheduler.step()
            
            
            if i % LOG_INTERVAL == LOG_INTERVAL - 1:
                 print('Train [%d/%d]\t | \tLoss: %.5f' % (i * batch_size, len(train_loader.dataset), loss.item() * batch_size))
        train_loss /= i
        print('==> epoch : %d Train | Average loss: %.5f' % (epoch, train_loss))
        
        ####################
        # Test
        ####################
        
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        
        for i, data in enumerate(test_loader):
            data, label = data[0].to(device), data[1].to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        print('==> Test | loss: %.5f, test acc: %.5f, test avg acc: %.5f' % ( test_loss*1.0/i,
                                                                              test_acc * 100,
                                                                              avg_per_class_acc * 100))
                                                                               
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, 'checkpoint.pt')
            
def test(model):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=num_points),
                             batch_size=test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("testing on ", device)
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    print('==> Test | loss: %.5f, test acc: %.5f, test avg acc: %.5f' % ( test_loss*1.0/i,
                                                                              test_acc,
                                                                              avg_per_class_acc))
    
train_data = np.load('modelnet40_train_data.npy')
test_data = np.load('modelnet40_test_data.npy').astype('float32')

train_labels = np.load('modelnet40_train_labels.npy').astype('int64')
test_labels = np.load('modelnet40_test_labels.npy').astype('int64')