from Pointnet_model import Pointnet
from modelnet40_data import PointnetDataset, load_data, PointSampler, Normalize, RandomNoise, RandomRotate, toTensor
from Pointnet_train_test import run

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    train_X, train_y = load_data()
    test_X, test_y = load_data(mode='test')

    default_transform = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            toTensor()]
    )

    train_transform = transforms.Compose(
        [
            PointSampler(1024),
            Normalize(),
            RandomNoise(),
            RandomRotate(),
            toTensor()]
    )

    trainset = PointnetDataset(train_X, train_y, transform=train_transform)
    testset = PointnetDataset(test_X, test_y, transform=default_transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('running on GPU')
    else:
        device = torch.device('cpu')
        print('running on CPU')

    model = Pointnet(device=device).to(device)
    n_epochs = 250

    train_hist, test_hist = run(model, n_epochs, device, trainloader, testloader)
    torch.save({
                'model_state_dict': model.state_dict()}, 'model.pt')