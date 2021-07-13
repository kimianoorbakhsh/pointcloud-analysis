import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm, trange

from Pointnet_model import loss_function

LOG_INTERVAL = 20


def train(model, optimizer, device, trainloader, verbose=True):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(tqdm(trainloader, desc='Batches', leave=False)):
        points, labels = data
        points = points.to(device)
        labels = labels.to(device)
        points = points.transpose(1, 2).float()

        optimizer.zero_grad()

        o, rot3, rot64 = model(points)

        loss, ce, reg = loss_function(o, labels, rot64, device)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        if verbose and batch_idx % LOG_INTERVAL == LOG_INTERVAL - 1:
            print('    Train [%d/%d]\t | \tLoss: %.5f, \tCross Entropy: %.5f, \tRegularization: %.5f' % (
            batch_idx * o.shape[0], len(trainloader.dataset), loss.item(), ce.item(), reg.item()))
    train_loss /= batch_idx
    if verbose:
        print('==> Train | Average loss: %.4f' % train_loss)
    return train_loss


def test(model, testloader, device, verbose=True):
    model.eval()
    test_loss = 0

    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            points, labels = data
            points = points.to(device)
            labels = labels.to(device)
            points = points.transpose(1, 2).float()

            o, rot3, rot64 = model(points)

            _, predicted = torch.max(o.data, 1)

            total += labels.shape[0]

            correct += (labels == predicted).sum().item()

            loss, _, _ = loss_function(o, labels, rot64, device)
            test_loss += loss.item()

        test_loss /= i
        acc = 100 * (correct / total)
        if verbose:
            print('==> Test  | Average loss: %.4f' % test_loss)
            print('==> Test  | Accuracy: %.4f' % acc)
        return test_loss, acc


def run(model, n_epoch, device, trainloader, testloader, optimizer_state_dict=None, verbose=True):
    model.to(device)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    train_hist = []
    test_hist = []
    for epoch in trange(1, n_epoch + 1, desc='Epochs', leave=True):
        if epoch % 20 == 19:
            lr = lr * 0.5
            # lr = max(lr, 1e-5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if verbose:
            print('\nEpoch %d:' % epoch)
        train_loss = train(model, optimizer, device, trainloader, verbose)
        test_loss, acc = test(model, testloader, device, verbose)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'checkpoint.pt')
        train_hist.append(train_loss)
        test_hist.append(test_loss)

    return train_hist, test_hist