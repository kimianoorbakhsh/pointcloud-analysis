import torch
import numpy as np
from Pointnet_model import loss_function


def dynamic_saliency_map(model, points, labels, device, n=100, T=20, alpha=1):
    model.eval()
    saliency_points = np.zeros((points.shape[0], 3, n))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    points = points.transpose(1, 2).float()  # B * 3 * 1024
    a = int(n/T)
    for i in range(T):
        points = points.to(device)
        labels = labels.to(device)
        # computing the gradient of the loss with respect to the points
        points.requires_grad = True
        o, rot3, rot64 = model(points)
        loss, _, _ = loss_function(o, labels, rot64, device)
        loss.backward()
        grad = points.grad
        # saliency map calculations
        sphere_core = torch.median(points, axis=2, keepdims=True)  # B * 3 * 1

        sphere_r = torch.sqrt(torch.sum(torch.square(
            points - sphere_core.values), axis=1))  # B * 1024

        sphere_axis = points - sphere_core.values  # B * 3 * 1024

        before_map = torch.sum(torch.multiply(
            grad, sphere_axis), axis=1)  # B * 1024

        map = -torch.multiply(before_map,
                              torch.pow(sphere_r, alpha))  # B * 1024

        drop_indices = torch.topk(map, a).indices  # B * n/T

        tmp = torch.zeros((points.shape[0], 3, points.shape[2] - a))
        numpy_points = points.cpu().detach().numpy()
        numpy_indices = drop_indices.cpu().detach().numpy()
        for j in range(points.shape[0]):
            saliency_points[j][:, i *
                               a: (i + 1) * a] = numpy_points[j][:, numpy_indices[j]]
            tmp[j] = torch.from_numpy(
                np.delete(numpy_points[j], numpy_indices[j], axis=1))

        points = tmp.detach().clone()

    # returns the points after dropping the most important ones and the most important points
    return points, saliency_points


def correct_reshape(point):
    p = torch.zeros((point.shape[1], point.shape[0]))
    for i in range(point.shape[1]):
        p[i] = torch.from_numpy(
            np.array([point[0][i], point[1][i], point[2][i]]))
    return p

def setdiff2(p1, p2):
  new_len = len(p1) - len(p2)
  diff_arr = []
  diff = np.zeros((new_len, 3))
  for i in range(len(p1)):
    if p1[i] not in p2:
      diff_arr.append(p1[i])
  for i in range(len(diff_arr)):
    diff[i] = diff_arr[i]
  return diff