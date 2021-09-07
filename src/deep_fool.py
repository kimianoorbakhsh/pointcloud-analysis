import copy
import torch
import numpy as np
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torch.nn.functional as F

def deepfool_saliency(model, ori_pointcloud, others, max_iter=1000, overshoot=0.02, num_classes=10):
    i = 0

    x_pert = copy.deepcopy(ori_pointcloud)

    logits_ori, _, _ = model(torch.cat((ori_pointcloud, others), dim = 2))

    sorted_classes_ori = logits_ori.argsort(descending=True)[0]
    k_ori = sorted_classes_ori[0]

    logits_pert, _, _ = model(torch.cat((x_pert, others), dim = 2))
    sorted_classes_pert = logits_pert.argsort(descending=True)[0]
    k_pert = sorted_classes_pert[0]

    pert_final = np.zeros(ori_pointcloud.shape)
    while k_pert == k_ori and i < max_iter:

        logits_pert[0, sorted_classes_pert[0]].backward(retain_graph=True)
        true_grads = copy.deepcopy(x_pert.grad.data)
        zero_gradients(x_pert)

        min_w = 0
        min_f = 0
        min_dist = np.inf

        for j in range(1, num_classes):
            logits_pert[0, sorted_classes_pert[j]].backward(retain_graph=True)
            other_grad = copy.deepcopy(x_pert.grad.data)
            zero_gradients(x_pert)

            w = (other_grad - true_grads).detach().cpu().numpy()
            f = (logits_pert[0, j] - logits_pert[0, 0]).data.cpu().numpy()

            dist = np.abs(f) / np.linalg.norm(w.flatten())
            if dist < min_dist:
                min_dist = dist
                min_w = w
                min_f = f

        pert = (w / np.linalg.norm(w.flatten())) * (min_dist + 1e-4)
        pert_final = np.float32(pert_final + pert)

        x_pert = ori_pointcloud + (1 + overshoot) * torch.from_numpy(pert_final).to(device)

        x_pert = Variable(x_pert, requires_grad=True)

        logits_pert, _, _ = model(torch.cat((x_pert, others), dim = 2))
        sorted_classes_pert = logits_pert.argsort(descending=True)[0]
        k_pert = sorted_classes_pert[0]

        # print(k_pert.item(), logits_pert[0, sorted_classes_pert[0]].item(), min_dist)

        pert_final = pert_final * (1 + overshoot)
        i += 1
    confidence = F.log_softmax(logits_pert, dim=1)
    return pert_final, x_pert, k_pert.item(), F.softmax(logits_pert, dim=1)