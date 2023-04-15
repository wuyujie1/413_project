
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class Boundary_Loss(nn.Module):
    def __init__(self):
        super(Boundary_Loss, self).__init__()
        self.reference_loss = nn.BCELoss()

    def __call__(self, y_pred, y_true, dist_maps):
        y_pred = torch.nn.Softmax(y_pred)

        prediction = y_pred.type(torch.float32)
        dist_map = dist_maps.type(torch.float32)

        boundary_loss = torch.einsum("bkwh,bkwh->bkwh", prediction, dist_map).mean()
        return self.reference_loss(y_pred, y_true) + 0.01 * boundary_loss


class Boundary_Loss_Modified(nn.Module):
    def __init__(self):
        super(Boundary_Loss_Modified, self).__init__()

    def __call__(self, y_pred, y_true, dist_maps):
        y_pred = torch.nn.Softmax(y_pred)

        dc = dist_maps.type(torch.float32)
        pc = y_pred.type(torch.float32)

        label_target = torch.where(dc <= 0., -dc.type(torch.double), torch.tensor(0.0).type(torch.double)).type(torch.float32)
        label_background = torch.where(dc > 0., dc.type(torch.double), torch.tensor(0.0).type(torch.double)).type(
            torch.float32)
        # label_background_unweighted = torch.where(dc > 0, 1, 0).type(torch.float32)

        c_fg = torch.einsum("bcwh,bcwh->bcwh", pc, label_target)
        sum_gt_fg = label_target.sum()
        ic_bg = sum_gt_fg - c_fg.sum()

        ic_fg = torch.einsum("bcwh,bcwh->bcwh", pc, label_background)

        boundary_loss = - ((c_fg.sum() / ((ic_bg + ic_fg.sum()) + 1e-10)).mean())

        return boundary_loss
