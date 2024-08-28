import torch
from torch import nn


class SmoothLoss(nn.Module):
    def __init__(self, frame_dim=0, smooth_type=None, loss_type="l2"):
        super(SmoothLoss, self).__init__()
        self.frame_dim = frame_dim
        supported_smooth_types = ["mid_frame", "dislocation", "avg", "median"]
        assert smooth_type in supported_smooth_types, f"supported smooth type: {supported_smooth_types}"
        self.smooth_type = smooth_type
        supported_loss_types = ["l2", "mse", "l1"]
        assert loss_type in supported_loss_types, f"supported loss type: {supported_loss_types}"
        self.loss_type = loss_type
        if self.loss_type in ["l2", "mse"]:
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif self.loss_type in ["l1"]:
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise NotImplementedError

    def mid_frame_smooth(self, inputs):
        nframe = inputs.shape[self.frame_dim]
        mid_num = (nframe -1) // 2
        mid_frame = torch.index_select(inputs, self.frame_dim, torch.tensor([mid_num], device=inputs.device))
        repeat_num = self.get_repeat_num(inputs)
        smooth = mid_frame.repeat(repeat_num)
        loss = self.loss_fn(inputs, smooth)
        return loss

    def dislocation_smooth(self, inputs):
        nframe = inputs.shape[self.frame_dim]
        t = torch.index_select(inputs, self.frame_dim, torch.arange(0, nframe-1).to(inputs.device))
        t_1 = torch.index_select(inputs, self.frame_dim, torch.arange(1, nframe).to(inputs.device))
        loss = self.loss_fn(t, t_1)
        return loss

    def avg_smooth(self, inputs):
        avg = inputs.mean(dim=self.frame_dim, keepdim=True)
        repeat_num = self.get_repeat_num(inputs)
        smooth = avg.repeat(repeat_num)
        loss = self.loss_fn(inputs, smooth)
        return loss

    def median_smooth(self, inputs):
        median = inputs.median(dim=self.frame_dim, keepdim=True)[0]
        repeat_num = self.get_repeat_num(inputs)
        smooth = median.repeat(repeat_num).detach()
        loss = self.loss_fn(inputs, smooth)
        return loss

    def get_repeat_num(self, inputs):
        repeat_num = [1] * inputs.dim()
        repeat_num[self.frame_dim] = inputs.shape[self.frame_dim]
        return repeat_num

    def forward(self, inputs):
        if self.smooth_type is None:
            return 0.
        elif self.smooth_type == "mid_frame":
            return self.mid_frame_smooth(inputs)
        elif self.smooth_type == "dislocation":
            return self.dislocation_smooth(inputs)
        elif self.smooth_type == "avg":
            return self.avg_smooth(inputs)
        elif self.smooth_type == "median":
            return self.median_smooth(inputs)
        else:
            raise NotImplementedError