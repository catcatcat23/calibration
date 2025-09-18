import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelMCDropout(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    """
    def __init__(self, model, M=100):
        super(ModelMCDropout, self).__init__()
        self.model = model
        # number of MC samples
        self.M = M

    def forward(self, x_1, x_2):
        # turn on dropout during inference
        self.model.train()
        logits_output_bos_mc, logits_output_bom_mc = [], []
        for i in range(self.M):
            logits_output_bos, logits_output_bom = self.model(x_1, x_2)
            logits_output_bos_mc.append(logits_output_bos)
            logits_output_bom_mc.append(logits_output_bom)

        return logits_output_bos_mc, logits_output_bom_mc