import torch
from torch import nn, optim
from torch.nn import functional as F


def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    return torch.div(logits, temperature)

def optimize_temperature(nll_criterion, ece_criterion, logits, labels, T_init=1.0):
    # calculate NLL and ECE before temperature scaling
    before_TS_nll = nll_criterion(logits, labels).item()
    before_TS_ece = ece_criterion(logits, labels).item()
    print(f'Before TS - T={T_init}, NLL: {before_TS_nll:.3f}, ECE: {before_TS_ece:.3f}')
    # optimize the temperature w.r.t. NLL
    temperature = nn.Parameter((T_init*torch.ones(1)).cuda())
    optimizer_binary = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    def eval_binary():
        optimizer_binary.zero_grad()
        loss = nll_criterion(temperature_scale(logits, temperature), labels)
        loss.backward()
        return loss

    optimizer_binary.step(eval_binary)

    # calculate NLL and ECE after temperature scaling
    after_TS_nll = nll_criterion(temperature_scale(logits, temperature),
                                               labels).item()
    after_TS_ece = ece_criterion(temperature_scale(logits, temperature),
                                               labels).item()
    print(f'After TS - T_opt={temperature.item():.3f}, NLL: {after_TS_nll:.3f}, ECE: {after_TS_ece:.3f}')
    return temperature.item(), after_TS_nll, after_TS_ece


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    """
    def __init__(self, model, T_binary, T_mclass):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.T_mclass = nn.Parameter((torch.ones(1) * T_mclass).cuda())
        self.T_binary = nn.Parameter((torch.ones(1) * T_binary).cuda())

    def forward(self, x_1, x_2):
        logits_output_bos, logits_output_bom = self.model(x_1, x_2)
        return temperature_scale(logits_output_bos, self.T_binary), temperature_scale(logits_output_bom, self.T_mclass)



class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=25):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece