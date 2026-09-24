import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorScaling(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits):
        return logits * self.log_scale.exp().clamp(0.25, 4.0) + self.bias

def fit_vector_scaling(logits, labels, max_iter=100):
    """Post-hoc per-attribute calibration on development validation logits."""
    device = logits.device
    cal = VectorScaling(logits.shape[1]).to(device)
    opt = torch.optim.LBFGS(cal.parameters(), lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
    x, y = logits.detach(), labels.detach()
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(cal(x), y)
        loss.backward()
        return loss
    opt.step(closure)
    return {
        "log_scale": cal.log_scale.detach().cpu(),
        "bias": cal.bias.detach().cpu(),
    }

def apply_calibration(logits, state):
    if state is None:
        return logits
    ls = state["log_scale"].to(logits.device)
    b = state["bias"].to(logits.device)
    return logits * ls.exp().clamp(0.25,4.0) + b
