import torch
import torch.nn as nn
import torch.nn.functional as F

#######6########
class SequenceLoss(nn.Module):
    def __init__(self, gamma: float = 0.9, max_flow: float = 400, var_min: float = 0, var_max: float = 10):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.var_min = var_min
        self.var_max = var_max

    def forward(self, outputs, flow_gt, valid, max_flow=None):
        """Loss function for sequence of flow predictions.
        Args:
            outputs: Dict with keys:
                - flow_preds: List[Tensor], each [B, 2, H, W]
                - info_preds: List[Tensor], each [B, 4, H, W]
            flow_gt: Tensor, [B, 2, H, W], ground truth flow.
            valid: Tensor, [B, H, W] or [B, 1, H, W], validity mask.
            max_flow: float, maximum flow magnitude for filtering.
        Returns:
            loss: Scalar, total weighted loss.
            metrics: Dict, evaluation metrics (e.g., EPE, mag).
        """
        if max_flow is None:
            max_flow = self.max_flow

        flow_preds = outputs["flow_preds"]
        info_preds = outputs["info_preds"]
        n_predictions = len(flow_preds)

        # Ensure valid has correct shape: [B, 1, H, W]
        if valid.dim() == 3:  # [B, H, W]
            valid = valid.unsqueeze(1)  # [B, 1, H, W]
        elif valid.dim() == 4 and valid.size(1) == 1:  # [B, 1, H, W]
            pass
        else:
            raise ValueError(f"Expected valid shape [B, H, W] or [B, 1, H, W], got {valid.shape}")

        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()  # [B, 1, H, W]
        valid = (valid >= 0.5) & (mag < max_flow)  # [B, 1, H, W]

        epe_list = []
        for i in range(n_predictions):
            flow_pred = flow_preds[i]  # [B, 2, H, W]
            epe = torch.norm(flow_gt - flow_pred, dim=1, p=2)  # [B, H, W]
            valid_mask = valid.squeeze(1).bool()  # [B, H, W]
            epe_list.append(epe[valid_mask].mean().item() if valid_mask.sum() > 0 else 0.0)

        flow_loss = 0.0
        nf_preds = []
        for i in range(n_predictions):
            raw_b = info_preds[i][:, 2:]  # [B, 2, H, W]
            log_b = torch.zeros_like(raw_b)
            weight = info_preds[i][:, :2]  # [B, 2, H, W]
            log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.var_max)  # Large b
            log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.var_min, max=0)  # Small b
            term2 = ((flow_gt - flow_preds[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
            term1 = weight - torch.log(torch.tensor(2.0, device=weight.device)) - log_b
            nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
            nf_preds.append(nf_loss)

            i_weight = self.gamma ** (n_predictions - i - 1)
            loss_i = nf_loss  # [B, 1, H, W]
            final_mask = valid & (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach()))
            flow_loss += i_weight * (final_mask * loss_i).sum() / (final_mask.sum() + 1e-10)

        mag_mean = mag[valid].mean().item() if valid.sum() > 0 else 0.0

        metrics = {
            "epe": epe_list[-1],
            "epe_all": epe_list,
            "mag": mag_mean
        }

        return flow_loss, metrics
#################