import torch
import torch.nn.functional as F
import math



def flow_loss_func(predictions, flow_gt, valid,
                   max_flow=400,
                   gamma=0.9,
                   var_max=1.0,    # Maximum variance value
                   var_min=-1.0,   # Minimum variance value
                   ):
    """
    Combined loss function incorporating MoL loss with MRF spatial regularization and homography constraints
    Args:
        predictions: Dictionary containing:
            - flow_preds: List of flow predictions at different scales/iterations
            - info_preds: List of info predictions containing weights and variance parameters
            - homography_preds: List of homography parameter predictions for geometric constraints
        flow_gt: Ground truth flow
        valid: Validity mask for flow
        max_flow: Maximum flow magnitude to consider
        gamma: Weight decay for different scales
        use_var: Whether to use variance-based loss
        var_max: Maximum variance value for large component
        var_min: Minimum variance value for small component
        mrf_weight: Weight for MRF spatial regularization term
        homography_weight: Weight for homography constraint loss term
        use_homography_smoothing: Whether to apply smoothing to homography parameters
        homography_smoothing_method: Type of smoothing ('gaussian', 'regional', 'hierarchical')
        homography_smoothness_weight: Weight for homography parameter smoothness regularization
        homography_guidance_type: Type of homography guidance ('regional_consistency', 'soft_constraint', 'flow_residual')
    """
    flow_preds = predictions['flow_preds']
    info_preds = predictions['info_preds']
    # Extract homography predictions for geometric constraints (may be empty during inference)
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5) & (mag < max_flow)

    # MoL loss with MRF spatial regularization
    for i in range(n_predictions):
        # Split info predictions into weights and variance parameters
        weight = info_preds[i][:, :2]  # First 2 channels are weights
        raw_b = info_preds[i][:, 2:]   # Last 2 channels are variance parameters
        
        log_b = torch.zeros_like(raw_b)
        # Large b Component                
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        
        # MoL term
        # term2: [N, 2, m, H, W]
        term2 = ((flow_gt - flow_preds[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        # term1: [N, m, H, W]
        term1 = weight - torch.log(torch.tensor(2.0, device=weight.device)) - log_b
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        
        # Handle NaN and Inf values in loss
        final_mask = (~torch.isnan(nf_loss.detach())) & (~torch.isinf(nf_loss.detach())) & valid[:, None]
        
        # Combine MoL and MRF losses with proper normalization
        i_weight = gamma**(n_predictions - i - 1)
        mol_loss = ((final_mask * nf_loss).sum() / (final_mask.sum() + 1e-6))
        
        flow_loss += i_weight * mol_loss

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        'mag': mag.mean().item()
    }

    return flow_loss, metrics
