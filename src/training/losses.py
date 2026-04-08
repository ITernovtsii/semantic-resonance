import torch
import torch.nn.functional as F


def bandpass_routing_loss(routing_probs, min_pct=0.001, max_pct=0.02, attention_mask=None):
    """
    Bandpass survival loss with floor (anti-death) and ceiling (anti-monopoly).
    Zero loss when all experts are in the [min_pct, max_pct] corridor.

    Args:
        routing_probs (Tensor): Pre-computed routing probabilities (B*S, N_experts).
        min_pct (float): Floor threshold (default 0.1% — experts below this are "dead").
        max_pct (float): Ceiling threshold (default 2.0% — experts above this are "monopolists").
        attention_mask (Tensor, optional): Boolean mask (B*S,) — True for real tokens, False for PAD.
    """
    N = routing_probs.size(-1)
    if attention_mask is not None:
        active = attention_mask.bool().view(-1)
        active_probs = routing_probs[active]
        if active_probs.size(0) == 0:
            return torch.tensor(0.0, device=routing_probs.device, requires_grad=True)
        mean_probs = active_probs.float().mean(dim=0)  # (N_experts,)
    else:
        mean_probs = routing_probs.float().mean(dim=0)  # (N_experts,)

    # Floor penalty: activates ONLY if traffic < min_pct (rescues dying experts)
    floor_penalty = torch.relu(min_pct - mean_probs)

    # Ceiling penalty: activates ONLY if traffic > max_pct (breaks monopolies)
    ceil_penalty = torch.relu(mean_probs - max_pct)

    # N² scaling for magnitude compatibility with CV² loss
    loss = (floor_penalty ** 2 + ceil_penalty ** 2).mean() * (N ** 2)
    return loss.to(routing_probs.dtype)


def elastic_ceiling_cv2_loss(routing_probs, ceiling=0.010, tax_multiplier=50.0, attention_mask=None):
    """
    Synthesis: CV² soft bottom + hard ceiling + PAD masking.
    - Soft bottom: continuous CV² gravity toward uniform (no dead zone, rescues dying experts)
    - Hard ceiling: squared penalty when expert exceeds ceiling (forces capacity brute-forcing)
    - ceiling: 0.010 (~128% of uniform 1/128)
    - tax_multiplier: 50.0 — makes ceiling penalty dominate when breached
    """
    N = routing_probs.size(-1)
    if attention_mask is not None:
        active = attention_mask.bool().view(-1)
        active_probs = routing_probs[active]
        if active_probs.size(0) == 0:
            return torch.tensor(0.0, device=routing_probs.device, requires_grad=True)
        mean_probs = active_probs.float().mean(dim=0)
    else:
        mean_probs = routing_probs.float().mean(dim=0)

    target_mu = 1.0 / N
    # Soft bottom: continuous CV² gravity (no dead zone)
    base_cv2 = ((mean_probs - target_mu) ** 2).mean() * (N ** 2)
    # Hard ceiling: only fires when expert > ceiling
    overflow = torch.relu(mean_probs - ceiling)
    ceiling_tax = (overflow ** 2).mean() * (N ** 2) * tax_multiplier

    return (base_cv2 + ceiling_tax).to(routing_probs.dtype)


def load_balancing_loss(routing_probs):
    """
    Calculates the Load Balancing Loss (L_balance) using Coefficient of Variation (CV).
    Ensures that all experts are utilized relatively equally (Switch Transformer approach).

    Args:
        routing_probs (Tensor): Pre-computed routing probabilities (B*S, N_experts).
            Already softmax'd in csr.py — no redundant softmax needed here.
    """
    # Calculate the average probability of selecting each expert across the batch
    mean_probs = routing_probs.float().mean(dim=0)  # (N_experts,)

    # Coefficient of Variation (CV) squared = Variance / Mean^2
    avg_mean_prob = mean_probs.mean()
    variance = ((mean_probs - avg_mean_prob) ** 2).mean()
    cv_squared = variance / (avg_mean_prob ** 2 + 1e-8)

    # Return raw CV^2 — it inherently scales with N_experts (no explicit multiplier needed)
    return cv_squared.to(routing_probs.dtype)


def calculate_sra_losses(logits, labels, aux_data_list, loss_config, use_aux_losses, attention_mask=None):
    """
    Central function to calculate all losses for the SRA model.
    Post-bugfix: only LM loss + load balancing (dispersion and z-loss removed).

    Args:
        attention_mask (Tensor, optional): (B, S) mask — True for real tokens, False for PAD.
            Flattened to (B*S,) and passed to bandpass loss to exclude PAD from routing statistics.
    """
    # 1. Language Modeling Loss (Cross-Entropy)
    lm_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    total_loss = lm_loss
    loss_dict = {"lm_loss": lm_loss}

    # 2. Auxiliary Loss (Load Balancing only)
    if use_aux_losses and aux_data_list:
        num_layers = len(aux_data_list)
        balance_weight = getattr(loss_config, 'balance_weight', 0.0)

        if balance_weight > 0:
            total_balance_loss = 0.0
            loss_type = getattr(loss_config, 'loss_type', None) or 'cv_squared'

            # Flatten attention_mask to (B*S,) to match routing_probs shape
            flat_mask = attention_mask.view(-1) if attention_mask is not None else None

            for aux_data in aux_data_list:
                if "routing_probs" in aux_data:
                    if loss_type == 'bandpass':
                        min_pct = getattr(loss_config, 'bandpass_min_pct', None) or 0.001
                        max_pct = getattr(loss_config, 'bandpass_max_pct', None) or 0.02
                        assert 0 < min_pct < max_pct < 1, f"Invalid bandpass bounds: {min_pct}, {max_pct}"
                        total_balance_loss += bandpass_routing_loss(aux_data["routing_probs"], min_pct, max_pct, flat_mask)
                    elif loss_type == 'elastic_ceiling_cv2':
                        ceiling = getattr(loss_config, 'ceiling', None) or 0.010
                        tax_mult = getattr(loss_config, 'tax_multiplier', None) or 50.0
                        total_balance_loss += elastic_ceiling_cv2_loss(
                            aux_data["routing_probs"], ceiling, tax_mult, flat_mask
                        )
                    else:
                        total_balance_loss += load_balancing_loss(aux_data["routing_probs"])

            avg_balance_loss = total_balance_loss / num_layers
            total_loss = total_loss + balance_weight * avg_balance_loss
            loss_dict["balance_loss"] = avg_balance_loss

    loss_dict["total_loss"] = total_loss
    return loss_dict
