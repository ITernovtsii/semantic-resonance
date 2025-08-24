import torch
import torch.nn.functional as F


def dispersion_loss(anchors_norm):
    """
    Calculates the Dispersion Loss (L_dispersion).
    Encourages semantic specialization by minimizing the average pairwise cosine similarity between anchors.
    
    Args:
        anchors_norm (Tensor): Normalized semantic anchors (N_experts, D).
    """
    N = anchors_norm.size(0)
    if N <= 1: return torch.tensor(0.0, device=anchors_norm.device)

    # Calculate pairwise similarity matrix: (N, D) @ (D, N) -> (N, N)
    # Perform in FP32 for precision
    similarity_matrix = torch.matmul(anchors_norm.float(), anchors_norm.float().T)

    # We want to ignore the diagonal (similarity with itself, which is always 1)
    mask = torch.eye(N, device=anchors_norm.device).bool()

    # Fill the diagonal with zeros before summing
    similarity_matrix = similarity_matrix.masked_fill(mask, 0)

    # Calculate the average similarity of off-diagonal elements (N*(N-1))
    loss = similarity_matrix.sum() / (N * (N - 1))

    # We minimize this average similarity to maximize dispersion.
    return loss.to(anchors_norm.dtype)


def load_balancing_loss(resonance_scores):
    """
    Calculates the Load Balancing Loss (L_balance) using Coefficient of Variation (CV).
    Ensures that all experts are utilized relatively equally (Switch Transformer approach).
    
    Args:
        resonance_scores (Tensor): Raw resonance scores before Top-K (B*S, N_experts).
    """
    N_experts = resonance_scores.size(1)

    # Calculate probabilities across all experts (in FP32 for stability)
    routing_probs = F.softmax(resonance_scores.float(), dim=-1)  # (B*S, N_experts)

    # Calculate the average probability of selecting each expert across the batch
    mean_probs = routing_probs.mean(dim=0)  # (N_experts,)

    # Calculate the Coefficient of Variation (CV) squared.
    # CV^2 = Variance / Mean^2.

    # Mean of the probabilities across experts
    avg_mean_prob = mean_probs.mean()

    # Variance of the probabilities across experts
    variance = ((mean_probs - avg_mean_prob) ** 2).mean()

    # CV squared (with epsilon for stability)
    cv_squared = variance / (avg_mean_prob ** 2 + 1e-8)

    # The loss is proportional to CV squared. Scaling by N_experts helps maintain scale invariance.
    loss = cv_squared * N_experts
    return loss.to(resonance_scores.dtype)


def router_z_loss(resonance_scores):
    """Calculates the Z-loss for the router logits."""
    # Calculate in FP32
    scores = resonance_scores.float()
    # log(sum(exp(scores)))^2 - stabilizes logit magnitude.
    loss = torch.logsumexp(scores, dim=-1).pow(2).mean()
    return loss.to(resonance_scores.dtype)


def calculate_sra_losses(logits, labels, aux_data_list, loss_config, use_aux_losses):
    """
    Central function to calculate all losses for the SRA model.
    """
    # 1. Language Modeling Loss (Cross-Entropy)
    # Flatten logits (B*S, Vocab) and labels (B*S)
    lm_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        # ignore_index can be used if there is padding, but we have fixed chunks.
    )

    total_loss = lm_loss
    loss_dict = {"lm_loss": lm_loss}

    # 2. Auxiliary Losses (Balance, Dispersion, Z-loss)
    if use_aux_losses and aux_data_list:
        num_layers = len(aux_data_list)

        # Read weights upfront
        balance_weight = getattr(loss_config, 'balance_weight', 0.0)
        dispersion_weight = getattr(loss_config, 'dispersion_weight', 0.0)
        z_loss_weight = getattr(loss_config, 'z_loss_weight', 0.001)

        compute_balance = balance_weight > 0
        compute_dispersion = dispersion_weight > 0
        compute_z = z_loss_weight > 0

        total_balance_loss = 0.0
        total_dispersion_loss = 0.0
        total_z_loss = 0.0

        # Aggregate losses from all CSR layers (only necessary components)
        for aux_data in aux_data_list:
            # L_balance & Z-loss use resonance_scores
            if "resonance_scores" in aux_data:
                scores = aux_data["resonance_scores"]
                if compute_balance:
                    total_balance_loss += load_balancing_loss(scores)
                if compute_z:
                    total_z_loss += router_z_loss(scores)

            # L_dispersion uses anchors_norm
            if compute_dispersion and "anchors_norm" in aux_data:
                total_dispersion_loss += dispersion_loss(aux_data["anchors_norm"])

        aux_loss_components = []

        # Average and add weighted components only if they were computed
        if compute_balance:
            avg_balance_loss = total_balance_loss / num_layers
            aux_loss_components.append(balance_weight * avg_balance_loss)
            loss_dict["balance_loss"] = avg_balance_loss
        if compute_dispersion:
            avg_dispersion_loss = total_dispersion_loss / num_layers
            aux_loss_components.append(dispersion_weight * avg_dispersion_loss)
            loss_dict["dispersion_loss"] = avg_dispersion_loss
        if compute_z:
            avg_z_loss = total_z_loss / num_layers
            aux_loss_components.append(z_loss_weight * avg_z_loss)
            loss_dict["z_loss"] = avg_z_loss

        if len(aux_loss_components) > 0:
            aux_loss = sum(aux_loss_components)
            total_loss += aux_loss
            # Log components
            loss_dict["aux_loss"] = aux_loss

    loss_dict["total_loss"] = total_loss
    return loss_dict
