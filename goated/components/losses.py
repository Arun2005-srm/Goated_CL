"""Distillation losses and entropy-gated multi-source fusion (components 1, 2, 4, 9)."""
import math
import torch
import torch.nn.functional as F


def soften(p1, T):
    """Distribution at temperature T from a T=1 distribution: softmax(log p / T)."""
    if T == 1:
        return p1
    return torch.softmax(torch.log(p1.clamp_min(1e-8)) / T, dim=1)


def match_size(p, size):
    if p.shape[-2:] == tuple(size):
        return p
    p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
    return p / p.sum(1, keepdim=True).clamp_min(1e-8)


def kd_loss(student_logits, target_pT, T, weight=None):
    """T^2 * KL(target || student_T), averaged over pixels. `weight` broadcasts to [B,1,h,w]."""
    target_pT = match_size(target_pT, student_logits.shape[-2:])
    logq = F.log_softmax(student_logits.float() / T, dim=1)
    kl = (target_pT * (torch.log(target_pT.clamp_min(1e-8)) - logq)).sum(1, keepdim=True)
    if weight is not None:
        kl = kl * weight
    return kl.mean() * T * T


def normalized_entropy(p1):
    return -(p1 * torch.log(p1.clamp_min(1e-8))).sum(1, keepdim=True) / math.log(p1.shape[1])


def fuse(sources, T, tau, size=None):
    """Entropy-gated fusion. sources: [(p1 [B,C,h,w], base_weight broadcastable)].
    alpha_k = w_k * exp(-H(p1_k)/tau); target = sum alpha_k soften(p1_k) / sum alpha_k.
    Returns (target at temperature T, per-pixel loss weight normalised to mean 1)."""
    size = size or sources[0][0].shape[-2:]
    num, den = 0.0, 0.0
    for p1, w in sources:
        p1 = match_size(p1, size)
        a = w * torch.exp(-normalized_entropy(p1) / tau)
        num = num + a * soften(p1, T)
        den = den + a
    target = num / den.clamp_min(1e-8)
    pw = den / den.mean().clamp_min(1e-8)
    return target, pw.detach()


def multi_source_kd(student_logits, sources, T, use_fusion, tau):
    """Either one KD to the fused target, or a weighted sum of independent KD terms."""
    sources = [(p, w) for p, w in sources if p is not None]
    if not sources:
        return student_logits.new_zeros(())
    if use_fusion and len(sources) > 1:
        target, pw = fuse(sources, T, tau, student_logits.shape[-2:])
        return kd_loss(student_logits, target, T, pw)
    total = 0.0
    for p1, w in sources:
        total = total + kd_loss(student_logits, soften(match_size(p1, student_logits.shape[-2:]), T),
                                T, w if torch.is_tensor(w) else None) * (1.0 if torch.is_tensor(w) else w)
    return total


def pseudo_label_ce(student_logits, teacher_p1, threshold, ignore):
    """DoSe Eq. 8-9: CE on (flip-averaged) teacher argmax, pixels below confidence ignored."""
    p = match_size(teacher_p1, student_logits.shape[-2:])
    conf, lab = p.max(1)
    lab[conf < threshold] = ignore
    if (lab != ignore).sum() == 0:
        return student_logits.new_zeros(())
    return F.cross_entropy(student_logits.float(), lab, ignore_index=ignore)


def supervised_ce(student_logits, mask, ignore, class_weights=None):
    logits = F.interpolate(student_logits.float(), size=mask.shape[-2:], mode="bilinear", align_corners=False)
    if (mask != ignore).sum() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(logits, mask, weight=class_weights, ignore_index=ignore)
