"""
spatial_heterogeneity_and_pmr_purification_analysis.py
================================
 for i in range(6):                                   # each stripe
    F = feats[:, i, :]                               # (N, 2048)  only stripe i
    F = standardize_per_channel(F)                   # z-score each of 2048 channels

    d_list = []
    for pid in unique_identities:                    # real labels, not 0..95
        rgb = F[(ids == pid) & (modality == 0)]      # (n_rgb_pid, 2048)
        ir  = F[(ids == pid) & (modality == 1)]      # (n_ir_pid,  2048)
        if len(rgb) == 0 or len(ir) == 0:
            continue                                 # skip if a modality is missing

        rgb_mean = rgb.mean(axis=0)                  # (2048,)
        ir_mean  = ir.mean(axis=0)                   # (2048,)
        rgb_mean /= norm(rgb_mean)                   # unit length
        ir_mean  /= norm(ir_mean)

        d_list.append(1 - dot(rgb_mean, ir_mean))    # one number for this identity

    bar_height = mean(d_list)                        # plotted bar
    lo, hi = bootstrap_ci(d_list)                    # error bar (2000 resamples)


  R1) "You need to quantitatively validate your core assumption of spatial
      heterogeneity."
  R2) "The claimed spatially heterogeneous modality discrepancy is not
      directly validated, while the part branch relies on fixed horizontal
      stripes."
  R3) "The main motivation for PMR lacks direct validation ... A region-wise
      analysis before and after PMR would better support BOTH the spatial
      heterogeneity assumption AND the claimed modality purification effect."

Design, mapped to the comments
-------------------------------
This script uses THREE checkpoints, baseline->before->after

    --plain_resume  : model was never trained on PMR supervision, i.e baseline resnet50 trained on SYSU-MM01
    --no_pmr_resume : parts=True, PMR=False, same TMS/MAR/PGFA as the
                       proposed model (the paper's Fig. 4(a) "Without PMR"
                       row: part branch present, trained with only the
                       part identity loss L^P_id, no L^P_PMR).
                       Used ONLY by Step 2 (the "no_pmr" point).
    --pmr_resume    : Model trained with PMR

STEP 1 : Measure modality discrepancy on a GENERIC grid over the
    --plain_resume. This backbone has never been exposed to any
    stripe-shaped loss (no part branch, no PMR, no TMS), so a reviewer
    cannot object that the measured spatial structure is an artifact of
    the network having been trained to respect the 6-stripe partition.
    it is the strongest, assumption-free evidence for R1/R2's "validate
    spatial heterogeneity, independent of the fixed-stripe design" ask.
    We then aggregate the same grid into the model's actual num_parts
    stripes and report an alignment diagnostic (within-stripe vs.
    between-stripe variance) that tells you whether horizontal stripes are
    actually a reasonable way to carve up the measured heterogeneity, or
    whether they wash out structure that a finer/different partition would
    keep.

STEP 2 : Region-wise (per native stripe) modality discrepancy at
    THREE points, all pooled with the SAME NativeStripeExtractor:
        baseline = --plain_resume backbone (no part branch/PMR/TMS)
        no_pmr   = --no_pmr_resume backbone + part_bns output. without PMR only part branch use.
        with_pmr = --pmr_resume PMR trained model
    HEADLINE FIGURE (one figure for R1-C2 and R2-C7):
        step2_reduction/bar_baseline_no_pmr_pmr.png  -- 6 stripes x {Baseline, No PMR, PMR}.
    Only No PMR -> PMR is PMR-attributable (and only if those two checkpoints
    differ in exactly PMR -- VERIFY against Fig. 4(a)); Baseline -> No PMR
    bundles the part branch and any other component enabled in that checkpoint.

"""

from __future__ import annotations
import os
import csv
import json
import ast
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.transforms import Bbox
from matplotlib.font_manager import FontProperties

# --------------------------------------------------------------------------
# Plot style. Every text element in every figure is derived from ONE base
# font size, set with --font_size (default 17 = the original look). Ticks are
# drawn 2 pt smaller than labels/legend, as before.
# --------------------------------------------------------------------------
FIG5_AXES_HEIGHT_IN = 3.6   # plotting-area height (inches) shared by the Fig. 5
                            # bar chart and heatmap; set with --axes_height


def apply_plot_style(font_size: float = 17.0, axes_height_in: Optional[float] = None):
    global FIG5_AXES_HEIGHT_IN
    if axes_height_in is not None:
        FIG5_AXES_HEIGHT_IN = float(axes_height_in)
    fs = float(font_size)
    plt.rcParams.update({
        "font.size": fs,
        "font.family": "serif",
        "axes.labelsize": fs,
        "axes.labelweight": "normal",
        "axes.linewidth": 1.5,
        "xtick.labelsize": fs - 2,
        "ytick.labelsize": fs - 2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.fontsize": fs,
        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "patch.linewidth": 1.2,
        "figure.dpi": 300,     # on-screen only; savefig.dpi below controls the file
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


def _pts(key: str) -> float:
    """rcParams font size (may be a number or a word like 'large') -> points."""
    return FontProperties(size=plt.rcParams[key]).get_size_in_points()


apply_plot_style(17.0)


# --------------------------------------------------------------------------
# Fixed-height layout shared by the two Fig. 5 panels.
# Both figures get the SAME total height, the SAME axes height and the SAME
# top/bottom margins (all in inches, derived from the font size), so the two
# PNGs have identical pixel heights and their plotting areas line up exactly
# when placed side by side. Only the width is cropped tightly.
# --------------------------------------------------------------------------
def _fig5_vertical_margins() -> Tuple[float, float]:
    tick, lab, leg = _pts("xtick.labelsize"), _pts("axes.labelsize"), _pts("legend.fontsize")
    pads = (plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"]
            + plt.rcParams["axes.labelpad"])
    bottom = (1.25 * tick + 1.3 * lab + pads) / 72.0 + 0.10
    top = (2.0 * leg) / 72.0 + 0.15            # one-row legend above the bars
    return top, bottom


def _new_fig5_canvas(axes_width_in: float):
    """Figure with one axes of fixed height FIG5_AXES_HEIGHT_IN, generous
    horizontal room (cropped away again at save time)."""
    top, bottom = _fig5_vertical_margins()
    left_room, right_room = 5.0, 5.0       # cropped away at save time
    W = left_room + axes_width_in + right_room
    H = bottom + FIG5_AXES_HEIGHT_IN + top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left_room / W, bottom / H, axes_width_in / W, FIG5_AXES_HEIGHT_IN / H])
    return fig, ax


def _wrap_if_too_long(fig, text_obj, avail_in: float):
    """Break a vertical axis/colorbar label onto two lines if, at the current
    font size, it is taller than the axes it belongs to."""
    fig.canvas.draw()
    ext = text_obj.get_window_extent(fig.canvas.get_renderer())
    if ext.height / fig.dpi > avail_in and "\n" not in text_obj.get_text():
        words = text_obj.get_text().split(" ")
        if len(words) > 1:
            cut = (len(words) + 1) // 2
            text_obj.set_text(" ".join(words[:cut]) + "\n" + " ".join(words[cut:]))


def _save_fixed_height(fig, out_path: str, pad_in: float = 0.05):
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    r = fig.canvas.get_renderer()
    # Union of the figure's tight bbox with the full extent of every axis
    # label: get_tightbbox() treats axis labels as zero-width for layout
    # purposes, which would crop a label wider than its (narrow) axes.
    wm = [t for t in fig.findobj(matplotlib.text.Text) if t.get_gid() == "watermark"]
    for t in wm: t.set_visible(False)
    boxes = [fig.get_tightbbox(r)]
    for a in fig.axes:
        for t in (a.xaxis.label, a.yaxis.label):
            if t.get_visible() and t.get_text():
                e = t.get_window_extent(r)
                boxes.append(Bbox.from_extents(e.x0 / fig.dpi, e.y0 / fig.dpi,
                                               e.x1 / fig.dpi, e.y1 / fig.dpi))
    tb = Bbox.union(boxes)                               # inches
    for t in wm: t.set_visible(True)
    if tb.y0 < -1e-3 or tb.y1 > H + 1e-3:
        print(f"[plot] WARNING: some text in {os.path.basename(out_path)} extends past "
              "the fixed vertical margins and will be clipped; try a smaller --font_size "
              "or a larger --axes_height.")
    crop = Bbox([[max(tb.x0 - pad_in, 0.0), 0.0], [min(tb.x1 + pad_in, W), H]])
    fig.savefig(out_path, bbox_inches=crop)
    plt.close(fig)


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def _standardize_channels(x: np.ndarray, eps_rel: float = 1e-2) -> np.ndarray:
    """Per-channel z-score standardization (zero mean, unit variance),
    fit fresh on whatever samples are passed in. This is the SAME operation
    BatchNorm performs. It matters here because we deliberately compare
    features from very different points in the pipeline on a single cosine-
    distance scale: raw post-ReLU GeM-pooled features (non-negative,
    magnitude dominated by a handful of channels -> cosine similarity
    inflated almost by construction) vs. BatchNorm+LayerNorm'd PMR features
    (zero-centered, channel-balanced). Comparing those directly makes any
    'raw' stage look artificially low-discrepancy regardless of true
    content -- standardizing every stage the same way before computing
    cosine discrepancy removes that artifact while preserving the genuine
    RGB/IR mean-shift signal (standardization only rescales axes; both
    modalities get the exact same per-channel mean/std within a region).

    The std floor is RELATIVE to the typical channel std in this array, not
    a tiny absolute constant. Post-ReLU conv features are often sparse
    (many near-dead channels with std close to zero); dividing those by an
    absolute epsilon like 1e-6 amplifies their sampling noise into huge
    standardized values that swamp the real signal. Flooring at a fraction
    of the median channel std keeps near-dead channels near-zero instead."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    floor = max(eps_rel * float(np.median(std)), 1e-8)
    return (x - mean) / np.clip(std, floor, None)


def linear_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    """Linear-kernel MMD^2 (= squared distance between sample means).
    Cheap, interpretable, easy to permute for significance. This is a
    population-level, identity-agnostic cross-check -- NOT the primary
    discrepancy metric (see per_identity_centroid_discrepancy below), since
    it is sensitive to which identities happen to have more RGB vs. IR
    samples. Report it as a secondary robustness check only."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    return float(np.sum((mx - my) ** 2))


@dataclass
class RegionSamples:
    feats: np.ndarray      # (N, D)
    ids: np.ndarray        # (N,)
    modality: np.ndarray   # (N,)  0 = RGB, 1 = IR


def per_identity_centroid_discrepancy(rs: RegionSamples) -> Dict[int, float]:
    """{identity: 1 - cos(centroid_RGB, centroid_IR)} for identities present
    in both modalities within this region. Computed region-by-region."""
    out = {}
    for pid in np.unique(rs.ids):
        mask_id = rs.ids == pid
        rgb = rs.feats[mask_id & (rs.modality == 0)]
        ir = rs.feats[mask_id & (rs.modality == 1)]
        if len(rgb) == 0 or len(ir) == 0:
            continue
        c_rgb = _l2_normalize(rgb.mean(axis=0))
        c_ir = _l2_normalize(ir.mean(axis=0))
        out[int(pid)] = float(1.0 - np.dot(c_rgb, c_ir))
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95,
                  seed: Optional[int] = 0) -> Tuple[float, float, float]:
    """Vectorized percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = values[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def permutation_test_mmd(rgb_feats: np.ndarray, ir_feats: np.ndarray,
                          n_perm: int = 2000, seed: Optional[int] = 0) -> Tuple[float, float]:
    """One-sided permutation test on linear MMD^2. Vectorized: all n_perm
    random group assignments generated at once, one matmul for every null
    MMD^2 value."""
    n_rgb, n_ir = len(rgb_feats), len(ir_feats)
    if n_rgb == 0 or n_ir == 0:
        return float("nan"), float("nan")
    observed = linear_mmd2(rgb_feats, ir_feats)

    pooled = np.concatenate([rgb_feats, ir_feats], axis=0).astype(np.float32, copy=False)
    n_total = len(pooled)
    rng = np.random.default_rng(seed)

    total_sum = pooled.sum(axis=0)
    ranks = np.argsort(rng.random((n_perm, n_total)), axis=1)
    mask = (ranks < n_rgb).astype(np.float32)
    group_a_sum = mask @ pooled
    group_b_sum = total_sum[None, :] - group_a_sum
    diff = group_a_sum / n_rgb - group_b_sum / n_ir
    null = np.sum(diff ** 2, axis=1)

    p_value = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return observed, p_value


def mean_inter_identity_distance(rs: RegionSamples) -> float:
    """Mean pairwise (1 - cos) between identity centroids, each centroid
    pooled over BOTH modalities, restricted to identities present in both
    modalities (same set used by the primary metric). Used ONLY as a
    normaliser for the robustness column `relative_gap`.

    Why it is needed: the primary metric is an absolute cosine distance
    measured in three DIFFERENT feature spaces (plain backbone / part_bns /
    linear-projection + LayerNorm). A learned projection can change the
    anisotropy of the space, shrinking ALL cosine distances, which would look
    like 'purification' without any modality-specific effect. Dividing by the
    inter-identity spread of the same space cancels such global rescaling."""
    cents = []
    for pid in np.unique(rs.ids):
        m = rs.ids == pid
        if (m & (rs.modality == 0)).any() and (m & (rs.modality == 1)).any():
            cents.append(_l2_normalize(rs.feats[m].mean(axis=0)))
    if len(cents) < 2:
        return float("nan")
    C = np.stack(cents)
    sim = C @ C.T
    iu = np.triu_indices(len(C), k=1)
    return float(np.mean(1.0 - sim[iu]))


@dataclass
class RegionResult:
    region_id: int
    n_identities: int
    centroid_mean: float
    centroid_ci_lo: float
    centroid_ci_hi: float
    mmd2: float
    mmd2_pvalue: float
    per_identity: Dict[int, float] = field(default_factory=dict)
    inter_id_dist: float = float("nan")   # normaliser, see mean_inter_identity_distance
    relative_gap: float = float("nan")    # centroid_mean / inter_id_dist (robustness only)


def analyze_region(rs: RegionSamples, region_id: int, n_boot: int = 2000,
                    n_perm: int = 2000, seed: Optional[int] = 0,
                    standardize: bool = True) -> RegionResult:
    """standardize=True (default) applies _standardize_channels to rs.feats
    before computing either metric, fit fresh on this region's own samples.
    Required for ANY comparison across pipeline stages with different
    normalization histories (e.g. raw backbone vs. BatchNorm'd PMR output)
    to be on a comparable cosine-distance scale -- see _standardize_channels
    docstring. Keep True unless you have a specific reason to compare raw,
    unstandardized magnitudes."""
    feats = _standardize_channels(rs.feats) if standardize else rs.feats
    rs_use = RegionSamples(feats, rs.ids, rs.modality)
    per_id = per_identity_centroid_discrepancy(rs_use)
    vals = np.array(list(per_id.values()))
    mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
    rgb = feats[rs.modality == 0]
    ir = feats[rs.modality == 1]
    mmd2, p = permutation_test_mmd(rgb, ir, n_perm=n_perm, seed=seed)
    inter = mean_inter_identity_distance(rs_use)
    rel = mean / inter if (inter == inter and inter > 1e-9) else float("nan")
    return RegionResult(region_id, len(per_id), mean, lo, hi, mmd2, p, per_id,
                        inter_id_dist=inter, relative_gap=rel)


def heterogeneity_test(region_results: List[RegionResult]) -> Dict[str, float]:
    """Friedman (repeated measures, matched identities across regions) is
    primary; Kruskal-Wallis reported for reference only. Works on ANY list
    of RegionResult -- fine grid (many regions), stripes, or pipeline
    stages (baseline/before/after) -- reused throughout."""
    if len(region_results) < 2:
        return {"friedman_stat": float("nan"), "friedman_p": float("nan")}

    common_ids = set(region_results[0].per_identity.keys())
    for r in region_results[1:]:
        common_ids &= set(r.per_identity.keys())
    common_ids = sorted(common_ids)

    if len(common_ids) < 3:
        return {"friedman_stat": float("nan"), "friedman_p": float("nan"),
                "n_matched_identities": len(common_ids),
                "note": "fewer than 3 identities common to all regions."}

    matrix = np.array([[r.per_identity[i] for i in common_ids] for r in region_results])
    stat, p = stats.friedmanchisquare(*[matrix[i] for i in range(matrix.shape[0])])
    all_vals = [np.array(list(r.per_identity.values())) for r in region_results]
    kw_stat, kw_p = stats.kruskal(*all_vals)

    return {
        "friedman_stat": float(stat), "friedman_p": float(p),
        "n_matched_identities": len(common_ids),
        "kruskal_stat_reference_only": float(kw_stat),
        "kruskal_p_reference_only": float(kw_p),
    }

@dataclass
class ReductionResult:
    region_id: int
    n_paired_identities: int
    before_mean: float
    after_mean: float
    abs_reduction: float
    relative_reduction_pct: float
    mean_delta: float
    delta_ci_lo: float
    delta_ci_hi: float
    wilcoxon_stat: float
    wilcoxon_p: float   # paired, one-sided: is delta > 0 (reduction)?


def analyze_reduction(before: RegionResult, after: RegionResult, region_id: int,
                       n_boot: int = 2000, seed: Optional[int] = 0) -> ReductionResult:
    common_ids = sorted(set(before.per_identity.keys()) & set(after.per_identity.keys()))
    deltas = np.array([before.per_identity[i] - after.per_identity[i] for i in common_ids])

    mean_delta, lo, hi = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)

    if len(deltas) >= 3 and np.any(deltas != 0):
        try:
            stat, p_two_sided = stats.wilcoxon(deltas, alternative="two-sided")
            p_one_sided = p_two_sided / 2 if mean_delta > 0 else 1.0 - p_two_sided / 2
        except ValueError:
            stat, p_one_sided = float("nan"), float("nan")
    else:
        stat, p_one_sided = float("nan"), float("nan")

    abs_reduction = before.centroid_mean - after.centroid_mean
    rel_reduction = (abs_reduction / before.centroid_mean * 100
                      if before.centroid_mean > 1e-9 else float("nan"))

    return ReductionResult(
        region_id=region_id, n_paired_identities=len(common_ids),
        before_mean=before.centroid_mean, after_mean=after.centroid_mean,
        abs_reduction=abs_reduction, relative_reduction_pct=rel_reduction,
        mean_delta=mean_delta, delta_ci_lo=lo, delta_ci_hi=hi,
        wilcoxon_stat=stat, wilcoxon_p=p_one_sided,
    )

def derive_modality_labels(cam_ids):
    """Mirrors model.py's train_forward RGB/IR camera convention exactly
    (SYSU-MM01: RGB cams {1,2,4,5}, IR cams {3,6})."""
    import torch
    rgb_mask = (cam_ids == 1) | (cam_ids == 2) | (cam_ids == 4) | (cam_ids == 5)
    ir_mask = (cam_ids == 3) | (cam_ids == 6)
    modality_labels = torch.zeros(len(cam_ids), dtype=torch.long, device=cam_ids.device)
    modality_labels[ir_mask] = 1
    assert bool((rgb_mask | ir_mask).all()), (
        "Found cam_ids outside the expected SYSU-MM01 RGB={1,2,4,5}/IR={3,6} "
        "convention -- check the dataset/camera mapping before trusting the split."
    )
    return modality_labels


def build_minimal_backbone(pretrained_imagenet: bool = False):
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
    backbone = resnet50(weights=weights)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    container = nn.Module()
    container.conv1 = backbone.conv1
    container.bn1 = backbone.bn1
    container.relu = backbone.relu
    container.maxpool = backbone.maxpool
    container.layer1 = backbone.layer1
    container.layer2 = backbone.layer2
    container.layer3 = backbone.layer3
    container.layer4 = backbone.layer4
    return container


def _backbone_forward(backbone, x):
    b = backbone
    x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
    x = b.layer1(x); x = b.layer2(x); x = b.layer3(x)
    return b.layer4(x)


def build_pmr_head(feat_dim: int, num_parts: int):
    """For --pmr_resume (With-PMR checkpoint). Minimal reconstruction of the
    exact submodules PMR's eval-mode forward needs: part_bns (BatchNorm1d
    per part -> model.py's 'reduced_parts') and PartModalityRemover.
    linear_projection (Linear + LayerNorm per part -- its eval-mode forward
    is literally just this, see part_modality_remover.py's `else:` branch).
    Deliberately excludes modality_classifier / recon_heads: those only
    produce training losses and are never touched by eval-mode forward."""
    import torch.nn as nn
    head = nn.Module()
    head.part_bns = nn.ModuleList([nn.BatchNorm1d(feat_dim) for _ in range(num_parts)])
    head.linear_projection = nn.ModuleList([
        nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim))
        for _ in range(num_parts)
    ])
    return head


def build_part_bns(feat_dim: int, num_parts: int):
    """For --no_pmr_resume (part branch present, PMR=False). Just the
    per-part BatchNorm1d -- there is no PartModalityRemover to reconstruct
    in this checkpoint at all, which is the point."""
    import torch.nn as nn
    return nn.ModuleList([nn.BatchNorm1d(feat_dim) for _ in range(num_parts)])


def _extract_raw_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        sample_values = list(checkpoint.values())[:5]
        if sample_values and all(hasattr(v, "shape") for v in sample_values):
            return checkpoint
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_backbone_weights(backbone, checkpoint_path: str, tag: str = ""):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_raw_state_dict(checkpoint)

    prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
    filtered = {}
    for k, v in state_dict.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith(prefixes):
            filtered[kk] = v

    if len(filtered) == 0:
        sample_keys = list(state_dict.keys())[:20]
        raise ValueError(
            f"[{tag}] No backbone-prefixed tensors (conv1./bn1./layer1../layer4.) "
            f"found in {checkpoint_path}. First 20 keys actually present: {sample_keys}"
        )

    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    print(f"[{tag}] backbone: loaded {len(filtered)} tensors from {checkpoint_path} "
          f"({len(missing)} missing, {len(unexpected)} unexpected).")
    if missing:
        print(f"[{tag}] WARNING -- missing backbone tensors (using random/ImageNet init "
              f"for these instead): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return checkpoint


def load_pmr_head_weights(head, checkpoint_or_path):
    import torch
    checkpoint = (torch.load(checkpoint_or_path, map_location="cpu")
                  if isinstance(checkpoint_or_path, str) else checkpoint_or_path)
    state_dict = _extract_raw_state_dict(checkpoint)

    remapped = {}
    for k, v in state_dict.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith("part_bns."):
            remapped[kk] = v
        elif kk.startswith("part_modality_purifier.linear_projection."):
            remapped[kk[len("part_modality_purifier."):]] = v

    if len(remapped) == 0:
        raise ValueError(
            "No part_bns.* or part_modality_purifier.linear_projection.* tensors found "
            "in --pmr_resume -- was this checkpoint trained with PMR=True and parts=True? "
            f"First 20 keys present: {list(state_dict.keys())[:20]}"
        )
    missing, unexpected = head.load_state_dict(remapped, strict=False)
    print(f"[pmr] head: loaded {len(remapped)} tensors ({len(missing)} missing, "
          f"{len(unexpected)} unexpected).")
    if missing:
        raise ValueError(
            f"Missing required PMR head tensors: {missing}. This checkpoint may not have "
            "been trained with PMR enabled, or --num_parts doesn't match the checkpoint's "
            "actual training config (SYSU.yml default is 6)."
        )

def load_part_bns_weights(part_bns, checkpoint_path: str):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_raw_state_dict(checkpoint)

    remapped = {}
    for k, v in state_dict.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith("part_bns."):
            remapped[kk[len("part_bns."):]] = v   # <-- strip the prefix here

    if len(remapped) == 0:
        raise ValueError(
            f"No part_bns.* tensors found in --no_pmr_resume ({checkpoint_path}) -- was "
            "this checkpoint trained with parts=True (PMR=False)? "
            f"First 20 keys present: {list(state_dict.keys())[:20]}"
        )
    missing, unexpected = part_bns.load_state_dict(remapped, strict=False)
    print(f"[no_pmr] part_bns: loaded {len(remapped)} tensors ({len(missing)} missing, "
          f"{len(unexpected)} unexpected).")
    if missing:
        raise ValueError(
            f"Missing required part_bns tensors: {missing}. --num_parts may not match "
            "this checkpoint's actual training config (SYSU.yml default is 6)."
        )
    return checkpoint

class RegionalGridExtractor:
    """Fine, generic grid_rows x grid_cols grid over the layer4 spatial
    map, GeM-pooled per cell. Deliberately NOT tied to num_parts, so
    heterogeneity measured with this extractor cannot be accused of
    assuming the stripe partition it's meant to interrogate (R2)."""

    def __init__(self, backbone, grid_rows: int = 12, grid_cols: int = 4, p: float = 3.0):
        self.backbone = backbone
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.p = p

    def _gem_pool_cell(self, cell):
        import torch
        cell = cell.clamp(min=1e-12)
        return torch.mean(cell.pow(self.p), dim=(2, 3)).pow(1.0 / self.p)

    def __call__(self, images):
        import torch
        fmap = _backbone_forward(self.backbone, images)
        row_strips = torch.chunk(fmap, self.grid_rows, dim=2)
        region_feats = []
        for row in row_strips:
            for cell in torch.chunk(row, self.grid_cols, dim=3):
                region_feats.append(self._gem_pool_cell(cell))
        return torch.stack(region_feats, dim=1)   # B x (rows*cols) x C


class NativeStripeExtractor:
    """The model's ACTUAL native num_parts horizontal stripes, single GeM
    pool per full-width stripe -- byte-for-byte the same computation as
    Part_Features.forward (part_features.py): torch.chunk(x, num_parts,
    dim=2), then mean(strip.clamp(min=eps).pow(p), dim=(2,3)).pow(1/p).
    Used to get a faithful 'raw, pre-BN' stripe feature from EITHER
    backbone (baseline or PMR), so Step 2's before/after/baseline
    comparison is apples-to-apples with what model.py actually computes."""

    def __init__(self, backbone, num_parts: int = 6, p: float = 3.0):
        self.backbone = backbone
        self.num_parts = num_parts
        self.p = p

    def _gem_pool_stripe(self, strip):
        import torch
        strip = strip.clamp(min=1e-12)
        return torch.mean(strip.pow(self.p), dim=(2, 3)).pow(1.0 / self.p)

    def __call__(self, images):
        import torch
        fmap = _backbone_forward(self.backbone, images)
        strips = torch.chunk(fmap, self.num_parts, dim=2)
        return torch.stack([self._gem_pool_stripe(s) for s in strips], dim=1)  # B x P x C


def pmr_head_forward(head, raw_stripe_feats, num_parts: int):
    """For --pmr_resume. raw_stripe_feats: B x P x C (from
    NativeStripeExtractor on the With-PMR checkpoint's backbone). Returns
    the post-projector purified feature, B x P x C, matching model.py's
    fuse_features/train_forward computation graph: pre = part_bns(raw);
    purified = linear_projection(pre) (eval-mode PartModalityRemover, see
    part_modality_remover.py's `else:` branch). This is the feature the
    proposed model actually uses at inference -- the "with_pmr" point in
    Step 2."""
    import torch
    pre = torch.stack(
        [head.part_bns[i](raw_stripe_feats[:, i]) for i in range(num_parts)], dim=1
    )
    purified = torch.stack(
        [head.linear_projection[i](pre[:, i]) for i in range(num_parts)], dim=1
    )
    return purified


def part_bns_forward(part_bns, raw_stripe_feats, num_parts: int):
    """For --no_pmr_resume. raw_stripe_feats: B x P x C (from
    NativeStripeExtractor on the No-PMR checkpoint's backbone). Returns the
    GeM+BN part feature, B x P x C -- the "no_pmr" point in Step 2, i.e.
    exactly the feature Fig. 4(a)'s "Without PMR" row is computed from."""
    import torch
    return torch.stack(
        [part_bns[i](raw_stripe_feats[:, i]) for i in range(num_parts)], dim=1
    )


def _unpack_batch(batch):
    """Confirmed against data/__init__.py collate_fn + dataset.py
    __getitem__: a batch is [images, labels, cams, paths, items]."""
    if isinstance(batch, dict):
        images = batch.get("img", batch.get("image"))
        ids = batch.get("id", batch.get("pid"))
        cam_ids = batch.get("cam_id", batch.get("camid"))
        if images is None or ids is None or cam_ids is None:
            raise ValueError(f"Batch dict has keys {list(batch.keys())}; edit _unpack_batch().")
        return images, ids, cam_ids
    if not isinstance(batch, (list, tuple)) or len(batch) < 3:
        raise ValueError(f"Unrecognized batch structure; edit _unpack_batch().")
    return batch[0], batch[1], batch[2]


def _set_seed(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_feature_extraction(loaders, extract_fn, device, max_batches=None, tag=""):
    """Single aligned pass over the loader(s): calls extract_fn(images) ->
    dict[str, Tensor] (each B x R x C) per batch, and accumulates ids /
    modality alongside. Using ONE pass for baseline + PMR features together
    (rather than two separate passes) guarantees index-for-index alignment
    between them -- required for the paired per-identity reduction stats."""
    import torch
    collected = None
    all_ids, all_modality = [], []
    with torch.no_grad():
        for split_name, loader in loaders:
            n_batches = 0
            for batch_idx, batch in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                images, ids, cam_ids = _unpack_batch(batch)
                images = images.to(device)
                cam_ids = cam_ids.to(device) if torch.is_tensor(cam_ids) else torch.as_tensor(cam_ids, device=device)
                out = extract_fn(images)
                if collected is None:
                    collected = {k: [] for k in out}
                for k, v in out.items():
                    collected[k].append(v.cpu().numpy())
                modality = derive_modality_labels(cam_ids)
                all_ids.append(ids.cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids))
                all_modality.append(modality.cpu().numpy())
                n_batches += 1
            print(f"[{tag}] extracted {n_batches} batches from {split_name}_loader")
    result = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    result["ids"] = np.concatenate(all_ids, axis=0)
    result["modality"] = np.concatenate(all_modality, axis=0)
    return result


# ==========================================================================
# 4. VISUALIZATION
# ==========================================================================

def plot_grid_heatmap(region_results: List[RegionResult], grid_rows: int, grid_cols: int,
                       out_path: str):
    """Fig. 5(b). Drawn on the same fixed-height canvas as
    plot_three_way_bars (Fig. 5(a)), so both images have identical height and
    aligned plotting areas. Cells are square. No title is drawn -- caption
    this figure from the LaTeX \\caption when placing it."""
    values = np.array([r.centroid_mean for r in region_results]).reshape(grid_rows, grid_cols)
    cell = FIG5_AXES_HEIGHT_IN / grid_rows
    fig, ax = _new_fig5_canvas(axes_width_in=cell * grid_cols)
    im = ax.imshow(values, cmap="inferno", aspect="auto", interpolation="nearest")

    # Row ticks: thin them out automatically if the font is too big for the cells.
    tick_in = 1.2 * _pts("ytick.labelsize") / 72.0
    step = max(1, int(np.ceil(tick_in / cell)))
    ax.set_yticks(range(0, grid_rows, step)); ax.set_xticks(range(grid_cols))
    ax.tick_params(length=3)
    ax.set_xlabel("column (left $\\rightarrow$ right)")
    ylab = ax.set_ylabel("row (head $\\rightarrow$ feet)")

    # Colorbar placed by hand so it has exactly the axes' height.
    pos = ax.get_position()
    W, H = fig.get_size_inches()
    cax = fig.add_axes([pos.x1 + 0.12 / W, pos.y0, 0.20 / W, pos.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=_pts("ytick.labelsize"))
    cbar.set_label("Mean Cross-Modal Discrepancy", fontsize=_pts("axes.labelsize"))
    cbar.locator = matplotlib.ticker.MaxNLocator(5); cbar.update_ticks()

    _wrap_if_too_long(fig, ylab, FIG5_AXES_HEIGHT_IN)
    _wrap_if_too_long(fig, cbar.ax.yaxis.label, FIG5_AXES_HEIGHT_IN)
    _save_fixed_height(fig, out_path)
    return out_path


def plot_region_bar_with_ci(region_results: List[RegionResult], out_path: str,
                             labels: Sequence[str] = None):
    n = len(region_results)
    means = [r.centroid_mean for r in region_results]
    los = [r.centroid_mean - r.centroid_ci_lo for r in region_results]
    his = [r.centroid_ci_hi - r.centroid_mean for r in region_results]
    labels = labels if labels is not None else [f"R{r.region_id}" for r in region_results]
    fig, ax = plt.subplots(figsize=(max(4.5, n * 0.75), 4.5))
    ax.bar(range(n), means, yerr=[los, his], capsize=4, color="#c0392b",
           edgecolor="black", linewidth=1.0, error_kw={"elinewidth": 1.6, "capthick": 1.6})
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Cross-Modal Discrepancy")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_pipeline_bars(no_pmr_results, with_pmr_results,
                        reduction: List["ReductionResult"], out_path: str,
                        part_labels: Sequence[str] = None):
    """The headline Step-2 figure: No PMR vs. With PMR per part (the
    checkpoint pair also used for Fig. 4(a)'s retrieval-metric ablation).
    No title is drawn; no reduction-% text is overlaid on the bars either
    -- report the reduction_by_part numbers (summary_step2.json /
    reduction_no_pmr_vs_with_pmr.csv) in the caption or body text instead,
    so the figure itself stays legible at column width."""
    n = len(no_pmr_results)
    labels = part_labels if part_labels is not None else [f"part {i+1}" for i in range(n)]
    no_pmr = np.array([r.centroid_mean for r in no_pmr_results])
    with_pmr = np.array([r.centroid_mean for r in with_pmr_results])

    x = np.arange(n); width = 0.36
    fig, ax1 = plt.subplots(figsize=(max(6.5, n * 1.3), 4.8))
    ax1.bar(x - width / 2, no_pmr, width, label="No PMR", color="#5B6770",
            edgecolor="black", linewidth=1.0)
    ax1.bar(x + width / 2, with_pmr, width, label="With PMR", color="#CC4125",
            edgecolor="black", linewidth=1.0)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("Mean Cross-Modal Discrepancy")
    ax1.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_three_way_bars(baseline_results, no_pmr_results, with_pmr_results,
                         out_path: str, part_labels: Sequence[str] = None,
                         sig_holm: Optional[Sequence[bool]] = None,
                         watermark: Optional[str] = None):
    """THE single figure for R1-C2 and R2-C7: grouped bars, one group per
    horizontal stripe, three bars per group -- Baseline | No PMR | PMR.

      * Variation ACROSS stripes within any one colour  -> spatial
        heterogeneity (R1-C2). The Baseline bars are the assumption-free
        evidence: that network never saw a part branch or any stripe loss.
      * Variation ACROSS the three bars within a stripe -> effect of PMR
        (R2-C7). ONLY the No PMR -> PMR step isolates PMR; Baseline -> No PMR
        bundles TMS/MAR/PGFA/part-branch, so do not attribute it to PMR.

    Error bars are 95% bootstrap CIs of the per-identity mean. They are NOT
    the right uncertainty for a paired difference (state this in the
    caption); paired statistics live in reduction_no_pmr_vs_with_pmr.csv.
    Optional: if `sig_holm` is passed, '*' is drawn above the PMR bar of each
    part whose No PMR -> PMR reduction is significant after Holm-Bonferroni
    correction (one-sided paired Wilcoxon). Default call site passes None, so
    no asterisks appear; significance is reported in the CSV/JSON and text.
    No title is drawn (caption belongs in LaTeX). `watermark` is used by
    --self_test so a synthetic preview can never be mistaken for a result."""
    n = len(no_pmr_results)
    labels = part_labels if part_labels is not None else [f"S{i+1}" for i in range(n)]
    series = [
        ("Baseline", baseline_results, "#D5D9DD", ""),
        ("No PMR",   no_pmr_results,   "#5B6770", "//"),
        ("PMR",      with_pmr_results, "#CC4125", ""),
    ]
    x = np.arange(n); width = 0.26
    # Fixed-height canvas shared with plot_grid_heatmap (Fig. 5(b)).
    # Each stripe group gets at least room for its tick label at this font size.
    group_in = max(1.0, 3.0 * _pts("xtick.labelsize") / 72.0)
    fig, ax = _new_fig5_canvas(axes_width_in=n * group_in)
    tops = np.zeros(n)
    for k, (name, res, color, hatch) in enumerate(series):
        m = np.array([r.centroid_mean for r in res])
        lo = m - np.array([r.centroid_ci_lo for r in res])
        hi = np.array([r.centroid_ci_hi for r in res]) - m
        ax.bar(x + (k - 1) * width, m, width, yerr=[lo, hi], capsize=3, label=name,
               color=color, edgecolor="black", linewidth=1.0, hatch=hatch,
               error_kw={"elinewidth": 1.3, "capthick": 1.3})
        if name == "PMR":
            tops = m + hi
    if sig_holm is not None:
        ymax = max(float(tops.max()), 1e-9)
        for i, flag in enumerate(sig_holm):
            if flag:
                ax.text(x[i] + width, tops[i] + 0.02 * ymax, "*", ha="center",
                        va="bottom", fontsize=1.06 * _pts("font.size"), fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xlabel("Horizontal stripe (top $\\rightarrow$ bottom)")
    ylab = ax.set_ylabel("Mean Cross-Modal Discrepancy")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              columnspacing=1.6, handlelength=1.6, borderaxespad=0.3)
    if watermark:
        ax.text(0.5, 0.5, watermark, transform=ax.transAxes, ha="center", va="center",
                fontsize=1.3 * _pts("font.size"), color="red", alpha=0.35, rotation=20,
                fontweight="bold", gid="watermark")
    _wrap_if_too_long(fig, ylab, FIG5_AXES_HEIGHT_IN)
    _save_fixed_height(fig, out_path)
    return out_path


def plot_correlation_scatter(x_vals, y_vals, x_label: str, y_label: str,
                              out_path: str, point_labels: Sequence[str] = None):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(x_vals, y_vals, s=70, color="#8e44ad", edgecolor="black",
               linewidth=1.0, zorder=3)
    if point_labels is not None:
        for x, y, lab in zip(x_vals, y_vals, point_labels):
            ax.annotate(lab, (x, y), fontsize=_pts("xtick.labelsize") - 2, xytext=(5, 5), textcoords="offset points")
    if len(x_vals) >= 2:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        xs = np.linspace(min(x_vals), max(x_vals), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="#8e44ad", alpha=0.5, linestyle="--")
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


# ==========================================================================
# 5. STEP 1 -- spatial heterogeneity, independent of the stripe partition
#    (-> reviewer comments R1, R2)
# ==========================================================================

def run_step1_heterogeneity(fine_feats: np.ndarray, ids: np.ndarray, modality: np.ndarray,
                             grid_rows: int, grid_cols: int, num_parts: int, out_dir: str,
                             n_boot: int, n_perm: int, seed: int) -> dict:
    """fine_feats: (N, grid_rows*grid_cols, D), from the BASELINE (no-PMR,
    no-part-branch) backbone via RegionalGridExtractor. This never touches
    the model's own stripe partition, so it directly answers R1 ('validate
    the core spatial-heterogeneity assumption'). We additionally aggregate
    the same fine grid into num_parts bands and compare within- vs.
    between-stripe variance -- this is the direct, quantitative answer to
    R2's 'while the part branch relies on fixed horizontal stripes' jab:
    it tells you whether stripes are actually a defensible way to carve up
    the measured heterogeneity."""
    step_dir = os.path.join(out_dir, "step1_heterogeneity")
    os.makedirs(step_dir, exist_ok=True)
    os.makedirs(os.path.join(step_dir, "raw"), exist_ok=True)
    num_regions = fine_feats.shape[1]
    assert num_regions == grid_rows * grid_cols

    n_rgb, n_ir, n_ids = int((modality == 0).sum()), int((modality == 1).sum()), len(np.unique(ids))
    print(f"[step1] {len(fine_feats)} samples ({n_rgb} RGB / {n_ir} IR), {n_ids} identities, "
          f"{num_regions} fine regions ({grid_rows}x{grid_cols}).")

    fine_results = [analyze_region(RegionSamples(fine_feats[:, r, :], ids, modality), r,
                                    n_boot, n_perm, seed) for r in range(num_regions)]
    het_fine = heterogeneity_test(fine_results)

    def _write_csv(path, results):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region_id", "n_identities", "centroid_mean", "centroid_ci_lo",
                        "centroid_ci_hi", "mmd2", "mmd2_pvalue"])
            for r in results:
                w.writerow([r.region_id, r.n_identities, r.centroid_mean, r.centroid_ci_lo,
                            r.centroid_ci_hi, r.mmd2, r.mmd2_pvalue])

    _write_csv(os.path.join(step_dir, "region_results_fine_grid.csv"), fine_results)
    with open(os.path.join(step_dir, "heterogeneity_test_fine_grid.json"), "w") as f:
        json.dump(het_fine, f, indent=2)
    plot_grid_heatmap(fine_results, grid_rows, grid_cols,
                       out_path=os.path.join(step_dir, "heatmap_fine_grid.png"))

    assert grid_rows % num_parts == 0, (
        f"--grid_rows={grid_rows} must be a multiple of --num_parts={num_parts} "
        "for a clean stripe aggregation."
    )
    rows_per_stripe = grid_rows // num_parts
    stripe_src = fine_feats.reshape(len(fine_feats), grid_rows, grid_cols, -1)
    stripe_agg_results = []
    for s in range(num_parts):
        band = stripe_src[:, s * rows_per_stripe:(s + 1) * rows_per_stripe, :, :].mean(axis=(1, 2))
        stripe_agg_results.append(analyze_region(RegionSamples(band, ids, modality), s,
                                                   n_boot, n_perm, seed))
    het_stripes = heterogeneity_test(stripe_agg_results)
    stripe_labels = [f"stripe {i+1}" for i in range(num_parts)]
    _write_csv(os.path.join(step_dir, "region_results_stripe_aggregated.csv"), stripe_agg_results)
    with open(os.path.join(step_dir, "heterogeneity_test_stripe_aggregated.json"), "w") as f:
        json.dump(het_stripes, f, indent=2)
    plot_region_bar_with_ci(stripe_agg_results,
                             out_path=os.path.join(step_dir, "bar_stripe_aggregated.png"),
                             labels=stripe_labels)

    np.savez(os.path.join(step_dir, "raw", "step1_raw.npz"),
              fine_feats=fine_feats, ids=ids, modality=modality,
              fine_centroid_means=np.array([r.centroid_mean for r in fine_results]))

    fine_vals = np.array([r.centroid_mean for r in fine_results]).reshape(grid_rows, grid_cols)
    flat_vals = fine_vals.flatten()
    effect_size = {
        "max_over_min_ratio": float(flat_vals.max() / max(flat_vals.min(), 1e-9)),
        "coefficient_of_variation": float(flat_vals.std() / max(flat_vals.mean(), 1e-9)),
        "note": ("Report alongside the Friedman p-value: with many fine regions, p<0.05 "
                 "alone can arise from pure sampling noise. A small ratio (<~1.2x) and low "
                 "CV mean statistically-detectable but practically negligible heterogeneity."),
    }
    within_stripe_var = [float(fine_vals[s * rows_per_stripe:(s + 1) * rows_per_stripe, :].var())
                          for s in range(num_parts)]
    stripe_vals = np.array([r.centroid_mean for r in stripe_agg_results])
    alignment = {
        "mean_within_stripe_variance": float(np.mean(within_stripe_var)),
        "between_stripe_variance": float(stripe_vals.var()),
        "interpretation": ("If between_stripe_variance >> mean_within_stripe_variance, the "
                            "fixed horizontal-stripe partition captures most of the measured "
                            "heterogeneity well (a defensible design choice). If comparable or "
                            "larger, substantial heterogeneity exists WITHIN stripes that the "
                            "fixed partition cannot resolve -- worth disclosing as a limitation "
                            "or motivating a finer/adaptive partition in future work."),
    }

    summary = {
        "n_samples": len(fine_feats), "n_rgb": n_rgb, "n_ir": n_ir, "n_identities": n_ids,
        "fine_grid": {"rows": grid_rows, "cols": grid_cols, "heterogeneity_test": het_fine,
                      "effect_size": effect_size,
                      "region_range": [float(flat_vals.min()), float(flat_vals.max())]},
        "stripe_aggregated": {"num_parts": num_parts, "heterogeneity_test": het_stripes,
                               "values": [r.centroid_mean for r in stripe_agg_results]},
        "alignment_diagnostic": alignment,
    }
    with open(os.path.join(step_dir, "summary_step1.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[step1] Friedman p (fine grid) = {het_fine['friedman_p']:.3e}, "
          f"effect ratio = {effect_size['max_over_min_ratio']:.2f}x, "
          f"CV = {effect_size['coefficient_of_variation']:.3f}")
    return {"fine_results": fine_results, "stripe_agg_results": stripe_agg_results,
            "summary": summary}


# ==========================================================================
# 6. STEP 2 -- region-wise discrepancy, No PMR vs. With PMR
#    (-> reviewer comment R3: "a region-wise analysis before and after
#    PMR"). Two points, two independently trained checkpoints that share
#    everything except PMR -- see the design note at the top of this file
#    for why this replaced an earlier three-point (baseline/before/after)
#    version.
# ==========================================================================

def run_step2_reduction(no_pmr_feats: np.ndarray, with_pmr_feats: np.ndarray,
                         ids: np.ndarray, modality: np.ndarray,
                         num_parts: int, out_dir: str, n_boot: int, n_perm: int, seed: int,
                         baseline_feats: Optional[np.ndarray] = None,
                         watermark: Optional[str] = None) -> dict:
    """no_pmr_feats:   (N,P,D) from --no_pmr_resume, part_bns output. Same
                        checkpoint/feature as the paper's Fig. 4(a) "Without
                        PMR" setting.
    with_pmr_feats:    (N,P,D) from --pmr_resume, PartModalityRemover.
                        linear_projection output. Same checkpoint/feature as
                        Fig. 4(a)'s "With PMR" setting -- what the final
                        model actually uses at inference.
    Both MUST be index-aligned (same images, same ids/modality) -- if they
    come from two separate checkpoints, extract them with the same
    dataloader order/seed (run_from_project does this by construction, one
    extract_fn call per batch covering both backbones).

    baseline_feats (optional, N,P,D): the PLAIN backbone (--plain_resume,
    Table III row 1) pooled with the IDENTICAL NativeStripeExtractor. When
    given, the headline figure becomes the three-bar Baseline | No PMR | PMR
    plot that answers R1-C2 (heterogeneity, from the Baseline bars) and R2-C7
    (before/after PMR, from the No PMR -> PMR step) in one figure.
    Only No PMR -> PMR is the PMR-attributable contrast and the only one
    that enters the Holm family; Baseline -> No PMR / Baseline -> PMR are
    descriptive (they bundle TMS, MAR, PGFA and the part branch)."""
    step_dir = os.path.join(out_dir, "step2_reduction")
    os.makedirs(step_dir, exist_ok=True)
    os.makedirs(os.path.join(step_dir, "raw"), exist_ok=True)
    part_labels = [f"part {i+1}" for i in range(num_parts)]

    no_pmr_results = [analyze_region(RegionSamples(no_pmr_feats[:, i, :], ids, modality),
                                      i, n_boot, n_perm, seed) for i in range(num_parts)]
    with_pmr_results = [analyze_region(RegionSamples(with_pmr_feats[:, i, :], ids, modality),
                                        i, n_boot, n_perm, seed) for i in range(num_parts)]

    baseline_results, het_baseline = None, None
    if baseline_feats is not None:
        baseline_results = [analyze_region(RegionSamples(baseline_feats[:, i, :], ids, modality),
                                            i, n_boot, n_perm, seed) for i in range(num_parts)]
        het_baseline = heterogeneity_test(baseline_results)

    # Diagnostic: compare with/without the per-channel standardization fix,
    # on the No-PMR features specifically (raw post-ReLU + plain BN -- the
    # stage most exposed to the channel-dominance / positive-orthant
    # artifact _standardize_channels corrects for). If these two rows look
    # wildly different, the fix was doing real work on your checkpoints;
    # if they're close, standardization wasn't changing much here.
    no_pmr_nostd = [analyze_region(RegionSamples(no_pmr_feats[:, i, :], ids, modality),
                                    i, n_boot, n_perm, seed, standardize=False) for i in range(num_parts)]
    print(f"[step2] diagnostic -- No-PMR discrepancy per part, WITHOUT standardization fix: "
          f"{[f'{r.centroid_mean:.4f}' for r in no_pmr_nostd]}")
    print(f"[step2] diagnostic -- No-PMR discrepancy per part, WITH standardization fix:    "
          f"{[f'{r.centroid_mean:.4f}' for r in no_pmr_results]}")

    het_no_pmr = heterogeneity_test(no_pmr_results)
    het_with_pmr = heterogeneity_test(with_pmr_results)

    def _write_region_csv(path, results):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region_id", "n_identities", "centroid_mean", "centroid_ci_lo",
                        "centroid_ci_hi", "mmd2", "mmd2_pvalue", "inter_id_dist", "relative_gap"])
            for r in results:
                w.writerow([r.region_id, r.n_identities, r.centroid_mean, r.centroid_ci_lo,
                            r.centroid_ci_hi, r.mmd2, r.mmd2_pvalue, r.inter_id_dist,
                            r.relative_gap])

    if baseline_results is not None:
        _write_region_csv(os.path.join(step_dir, "region_results_baseline.csv"), baseline_results)
        with open(os.path.join(step_dir, "heterogeneity_test_baseline.json"), "w") as f:
            json.dump(het_baseline, f, indent=2)
    _write_region_csv(os.path.join(step_dir, "region_results_no_PMR.csv"), no_pmr_results)
    _write_region_csv(os.path.join(step_dir, "region_results_with_PMR.csv"), with_pmr_results)
    for name, het in (("no_pmr", het_no_pmr), ("with_pmr", het_with_pmr)):
        with open(os.path.join(step_dir, f"heterogeneity_test_{name}.json"), "w") as f:
            json.dump(het, f, indent=2)

    def _write_reduction_csv(path, results, before_name="no_pmr", after_name="with_pmr"):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region_id", "n_paired_identities", f"{before_name}_mean", f"{after_name}_mean",
                        "abs_reduction", "relative_reduction_pct", "mean_delta",
                        "delta_ci_lo", "delta_ci_hi", "wilcoxon_stat", "wilcoxon_p_one_sided"])
            for r in results:
                w.writerow([r.region_id, r.n_paired_identities, r.before_mean, r.after_mean,
                            r.abs_reduction, r.relative_reduction_pct, r.mean_delta,
                            r.delta_ci_lo, r.delta_ci_hi, r.wilcoxon_stat, r.wilcoxon_p])

    # ---- the single paired comparison: No PMR -> With PMR ----
    reduction = [analyze_reduction(no_pmr_results[i], with_pmr_results[i], i, n_boot, seed)
                 for i in range(num_parts)]

    _write_reduction_csv(os.path.join(step_dir, "reduction_no_pmr_vs_with_pmr.csv"), reduction)

    # Descriptive (NOT PMR-attributable) contrasts against the plain baseline.
    reduction_baseline_to_no_pmr = reduction_baseline_to_pmr = None
    if baseline_results is not None:
        reduction_baseline_to_no_pmr = [analyze_reduction(baseline_results[i], no_pmr_results[i],
                                                           i, n_boot, seed) for i in range(num_parts)]
        reduction_baseline_to_pmr = [analyze_reduction(baseline_results[i], with_pmr_results[i],
                                                        i, n_boot, seed) for i in range(num_parts)]
        _write_reduction_csv(os.path.join(step_dir, "reduction_baseline_vs_no_pmr.csv"),
                              reduction_baseline_to_no_pmr, "baseline", "no_pmr")
        _write_reduction_csv(os.path.join(step_dir, "reduction_baseline_vs_with_pmr.csv"),
                              reduction_baseline_to_pmr, "baseline", "with_pmr")

    # multiple-comparisons correction (Holm-Bonferroni) on the headline test
    def _holm_flags(pvals, alpha=0.05):
        order = np.argsort(pvals)
        m = len(pvals)
        flags = [False] * m
        for rank, idx in enumerate(order):
            p = pvals[idx]
            if np.isnan(p):
                continue
            threshold = alpha / (m - rank)
            if p <= threshold:
                flags[idx] = True
            else:
                break
        return flags

    pvals = np.array([r.wilcoxon_p for r in reduction])
    sig_holm = _holm_flags(pvals)

    if baseline_results is not None:
        plot_three_way_bars(baseline_results, no_pmr_results, with_pmr_results,
                             os.path.join(step_dir, "bar_baseline_no_pmr_pmr.png"),
                             part_labels=[f"S{i+1}" for i in range(num_parts)],
                             sig_holm=None,  # no '*' drawn on the figure; Holm results stay in CSV/JSON
                             watermark=watermark)
    # legacy two-bar figure kept for backward compatibility
    plot_pipeline_bars(no_pmr_results, with_pmr_results, reduction,
                        os.path.join(step_dir, "bar_no_pmr_vs_with_pmr.png"), part_labels)

    np.savez(os.path.join(step_dir, "raw", "step2_raw.npz"),
              no_pmr_feats=no_pmr_feats, with_pmr_feats=with_pmr_feats,
              ids=ids, modality=modality)

    overall_no_pmr = float(np.mean([r.centroid_mean for r in no_pmr_results]))
    overall_with_pmr = float(np.mean([r.centroid_mean for r in with_pmr_results]))
    n_parts_improved = sum(1 for r in reduction if r.abs_reduction > 0)
    n_parts_significant_raw = sum(1 for r in reduction
                                   if not np.isnan(r.wilcoxon_p) and r.wilcoxon_p < 0.05)
    n_parts_significant_holm = sum(sig_holm)

    summary = {
        "n_samples": len(ids), "num_parts": num_parts,
        "overall_no_pmr_mean": overall_no_pmr, "overall_with_pmr_mean": overall_with_pmr,
        "overall_reduction_pct": (
            (overall_no_pmr - overall_with_pmr) / overall_no_pmr * 100
            if overall_no_pmr > 1e-9 else float("nan")
        ),
        "n_parts_with_positive_reduction": n_parts_improved,
        "n_parts_significant_p<0.05_raw": n_parts_significant_raw,
        "n_parts_significant_holm_bonferroni": n_parts_significant_holm,
        "heterogeneity_no_pmr": het_no_pmr, "heterogeneity_with_pmr": het_with_pmr,
        "reduction_by_part": [
            {"part": i + 1,
             "no_pmr": no_pmr_results[i].centroid_mean,
             "with_pmr": with_pmr_results[i].centroid_mean,
             "reduction_pct": reduction[i].relative_reduction_pct,
             "wilcoxon_p_raw": reduction[i].wilcoxon_p,
             "significant_holm": sig_holm[i]}
            for i in range(num_parts)
        ],
    }
    # Robustness: does the ordering survive identity-normalisation, and does the
    # PMR-vs-No-PMR conclusion depend on the cross-space cosine scale?
    summary["relative_gap_by_part"] = [
        {"part": i + 1,
         "no_pmr": no_pmr_results[i].relative_gap,
         "with_pmr": with_pmr_results[i].relative_gap,
         **({"baseline": baseline_results[i].relative_gap} if baseline_results else {})}
        for i in range(num_parts)
    ]
    summary["n_parts_pmr_lower_under_relative_gap"] = int(sum(
        1 for i in range(num_parts)
        if with_pmr_results[i].relative_gap < no_pmr_results[i].relative_gap))
    if baseline_results is not None:
        flat = np.array([r.centroid_mean for r in baseline_results])
        summary["baseline_by_part"] = [r.centroid_mean for r in baseline_results]
        summary["baseline_heterogeneity"] = het_baseline
        summary["baseline_effect_size"] = {
            "max_over_min_ratio": float(flat.max() / max(flat.min(), 1e-9)),
            "coefficient_of_variation": float(flat.std() / max(flat.mean(), 1e-9)),
        }
        summary["overall_baseline_mean"] = float(flat.mean())
        summary["descriptive_reductions_not_pmr_attributable"] = {
            "baseline_to_no_pmr_pct": [r.relative_reduction_pct for r in reduction_baseline_to_no_pmr],
            "baseline_to_with_pmr_pct": [r.relative_reduction_pct for r in reduction_baseline_to_pmr],
        }
    with open(os.path.join(step_dir, "summary_step2.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[step2] overall: No PMR={overall_no_pmr:.4f} -> With PMR={overall_with_pmr:.4f} "
          f"({summary['overall_reduction_pct']:.1f}% reduction)")
    print(f"[step2] {n_parts_improved}/{num_parts} parts improved, "
          f"{n_parts_significant_raw}/{num_parts} significant (raw p<0.05), "
          f"{n_parts_significant_holm}/{num_parts} significant (Holm-Bonferroni corrected)")

    return {"no_pmr_results": no_pmr_results, "with_pmr_results": with_pmr_results,
            "baseline_results": baseline_results,
            "reduction": reduction, "summary": summary}


# ==========================================================================
# 7. STEP 3 -- connect Step 1 and Step 2: does PMR purify MORE where
#    discrepancy was originally LARGER? (-> the "both...and" in R3)
# ==========================================================================

def run_step3_joint(step1: dict, step2: dict, out_dir: str, num_parts: int) -> dict:
    step_dir = os.path.join(out_dir, "step3_joint")
    os.makedirs(step_dir, exist_ok=True)

    # Primary: Step 1's stripe-aggregated discrepancy (measured on the
    # No-PMR backbone via the INDEPENDENT fine grid, not the native-stripe
    # GeM used in Step 2) vs. Step 2's No-PMR -> With-PMR reduction. Using
    # Step 1's number here is the strongest version of the claim, since it
    # was derived without ever touching num_parts stripes directly at
    # extraction time.
    het_x = np.array([r.centroid_mean for r in step1["stripe_agg_results"]])
    reduction_y = np.array([r.relative_reduction_pct for r in step2["reduction"]])

    spearman_r, spearman_p = stats.spearmanr(het_x, reduction_y)
    pearson_r, pearson_p = stats.pearsonr(het_x, reduction_y)

    part_labels = [f"part {i+1}" for i in range(num_parts)]
    plot_correlation_scatter(
        het_x, reduction_y,
        x_label="No-PMR discrepancy (Step 1, grid-independent)",
        y_label="PMR reduction % (Step 2)",
        out_path=os.path.join(step_dir, "scatter_heterogeneity_vs_reduction.png"),
        point_labels=part_labels,
    )

    # Secondary cross-check: Step 2's OWN native-stripe No-PMR discrepancy
    # (same extractor family used for the reduction numbers) vs. the same
    # reduction. Should tell a consistent story with the primary check.
    het_x2 = np.array([r.centroid_mean for r in step2["no_pmr_results"]])
    spearman_r2, spearman_p2 = stats.spearmanr(het_x2, reduction_y)

    result = {
        "n_parts": num_parts,
        "caveat": ("n = num_parts data points -- a small-sample correlation. Treat the "
                   "p-value as indicative, not confirmatory; the sign and scatter pattern "
                   "matter more than the exact p at this sample size."),
        "primary_grid_independent_no_pmr_vs_reduction": {
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        },
        "secondary_native_stripe_no_pmr_vs_reduction": {
            "spearman_r": float(spearman_r2), "spearman_p": float(spearman_p2),
        },
        "interpretation": (
            "A positive correlation here means: parts that showed larger modality "
            "discrepancy WITHOUT PMR (Step 1's independent measurement) are the same "
            "parts where PMR achieves a larger relative reduction (Step 2). That is the "
            "direct, connected evidence R3 asks for -- spatial heterogeneity and PMR's "
            "purification effect are shown as one story, not two independent tables. A "
            "weak or negative correlation would instead suggest PMR's effect is roughly "
            "uniform across parts regardless of where discrepancy concentrates, which "
            "would need to be reported honestly rather than suppressed."
        ),
    }
    with open(os.path.join(step_dir, "correlation_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[step3] primary Spearman r = {spearman_r:.3f} (p={spearman_p:.3f}), "
          f"secondary Spearman r = {spearman_r2:.3f} (p={spearman_p2:.3f})")
    return result


def write_reviewer_response_md(step1: dict, step2: dict, step3: dict, out_dir: str, args) -> str:
    """Auto-generated, numbers-populated draft paragraph for the rebuttal
    letter. Meant as a starting point to edit, not a final answer."""
    s1 = step1["summary"]; s2 = step2["summary"]
    fine = s1["fine_grid"]; stripes = s1["stripe_aggregated"]
    align = s1["alignment_diagnostic"]
    md = f"""# Reviewer response -- spatial heterogeneity & PMR purification

Dataset: {args.dataset} | Plain checkpoint (Step 1): `{os.path.basename(args.plain_resume)}` |
No-PMR checkpoint (Step 2): `{os.path.basename(args.no_pmr_resume)}` | With-PMR checkpoint
(Step 2): `{os.path.basename(args.pmr_resume)}` (the No-PMR/With-PMR pair is the same one used
for Fig. 4(a)'s retrieval-metric ablation) | n = {s2['n_samples']} samples, {s2['num_parts']}
native parts, {fine['rows']}x{fine['cols']} fine grid.

## R1 / R2 -- spatial heterogeneity, validated independent of the stripe partition AND of any
## part-level training

On the plain backbone (`--plain_resume`: parts=False, PMR=False, TMS=False -- never exposed to
any part-level supervision), modality discrepancy measured on a
{fine['rows']}x{fine['cols']} generic grid varies significantly across space (Friedman p =
{fine['heterogeneity_test'].get('friedman_p', float('nan')):.2e}, n_identities matched =
{fine['heterogeneity_test'].get('n_matched_identities', 'n/a')}), with a
{fine['effect_size']['max_over_min_ratio']:.2f}x max/min ratio across regions
(CV = {fine['effect_size']['coefficient_of_variation']:.3f}). This measurement never touches
the model's own 6-stripe partition, so it is not assuming what it sets out to prove.

Aggregating the same fine grid into the model's {stripes['num_parts']} native horizontal
stripes, between-stripe variance is {align['between_stripe_variance']:.3e} vs. a mean
within-stripe variance of {align['mean_within_stripe_variance']:.3e}
({'a ratio favoring the stripe partition' if align['between_stripe_variance'] > align['mean_within_stripe_variance'] else 'comparable to or smaller than within-stripe variance -- see limitation note below'}).

## R3 -- PMR's purification effect, No PMR vs. With PMR

Comparing the No-PMR checkpoint's native-stripe discrepancy against the With-PMR checkpoint's
post-projector discrepancy -- the same two checkpoints already used for Fig. 4(a)'s retrieval
metrics -- overall discrepancy drops from {s2['overall_no_pmr_mean']:.4f} to
{s2['overall_with_pmr_mean']:.4f} ({s2['overall_reduction_pct']:.1f}% relative reduction).
{('Baseline (plain backbone) mean discrepancy: ' + format(s2['overall_baseline_mean'], '.4f') + '. ') if 'overall_baseline_mean' in s2 else ''}{s2['n_parts_with_positive_reduction']}/{s2['num_parts']} parts show a reduction;
{s2['n_parts_significant_holm_bonferroni']}/{s2['num_parts']} remain significant after
Holm-Bonferroni correction for multiple comparisons (one-sided paired Wilcoxon, per part).

## Connecting the two claims

Correlating Step 1's grid-independent No-PMR discrepancy per part against Step 2's
reduction per part: Spearman r =
{step3['primary_grid_independent_no_pmr_vs_reduction']['spearman_r']:.3f}
(p = {step3['primary_grid_independent_no_pmr_vs_reduction']['spearman_p']:.3f}, n={step3['n_parts']}).
{'This supports the claim that PMR purifies more where discrepancy was originally larger.' if step3['primary_grid_independent_no_pmr_vs_reduction']['spearman_r'] > 0 else 'This does NOT show the expected positive relationship and should be reported honestly as a limitation or investigated further.'}

*Caveat: n = {step3['n_parts']} data points for the correlation above -- report the scatter
pattern (see `step3_joint/scatter_heterogeneity_vs_reduction.png`) alongside the p-value,
not the p-value alone.*
"""
    path = os.path.join(out_dir, "reviewer_response.md")
    with open(path, "w") as f:
        f.write(md)
    return path


# ==========================================================================
# 8. REAL-DATA ENTRY POINT
# ==========================================================================

def run_from_project(args):
    import torch
    from data import get_test_loader

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[main] WARNING -- no CUDA device found, running on CPU (will be slow).")

    # ---- Plain checkpoint (backbone only; parts=False, PMR=False,
    #      TMS=False -- Table III row 1 "Baseline". Never exposed to any
    #      part-level supervision. Used ONLY for Step 1.) ----
    plain_backbone = build_minimal_backbone(pretrained_imagenet=False).to(device).eval()
    load_backbone_weights(plain_backbone, args.plain_resume, tag="plain")

    # ---- No-PMR checkpoint (backbone + part_bns; parts=True, PMR=False --
    #      the exact checkpoint behind Fig. 4(a)'s "Without PMR" row.
    #      Used ONLY for Step 2.) ----
    no_pmr_backbone = build_minimal_backbone(pretrained_imagenet=False).to(device).eval()
    load_backbone_weights(no_pmr_backbone, args.no_pmr_resume, tag="no_pmr")
    no_pmr_part_bns = build_part_bns(feat_dim=2048, num_parts=args.num_parts).to(device).eval()
    load_part_bns_weights(no_pmr_part_bns, args.no_pmr_resume)

    # ---- With-PMR checkpoint (backbone + part_bns + linear_projection --
    #      Fig. 4(a)'s "With PMR" row / the proposed model) ----
    pmr_backbone = build_minimal_backbone(pretrained_imagenet=False).to(device).eval()
    pmr_checkpoint = load_backbone_weights(pmr_backbone, args.pmr_resume, tag="pmr")
    pmr_head = build_pmr_head(feat_dim=2048, num_parts=args.num_parts).to(device).eval()
    load_pmr_head_weights(pmr_head, pmr_checkpoint)

    gallery_loader, query_loader = get_test_loader(
        dataset=args.dataset, root=args.data_root, batch_size=args.batch_size,
        image_size=tuple(args.image_size), num_workers=args.num_workers,
    )
    loaders = []
    if args.split in ("gallery", "both"):
        loaders.append(("gallery", gallery_loader))
    if args.split in ("query", "both"):
        loaders.append(("query", query_loader))
    if not loaders:
        raise ValueError(f"Unrecognized --split {args.split}")

    # Step 1's grid extractor deliberately runs on plain_backbone (never
    # saw a part branch) -- NOT on no_pmr_backbone or pmr_backbone.
    grid_extractor = RegionalGridExtractor(plain_backbone, args.grid_rows, args.grid_cols, args.test_p)
    no_pmr_stripe_extractor = NativeStripeExtractor(no_pmr_backbone, args.num_parts, args.test_p)
    pmr_stripe_extractor = NativeStripeExtractor(pmr_backbone, args.num_parts, args.test_p)
    # Baseline stripes: SAME NativeStripeExtractor (same GeM p, same torch.chunk
    # partition) on the PLAIN backbone, so all three bars in the headline figure
    # differ only in the checkpoint, never in the pooling operation.
    plain_stripe_extractor = NativeStripeExtractor(plain_backbone, args.num_parts, args.test_p)

    def extract_fn(images):
        raw_no_pmr = no_pmr_stripe_extractor(images)
        raw_pmr = pmr_stripe_extractor(images)
        return {
            "plain_fine": grid_extractor(images),
            "baseline_stripe": plain_stripe_extractor(images),
            "no_pmr_stripe": part_bns_forward(no_pmr_part_bns, raw_no_pmr, args.num_parts),
            "with_pmr_stripe": pmr_head_forward(pmr_head, raw_pmr, args.num_parts),
        }

    data = run_feature_extraction(loaders, extract_fn, device, args.max_batches, tag="extract")
    ids, modality = data["ids"], data["modality"]

    out_dir = args.out_dir or os.path.join("visualizations/spatial_heterogeneity", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    step1 = run_step1_heterogeneity(data["plain_fine"], ids, modality, args.grid_rows,
                                     args.grid_cols, args.num_parts, out_dir,
                                     args.n_boot, args.n_perm, args.seed)
    step2 = run_step2_reduction(data["no_pmr_stripe"], data["with_pmr_stripe"],
                                 ids, modality, args.num_parts, out_dir,
                                 args.n_boot, args.n_perm, args.seed,
                                 baseline_feats=data["baseline_stripe"])
    step3 = run_step3_joint(step1, step2, out_dir, args.num_parts)
    md_path = write_reviewer_response_md(step1, step2, step3, out_dir, args)

    full_summary = {"step1": step1["summary"], "step2": step2["summary"], "step3": step3}
    with open(os.path.join(out_dir, "summary_full.json"), "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"[main] All results written to {out_dir}")
    print(f"[main] Draft rebuttal paragraph: {md_path}")
    return full_summary


# ==========================================================================
# 9. SYNTHETIC SELF-TEST -- validates the full Step1 -> Step2 -> Step3
#    orchestration and statistics with known ground truth, no torch, no
#    project files, no checkpoints needed.
# ==========================================================================

def _row_shift_profile(grid_rows: int) -> np.ndarray:
    t = np.arange(grid_rows) / (grid_rows - 1)
    return 0.15 + 0.9 * np.exp(-((t - 0.42) ** 2) / (2 * 0.16 ** 2))


def _build_synthetic_fine_grid(grid_rows, grid_cols, D, n_identities, spp, seed):
    """Same generator as experiment1.py's positive control: a row-peaked
    (torso-like) discrepancy profile, replicated across columns."""
    rng = np.random.default_rng(seed)
    num_regions = grid_rows * grid_cols
    row_profile = _row_shift_profile(grid_rows)
    region_shift = np.repeat(row_profile, grid_cols)

    feats, ids, modality = [], [], []
    for pid in range(n_identities):
        base = rng.normal(size=(num_regions, D)); base /= np.linalg.norm(base, axis=1, keepdims=True)
        shift_dir = rng.normal(size=(num_regions, D)); shift_dir /= np.linalg.norm(shift_dir, axis=1, keepdims=True)
        for m in (0, 1):
            sign = 1.0 if m == 0 else -1.0
            for _ in range(spp):
                noise = rng.normal(scale=0.08, size=(num_regions, D))
                sample = base + sign * 0.5 * region_shift[:, None] * shift_dir + noise
                feats.append(sample); ids.append(pid); modality.append(m)
    return (np.stack(feats).astype(np.float32), np.array(ids), np.array(modality), region_shift)


def _build_synthetic_pipeline(num_parts, D, n_identities, spp, seed, stripe_shift_mag,
                               reduction_profile):
    """Builds No-PMR / With-PMR native-stripe features for the SAME
    identities/images as _build_synthetic_fine_grid, with a per-part
    reduction_profile that we deliberately correlate with stripe_shift_mag
    (so Step 3's correlation check has known ground truth to recover)."""
    rng = np.random.default_rng(seed + 1000)
    no_pmr_list, with_pmr_list, ids, modality = [], [], [], []
    for pid in range(n_identities):
        base = rng.normal(size=(num_parts, D)); base /= np.linalg.norm(base, axis=1, keepdims=True)
        shift_dir = rng.normal(size=(num_parts, D)); shift_dir /= np.linalg.norm(shift_dir, axis=1, keepdims=True)
        for m in (0, 1):
            sign = 1.0 if m == 0 else -1.0
            for _ in range(spp):
                noise_no_pmr = rng.normal(scale=0.08, size=(num_parts, D))
                no_pmr = base + sign * 0.5 * stripe_shift_mag[:, None] * shift_dir + noise_no_pmr
                remaining = stripe_shift_mag * (1.0 - reduction_profile)
                noise_with_pmr = rng.normal(scale=0.08, size=(num_parts, D))
                with_pmr = base + sign * 0.5 * remaining[:, None] * shift_dir + noise_with_pmr
                no_pmr_list.append(no_pmr); with_pmr_list.append(with_pmr)
                ids.append(pid); modality.append(m)
    return (np.stack(no_pmr_list).astype(np.float32), np.stack(with_pmr_list).astype(np.float32),
            np.array(ids), np.array(modality))


def run_self_test():
    GRID_ROWS, GRID_COLS, NUM_PARTS, D, N_ID, SPP = 12, 4, 6, 64, 60, 4
    OUT = "self_test_reviewer_response_out"

    print("=" * 78)
    print("SELF-TEST -- combined Step1 -> Step2 -> Step3 pipeline, known ground truth")
    print("=" * 78)

    fine_feats, ids, modality, region_shift = _build_synthetic_fine_grid(
        GRID_ROWS, GRID_COLS, D, N_ID, SPP, seed=0)
    row_profile = _row_shift_profile(GRID_ROWS)
    rows_per_stripe = GRID_ROWS // NUM_PARTS
    stripe_shift_mag = row_profile.reshape(NUM_PARTS, rows_per_stripe).mean(axis=1)

    # Reduction profile deliberately correlated with stripe_shift_mag: parts
    # with larger baseline discrepancy get purified MORE (the interesting,
    # expected case a real PMR should exhibit).
    norm = (stripe_shift_mag - stripe_shift_mag.min()) / (np.ptp(stripe_shift_mag) + 1e-9)
    reduction_profile = 0.2 + 0.65 * norm  # in [0.2, 0.85]

    no_pmr_stripe, with_pmr_stripe, ids2, modality2 = _build_synthetic_pipeline(
        NUM_PARTS, D, N_ID, SPP, seed=0, stripe_shift_mag=stripe_shift_mag,
        reduction_profile=reduction_profile)

    assert np.array_equal(ids, ids2) and np.array_equal(modality, modality2), (
        "Synthetic fine-grid and stripe-pipeline generators drifted out of sync."
    )

    step1 = run_step1_heterogeneity(fine_feats, ids, modality, GRID_ROWS, GRID_COLS, NUM_PARTS,
                                     OUT, n_boot=1000, n_perm=1000, seed=0)
    fine_vals = np.array([r.centroid_mean for r in step1["fine_results"]])
    corr_gt = np.corrcoef(fine_vals, region_shift)[0, 1]
    print(f"[self_test] Step1: fine-grid vs. injected ground-truth correlation = {corr_gt:.3f} (expect > 0.8)")
    assert step1["summary"]["fine_grid"]["heterogeneity_test"]["friedman_p"] < 0.05 and corr_gt > 0.7, \
        "STEP 1 FAILED to recover known injected spatial pattern."
    print("[self_test] Step1 PASSED.")

    # Synthetic BASELINE: same identities/order, slightly LARGER injected shift than
    # No PMR (x1.15) and an independent noise seed. Purely to exercise the code path.
    baseline_stripe, _, ids_b, modality_b = _build_synthetic_pipeline(
        NUM_PARTS, D, N_ID, SPP, seed=1, stripe_shift_mag=stripe_shift_mag * 1.15,
        reduction_profile=reduction_profile)
    assert np.array_equal(ids, ids_b) and np.array_equal(modality, modality_b)

    step2 = run_step2_reduction(no_pmr_stripe, with_pmr_stripe, ids, modality,
                                 NUM_PARTS, OUT, n_boot=1000, n_perm=1000, seed=0,
                                 baseline_feats=baseline_stripe,
                                 watermark="SYNTHETIC - LAYOUT CHECK ONLY")
    fig_path = os.path.join(OUT, "step2_reduction", "bar_baseline_no_pmr_pmr.png")
    assert os.path.exists(fig_path), "three-way figure was not written"
    assert step2["summary"]["baseline_heterogeneity"]["friedman_p"] < 0.05, \
        "Baseline stripes should show injected heterogeneity."
    assert step2["summary"]["n_parts_pmr_lower_under_relative_gap"] >= 4, \
        "Normalised gap contradicts the injected reduction profile."
    print(f"[self_test] Three-way figure written: {fig_path}")
    reductions = [r.relative_reduction_pct for r in step2["reduction"]]
    # part with the largest injected reduction_profile should show (roughly)
    # the largest measured relative_reduction_pct
    assert int(np.argmax(reductions)) == int(np.argmax(reduction_profile)), \
        "STEP 2 FAILED: measured reduction ranking doesn't match injected profile's peak."
    print(f"[self_test] Step2: measured reduction % by part = {[f'{v:.1f}' for v in reductions]}")
    print("[self_test] Step2 PASSED.")

    step3 = run_step3_joint(step1, step2, OUT, NUM_PARTS)
    r = step3["primary_grid_independent_no_pmr_vs_reduction"]["spearman_r"]
    print(f"[self_test] Step3: primary Spearman r (heterogeneity vs. reduction) = {r:.3f} (expect > 0.5)")
    assert r > 0.5, "STEP 3 FAILED to recover the injected positive correlation."
    print("[self_test] Step3 PASSED.")

    write_reviewer_response_md(step1, step2, step3, OUT, argparse.Namespace(
        dataset="synthetic", plain_resume="synthetic_plain.pth",
        no_pmr_resume="synthetic_no_pmr.pth", pmr_resume="synthetic_pmr.pth"))

    print("\nAll self-tests PASSED. Full synthetic pipeline output in "
          f"./{OUT}/ (including reviewer_response.md).")


# ==========================================================================
# 10. CLI
# ==========================================================================

def _parse_image_size(value):
    if isinstance(value, (list, tuple)):
        return [int(value[0]), int(value[1])]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return [int(parsed[0]), int(parsed[1])]
    raise ValueError(f"Can't parse image_size from {value!r}")


def _load_cfg_defaults(cfg_path: str) -> dict:
    defaults = {}
    try:
        import yaml
    except ImportError:
        print(f"[main] PyYAML not available; ignoring --cfg {cfg_path}.")
        return defaults
    try:
        with open(cfg_path) as f:
            y = yaml.safe_load(f)
        for key in ("dataset", "batch_size", "num_parts"):
            if key in y:
                defaults[key] = y[key]
        if "image_size" in y:
            defaults["image_size"] = _parse_image_size(y["image_size"])
        if "test_p" in y:
            defaults["test_p"] = float(y["test_p"])
    except Exception as e:
        print(f"[main] Could not read --cfg {cfg_path} ({e}); using CLI defaults instead.")
    return defaults


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cfg", type=str, default=None,
                             help="Optional yaml (e.g. configs/SYSU.yml) to prefill a few "
                                  "defaults (dataset/image_size/batch_size/num_parts/test_p).")
    pre_args, _ = pre_parser.parse_known_args()
    cfg_defaults = _load_cfg_defaults(pre_args.cfg) if pre_args.cfg else {}

    parser = argparse.ArgumentParser(parents=[pre_parser], description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str, default="/home/vipin/PhD/datasets/cross_modality/SYSU-MM01/SYSU-MM01",
                         help="Root directory of the dataset (required unless --self_test).")
    parser.add_argument("--plain_resume", type=str, default="/home/vipin/PhD/models/vireid/ckptlog/sysu/SYSU_Time-2026-07-23_12-41-26/SYSU_Time-2026-07-23_12-41-26_model_best-245.pth",
                         help="Baseline checkpoint")
    parser.add_argument("--no_pmr_resume", type=str, default="/home/vipin/PhD/models/vireid/ckptlog/sysu/SYSU_Time-2026-07-25_23-56-12/SYSU_Time-2026-07-25_23-56-12_model_best-106.pth",
                         help="Non PMR Checkpoint")
    parser.add_argument("--pmr_resume", type=str, default="/home/vipin/PhD/models/vireid/ckptlog/sysu/SYSU_Time-2026-07-19_01-48-05/SYSU_Time-2026-07-19_01-48-05_model_best-204.pth",
                         help="PMR Trained model")
    parser.add_argument("--dataset", type=str, default=cfg_defaults.get("dataset", "sysu"),
                         choices=["sysu", "regdb", "llcm"])
    parser.add_argument("--image_size", type=int, nargs=2, default=cfg_defaults.get("image_size", [384, 192]))
    parser.add_argument("--batch_size", type=int, default=cfg_defaults.get("batch_size", 128))
    parser.add_argument("--test_p", type=float, default=cfg_defaults.get("test_p", 3.5),
                         help="GeM pooling exponent, matches SYSU.yml's test_p.")
    parser.add_argument("--num_parts", type=int, default=cfg_defaults.get("num_parts", 6),
                         help="Must match both checkpoints' num_parts (SYSU.yml default 6).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--split", choices=["gallery", "query", "both"], default="both",
                         help="'both' (default) is almost always correct -- SYSU-MM01's "
                              "gallery is RGB-only and query is IR-only, so combining them "
                              "is required to get both modalities.")
    parser.add_argument("--grid_rows", type=int, default=12,
                         help="Fine grid rows for Step 1 (heterogeneity). Must be a multiple "
                              "of --num_parts.")
    parser.add_argument("--grid_cols", type=int, default=4, help="Fine grid columns for Step 1.")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="Defaults to reviewer_response_out/<dataset>/")
    parser.add_argument("--max_batches", type=int, default=None,
                         help="Cap batches per loader, for a quick smoke test.")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--font_size", type=float, default=17.0,
                         help="Base font size (pt) for ALL figure text: axis labels and "
                              "legend use this, tick labels use font_size-2. Default 17.")
    parser.add_argument("--axes_height", type=float, default=FIG5_AXES_HEIGHT_IN,
                         help="Plotting-area height in inches shared by the Fig. 5 bar "
                              "chart and heatmap (both images get identical height).")
    parser.add_argument("--self_test", action="store_true",
                         help="Run the synthetic Step1->Step2->Step3 correctness check "
                              "instead of touching real data/checkpoints. No torch needed.")
    args = parser.parse_args()
    apply_plot_style(args.font_size, args.axes_height)

    if args.self_test:
        run_self_test()
    else:
        if not args.data_root or not args.plain_resume or not args.no_pmr_resume or not args.pmr_resume:
            parser.error("--data_root, --plain_resume, --no_pmr_resume and --pmr_resume are "
                         "all required unless --self_test is given.")
        run_from_project(args)