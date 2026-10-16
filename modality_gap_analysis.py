"""
modality_gap_analysis.py
=========================
Task-independent analysis of the image-level modality gap between REAL IR
images and various visible->IR "IR-like" transformations (Grayscale, CAJ,
FDA, TMS, ...), on SYSU-MM01.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python modality_gap_analysis.py \
        --sysu_root /path/to/SYSU-MM01 \
        --transform_py /path/to/transform.py \
        --caj_py       /path/to/caj.py \
        --fda_py       /path/to/fda.py \
        --rgpr_py      /path/to/rgpr.py \
        --maa_py       /path/to/maa.py \
        --out_dir results/modality_gap \
        --num_samples 5000

--------------------------------------------------------------------------
- FID assumes reasonably large sample sizes for stable covariance
  estimates; report the exact N used (--num_samples) and, ideally, run
  with >= 3 random seeds and report mean +/- std, not a single number.

- The frozen extractor is plain torchvision ResNet50 (ImageNet-1k V1
  weights, standard stride, NOT the modified/downsampling-removed variant
  used as this project's ReID backbone F, and NOT loaded with any
  ReID-finetuned checkpoint -- this is what keeps the metric non-circular).
  It is a natural-RGB feature space, not domain-specific to pedestrians or
  thermal imagery. Report the exact extractor + layer + resolution + N used
  (as here) so the numbers are reproducible; note that "FID"/"MMD" computed
  in a ResNet50 feature space are not the canonical Inception-v3-based FID
  from Heusel et al. (2017) -- comparable across the methods in THIS table,
  but not to externally published FID numbers.

- CAJ, FDA and TMS are run with p=1.0 (always transform) so every sampled
  image is actually converted -- this differs from their p<1 use as
  stochastic *training* augmentations, and is the correct setting for an
  analysis that asks "what does the transform produce", not "how often is
  it applied during training".

  - FDA draws its low-frequency reference from the same train-IR pool used
  as the "real IR" target distribution for FID/MMD. This mirrors how FDA
  is actually used in training, but means the FDA column is not fully
  independent of the real-IR reference set; note this if a reviewer asks.

  - RGPR, unlike FDA, needs no real-IR reference image at all -- it only
  reads the visible image's own grayscale conversion for the replaced
  patch, so its FID/MMD column is fully independent of the real-IR
  reference set used here. Worth stating explicitly alongside the FDA
  caveat above, since it's a genuine methodological difference between
  the two baselines, not just a missing feature.

  - MAA (MCJA's Visible-Infrared Modality Coordinator) is composed exactly
  as in MCJA's data/__init__.py: a RandomChoice over WeightedGrayscale,
  ChannelCutMix and SpectrumJitter(factor=(0, 1)), each taken from the
  project's own maa.py. Two deliberate differences from MCJA training,
  both consistent with how every other column here is produced:
    (i)  each branch is force-applied (p=1.0 instead of p=0.5), and
    (ii) MCJA's ColorJitter(hue) branch is excluded, since it is a generic
         colour augmentation and not part of MAA's VI-specific operators.
  MAA is therefore a MIXTURE of three transforms (~1/3 each), so its
  FID/MMD describes that mixture distribution, not any single operator.
  SpectrumJitter blends a single-channel image back toward the RGB input
  with factor ~ U(0, 1), so part of MAA's output retains colour by design.
  Like RGPR, MAA needs no real-IR reference image.

  - MAA runs on its own isolated RNG stream (see RNGIsolatedTransform),
  so adding it does not consume random numbers from the global stream and
  leaves every other method's transformed images, FID/MMD numbers and
  qualitative-panel cells bit-identical to runs without MAA (same seed).
"""

import os
import glob
import random
import argparse
import importlib.util
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import linalg
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


# ==========================================================================
# 1. Transform registry  (extensible; see module docstring)
# ==========================================================================
TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def register_transform(name: str, overwrite: bool = False):
    """Decorator/registrar: registers a visible->IR-like transform under `name`.

    The wrapped function must accept a PIL.Image (RGB) and return a
    PIL.Image (RGB, uint8). Set overwrite=True to replace an existing
    registration (used when this script re-imports project modules, e.g.
    in an interactive session where run_analysis() is called more than once).
    """
    def decorator(fn: Callable):
        if name in TRANSFORM_REGISTRY and not overwrite:
            raise ValueError(
                f"Transform '{name}' is already registered. "
                f"Pass overwrite=True to replace it."
            )
        TRANSFORM_REGISTRY[name] = fn
        return fn
    return decorator


@register_transform("Grayscale")
def to_grayscale(img: Image.Image) -> Image.Image:
    """Standard fixed-coefficient grayscale conversion, replicated to 3ch.
    """
    gray = img.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


# ==========================================================================
# 2. Dynamic import of the PROJECT'S ACTUAL CAJ / FDA / TMS implementations
#    (no reimplementation lives in this file)
# ==========================================================================
def import_from_path(module_name: str, file_path: str):
    file_path = str(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cannot find module file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_module_path(explicit_path: Optional[str], filename: str, script_dir: Path) -> str:
    """Explicit CLI path wins; otherwise look in cwd, then next to this script."""
    if explicit_path:
        return explicit_path
    candidates = [Path.cwd() / filename, script_dir / filename]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Could not locate '{filename}'. Pass its path explicitly, e.g. "
        f"--{filename.split('.')[0]}_py /path/to/{filename}"
    )


def register_project_methods(transform_py: str, caj_py: str, fda_py: str, rgpr_py: str,
                              ir_reference_paths: List[str]):
    """
    Loads TMS from transform.py, ChannelExchange from caj.py, FDATransform
    (+ sysu_ir_train_paths) from fda.py, and RandomGrayscalePatchReplacement
    from rgpr.py; instantiates each with p=1.0 (force-apply, see module
    docstring caveats), and registers them into TRANSFORM_REGISTRY as
    "TMS", "CAJ", "FDA", "RGPR".

    Returns the fda module's `sysu_ir_train_paths` function so the caller
    can reuse it (rather than duplicating SYSU path-collection logic).
    """
    transform_mod = import_from_path("project_transform_mod", transform_py)
    caj_mod = import_from_path("project_caj_mod", caj_py)
    fda_mod = import_from_path("project_fda_mod", fda_py)
    rgpr_mod = import_from_path("project_rgpr_mod", rgpr_py)

    TMS_cls = transform_mod.TMS
    ChannelExchange_cls = caj_mod.ChannelExchange
    FDATransform_cls = fda_mod.FDATransform
    RGPR_cls = rgpr_mod.RandomGrayscalePatchReplacement

    tms_instance = TMS_cls(p=1.0)
    caj_instance = ChannelExchange_cls(p=1.0)
    fda_instance = FDATransform_cls(ir_paths=ir_reference_paths, beta=0.01, p=1.0)
    rgpr_instance = RGPR_cls(p=1.0)

    register_transform("TMS", overwrite=True)(lambda img: tms_instance(img))
    register_transform("CAJ", overwrite=True)(lambda img: caj_instance(img))
    register_transform("FDA", overwrite=True)(lambda img: fda_instance(img))
    register_transform("RGPR", overwrite=True)(lambda img: rgpr_instance(img))

    return fda_mod.sysu_ir_train_paths


class RNGIsolatedTransform:
    """Runs `fn` on a private RNG stream (python `random`, numpy, torch CPU).

    The global RNG states are saved before each call and restored after it,
    and the private states are carried forward between calls. Consequences:
      - MAA is itself reproducible for a given seed, and
      - inserting MAA into the pipeline does not shift the random draws of
        any other (stochastic) method, in either the FID/MMD pass or the
        qualitative-panel pass.
    """

    def __init__(self, fn: Callable, seed: int):
        self.fn = fn
        self._py_state = random.Random(seed).getstate()
        self._np_state = np.random.RandomState(seed).get_state()
        g = torch.Generator()
        g.manual_seed(seed)
        self._torch_state = g.get_state()

    def __call__(self, img: Image.Image) -> Image.Image:
        outer_py = random.getstate()
        outer_np = np.random.get_state()
        outer_torch = torch.get_rng_state()
        random.setstate(self._py_state)
        np.random.set_state(self._np_state)
        torch.set_rng_state(self._torch_state)
        try:
            out = self.fn(img)
        finally:
            self._py_state = random.getstate()
            self._np_state = np.random.get_state()
            self._torch_state = torch.get_rng_state()
            random.setstate(outer_py)
            np.random.set_state(outer_np)
            torch.set_rng_state(outer_torch)
        return out


def register_maa(maa_py: str, seed: int):
    """
    Loads WeightedGrayscale, ChannelCutMix and SpectrumJitter from maa.py
    and registers MAA = RandomChoice([WG, CC, SJ]) under the name "MAA",
    mirroring MCJA's data/__init__.py composition (see module docstring for
    the two deliberate differences: p=1.0 per branch, ColorJitter excluded).

    Kept separate from register_project_methods() so the existing
    TMS/CAJ/FDA/RGPR registration is untouched, and called AFTER it so MAA
    is the last registered method (last column / last processed).
    """
    maa_mod = import_from_path("project_maa_mod", maa_py)

    maa_instance = T.RandomChoice([
        maa_mod.WeightedGrayscale(p=1.0),
        maa_mod.ChannelCutMix(p=1.0),
        maa_mod.SpectrumJitter(factor=(0.00, 1.00), p=1.0),
    ])

    def _maa(img: Image.Image) -> Image.Image:
        return maa_instance(img).convert("RGB")

    register_transform("MAA", overwrite=True)(RNGIsolatedTransform(_maa, seed=seed))


# ==========================================================================
# 3. SYSU-MM01 path collection (mirrors fda.py's sysu_ir_train_paths, for
#    the RGB side; the IR side is imported directly from fda.py, not
#    duplicated here)
# ==========================================================================
def sysu_train_identities(root: str) -> List[int]:
    ids = ''
    for name in ('train_id.txt', 'val_id.txt'):
        with open(os.path.join(root, 'exp', name)) as f:
            ids += ',' + f.readline().strip()
    return sorted({int(i) for i in ids.split(',') if i.strip()})


def sysu_rgb_train_paths(root: str) -> List[str]:
    """RGB counterpart of fda.sysu_ir_train_paths: cameras 1, 2, 4, 5."""
    ids = sysu_train_identities(root)
    paths = []
    for cam in ('cam1', 'cam2', 'cam4', 'cam5'):
        for pid in ids:
            paths += sorted(glob.glob(os.path.join(root, cam, '%04d' % pid, '*.jpg')))
    assert len(paths) > 0, 'No RGB images found - check the SYSU-MM01 root path'
    return paths


# ==========================================================================
# 4. Frozen feature extractor (shared by FID + MMD; avoids circularity
#    with the trained ReID backbone)
# ==========================================================================
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        net = resnet50(weights=weights)
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.to(device)
        self.device = device
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack(
                [self.preprocess(im.convert("RGB")) for im in batch]
            ).to(self.device)
            out = self.net(tensors)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


# ==========================================================================
# 5. Metrics: FID, and MMD
# ==========================================================================
def compute_fid(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    """Standard FID between two sets of features (Gaussian approximation)."""
    mu_a, sigma_a = feats_a.mean(axis=0), np.cov(feats_a, rowvar=False)
    mu_b, sigma_b = feats_b.mean(axis=0), np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean)
    return float(fid)


def compute_mmd(feats_a: np.ndarray, feats_b: np.ndarray, num_sigmas: int = 5) -> float:
    """Unbiased MMD^2 with a multi-scale RBF kernel (median-heuristic bandwidths)."""
    a = torch.from_numpy(feats_a).float()
    b = torch.from_numpy(feats_b).float()

    joint = torch.cat([a, b], dim=0)
    d_joint = torch.cdist(joint, joint, p=2) ** 2
    nonzero = d_joint[d_joint > 0]
    median_dist = nonzero.median().item() if nonzero.numel() > 0 else 1.0
    scale_factors = [0.25, 0.5, 1.0, 2.0, 4.0][:num_sigmas]
    sigmas = [median_dist * s for s in scale_factors]

    def kernel(x, y):
        dists = torch.cdist(x, y, p=2) ** 2
        return sum(torch.exp(-dists / (2 * s + 1e-12)) for s in sigmas)

    Kaa = kernel(a, a)
    Kbb = kernel(b, b)
    Kab = kernel(a, b)

    m, n = a.size(0), b.size(0)
    mmd2 = (
        (Kaa.sum() - Kaa.trace()) / (m * (m - 1))
        + (Kbb.sum() - Kbb.trace()) / (n * (n - 1))
        - 2 * Kab.mean()
    )
    return float(mmd2.item())

def find_matched_ir(rgb_path: str, ir_root: str, pid: int,
                     ir_cams=("cam3", "cam6")) -> Optional[str]:
    for cam in ir_cams:
        matches = sorted(glob.glob(os.path.join(ir_root, cam, "%04d" % pid, "*.jpg")))
        if matches:
            return matches[0]
    return None

def letterbox(img: Image.Image, target_size=(128, 256), fill=(255, 255, 255)):
    img = img.convert("RGB")
    tw, th = target_size
    w, h = img.size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_resized = img.resize((nw, nh), Image.BICUBIC)

    canvas = Image.new("RGB", target_size, fill)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas.paste(img_resized, (x0, y0))
    return canvas

# ==========================================================================
# 6. Qualitative panel
# ==========================================================================
# def save_qualitative_panel(
#     rgb_images: List[Image.Image],
#     ir_images: List[Image.Image],
#     transformed: Dict[str, List[Image.Image]],
#     out_path: str,
#     num_rows: int = 6,
# ):
#     method_names = list(transformed.keys())
#     n_cols = 2 + len(method_names)  # Visible, [methods...], Real IR
#     n_rows = min(num_rows, len(rgb_images))

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
#     if n_rows == 1:
#         axes = axes.reshape(1, -1)

#     col_titles = ["Visible"] + method_names + ["Real IR"]

#     for row in range(n_rows):
#         axes[row, 0].imshow(rgb_images[row])
#         for col, name in enumerate(method_names, start=1):
#             axes[row, col].imshow(transformed[name][row])
#         ir_idx = row % len(ir_images)
#         axes[row, -1].imshow(ir_images[ir_idx])
#         for col in range(n_cols):
#             axes[row, col].axis("off")

#     for col, title in enumerate(col_titles):
#         axes[0, col].set_title(title, fontsize=14)

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()

def save_qualitative_panel(
    rgb_images: List[Image.Image],
    ir_images_matched: List[Image.Image],   # MUST be same-identity as rgb_images[row], same length
    transformed: Dict[str, List[Image.Image]],
    out_path: str,
    num_rows: int = 6,
    cell_size=(128, 256),          # (w, h) in px, letterbox target — keep identical across all cells
    row_labels: Optional[List[str]] = None,  # e.g. identity IDs, for traceability
):
    assert len(ir_images_matched) >= len(rgb_images), \
        "ir_images_matched must be identity-paired 1:1 with rgb_images, not a separate pool."

    method_names = list(transformed.keys())
    n_cols = 2 + len(method_names)
    n_rows = min(num_rows, len(rgb_images))

    cell_w_in, cell_h_in = 1.6, 3.2  # inches per cell, matches cell_size aspect (128:256 = 1:2)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w_in * n_cols, cell_h_in * n_rows),
        gridspec_kw=dict(wspace=0.03, hspace=0.03),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Visible"] + method_names + ["Real IR"]

    for row in range(n_rows):
        cells = (
            [rgb_images[row]]
            + [transformed[name][row] for name in method_names]
            + [ir_images_matched[row]]
        )
        for col, im in enumerate(cells):
            im = letterbox(im, target_size=cell_size)
            axes[row, col].imshow(im)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        if row_labels is not None:
            axes[row, 0].set_ylabel(row_labels[row], fontsize=9, rotation=0,
                                     labelpad=25, va="center")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
# ==========================================================================
# 7. Main pipeline
# ==========================================================================
def sample_and_load(paths: List[str], num_samples: int, seed: int) -> List[Image.Image]:
    rng = random.Random(seed)
    paths = paths.copy()
    rng.shuffle(paths)
    paths = paths[:num_samples]
    return [Image.open(p).convert("RGB") for p in paths]


def sample_paths(paths: List[str], num_samples: int, seed: int) -> List[str]:
    """Reproduces the EXACT seeded shuffle+slice that sample_and_load() applies,
    but returns paths instead of loaded images.

    Used only to recover which file backs rgb_images[i] for the qualitative
    panel's identity matching -- sample_and_load() itself, and every FID/MMD
    call site that uses it, is left completely untouched.
    """
    rng = random.Random(seed)
    paths = paths.copy()
    rng.shuffle(paths)
    return paths[:num_samples]


def get_pid_from_path(path: str) -> int:
    """SYSU-MM01 layout: .../camX/%04d/xxxx.jpg -> pid is the parent directory name."""
    return int(os.path.basename(os.path.dirname(path)))


def build_and_save_qualitative_panel(
    rgb_paths_all: List[str],
    rgb_images: List[Image.Image],
    sysu_root: str,
    num_samples: int,
    seed: int,
    methods: List[str],
    out_path: str,
    num_rows: int = 6,
    cell_size=(128, 256),
) -> None:
    """
    Builds the Visible / [methods] / Real-IR qualitative panel with the
    Real-IR column identity-matched (same pid) to the Visible column in
    each row -- this is the piece that was previously broken: the old
    call site passed an independently-sampled `ir_images` pool straight
    into save_qualitative_panel's `ir_images_matched` argument, so the
    "Real IR" column showed a random identity, not the same person as
    the Visible column.

    Does NOT touch the FID/MMD pipeline: `rgb_images` and its `transformed`
    dict computed for the full FID sample are only read here, never
    mutated, and the panel's own transformed subset is computed separately
    below rather than by slicing the FID-sample `transformed` dict (whose
    row order does not correspond to identity-matched rows).

    Identities with no captured IR image (either cam3 or cam6) are
    skipped, never silently replaced by an unrelated IR image.
    """
    rgb_paths_sampled = sample_paths(rgb_paths_all, num_samples, seed)
    assert len(rgb_paths_sampled) == len(rgb_images), (
        "sample_paths() must reproduce the exact same seeded shuffle+slice as "
        "sample_and_load() -- got mismatched lengths, refusing to guess pairing."
    )

    panel_rgb_images, panel_ir_images, panel_pids = [], [], []
    for path, img in zip(rgb_paths_sampled, rgb_images):
        if len(panel_rgb_images) >= num_rows:
            break
        pid = get_pid_from_path(path)
        ir_path = find_matched_ir(path, sysu_root, pid)
        if ir_path is None:
            continue  # no real IR capture for this identity -- skip, don't substitute
        panel_rgb_images.append(img)
        panel_ir_images.append(Image.open(ir_path).convert("RGB"))
        panel_pids.append(pid)

    if len(panel_rgb_images) < num_rows:
        print(
            f"[warn] Only found {len(panel_rgb_images)}/{num_rows} identity-matched "
            f"RGB/IR pairs among the first {num_samples} sampled RGB paths; "
            f"panel will have fewer rows than requested."
        )

    print(
        "Applying transforms to the panel's matched RGB subset only "
        "(independent of the full-pool `transformed` dict used for FID/MMD) ..."
    )
    panel_transformed = {
        name: [TRANSFORM_REGISTRY[name](img) for img in panel_rgb_images]
        for name in methods
    }

    save_qualitative_panel(
        rgb_images=panel_rgb_images,
        ir_images_matched=panel_ir_images,
        transformed=panel_transformed,
        out_path=out_path,
        num_rows=len(panel_rgb_images),
        cell_size=cell_size,
        row_labels=[f"ID {pid}" for pid in panel_pids],
    )


def run_analysis(
    sysu_root: str,
    transform_py: str,
    caj_py: str,
    fda_py: str,
    rgpr_py: str,
    out_dir: str,
    num_samples: int = 500,
    device: str = "cuda",
    methods: Optional[List[str]] = None,
    seed: int = 8,
    maa_py: Optional[str] = None,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    device = device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] CUDA not available, running on CPU (will be slower).")

    print(f"Deriving RGB/IR training pools from SYSU-MM01 root: {sysu_root}")
    rgb_paths_all = sysu_rgb_train_paths(sysu_root)

    # sysu_ir_train_paths comes straight from the project's own fda.py --
    # not duplicated here.
    fda_mod_paths = import_from_path("project_fda_mod_paths_only", fda_py)
    ir_paths_all = fda_mod_paths.sysu_ir_train_paths(sysu_root)
    print(f"  RGB train pool: {len(rgb_paths_all)} images")
    print(f"  IR  train pool: {len(ir_paths_all)} images")

    print(f"Sampling {num_samples} images per domain (seed={seed}) ...")
    rgb_images = sample_and_load(rgb_paths_all, num_samples, seed)
    ir_images = sample_and_load(ir_paths_all, num_samples, seed)

    print("Registering project transforms (TMS, CAJ, FDA, RGPR) from your actual modules ...")
    register_project_methods(transform_py, caj_py, fda_py, rgpr_py, ir_reference_paths=ir_paths_all)

    if maa_py is not None:
        print("Registering MAA (WG / CC / SJ RandomChoice) from your maa.py ...")
        register_maa(maa_py, seed=seed)
    else:
        print("[warn] maa_py not given -- MAA will not be evaluated.")

    methods = methods or list(TRANSFORM_REGISTRY.keys())
    unknown = [m for m in methods if m not in TRANSFORM_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown method(s) {unknown}. Registered methods: "
            f"{list(TRANSFORM_REGISTRY.keys())}"
        )
    print(f"Evaluating methods: {methods}")

    print("Applying transforms to visible images (p=1.0, force-applied) ...")
    transformed: Dict[str, List[Image.Image]] = {}
    for name in methods:
        fn = TRANSFORM_REGISTRY[name]
        transformed[name] = [fn(img) for img in tqdm(rgb_images, desc=f"transform:{name}")]

    print("Extracting frozen ResNet50 features ...")
    extractor = ResNet50FeatureExtractor(device=device)
    feats_ir = extractor.extract(ir_images)
    feats_rgb = extractor.extract(rgb_images)  # untransformed upper-bound reference
    feats_by_method = {
        name: extractor.extract(imgs) for name, imgs in transformed.items()
    }

    print("Computing FID / MMD against real IR ...")
    rows = [{
        "Method": "RGB (no transform)",
        "FID": compute_fid(feats_rgb, feats_ir),
        "MMD": compute_mmd(feats_rgb, feats_ir),
        "N": len(rgb_images),
    }]
    for name, feats in feats_by_method.items():
        rows.append({
            "Method": name,
            "FID": compute_fid(feats, feats_ir),
            "MMD": compute_mmd(feats, feats_ir),
            "N": len(transformed[name]),
        })
    df = pd.DataFrame(rows).sort_values("FID").reset_index(drop=True)
    csv_path = os.path.join(out_dir, "fid_mmd_summary.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {csv_path}")

    print("Building qualitative panel (identity-matched real IR) ...")
    panel_path = os.path.join(out_dir, "qualitative_panel.png")
    build_and_save_qualitative_panel(
        rgb_paths_all=rgb_paths_all,
        rgb_images=rgb_images,
        sysu_root=sysu_root,
        num_samples=num_samples,
        seed=seed,
        methods=methods,
        out_path=panel_path,
        num_rows=6,
    )
    print(f"Saved: {panel_path}")

    return df


# ==========================================================================
# 8. CLI
# ==========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-contained SYSU-MM01 visible->IR modality-gap analysis "
                     "(FID, MMD, qualitative panel), using your project's "
                     "actual TMS/CAJ/FDA implementations."
    )
    parser.add_argument("--sysu_root", type=str, default="/home/vipin/PhD/datasets/cross_modality/SYSU-MM01/SYSU-MM01",
                         help="SYSU-MM01 root (contains cam1..cam6, exp/train_id.txt, exp/val_id.txt).")
    parser.add_argument("--transform_py", type=str, default="/home/vipin/PhD/models/vireid/data/transform.py",
                         help="Path to transform.py (defines TMS). "
                              "Defaults to ./transform.py or a copy next to this script.")
    parser.add_argument("--caj_py", type=str, default="/home/vipin/PhD/models/vireid/data/caj.py",
                         help="Path to caj.py (defines ChannelExchange). Same default rule as --transform_py.")
    parser.add_argument("--fda_py", type=str, default="/home/vipin/PhD/models/vireid/data/fda.py",
                         help="Path to fda.py (defines FDATransform, sysu_ir_train_paths). Same default rule.")
    parser.add_argument("--rgpr_py", type=str, default="/home/vipin/PhD/models/vireid/data/rgpr.py",
                         help="Path to rgpr.py (defines RandomGrayscalePatchReplacement). Same default rule.")
    parser.add_argument("--maa_py", type=str, default="/home/vipin/PhD/models/vireid/data/maa.py",
                         help="Path to maa.py (defines WeightedGrayscale, ChannelCutMix, SpectrumJitter). "
                              "Same default rule.")
    parser.add_argument("--out_dir", type=str, default="visualizations/modality_gap")
    parser.add_argument("--num_samples", type=int, default=10000,
                         help="Max number of images sampled per domain. "
                              "FID is unreliable at very small N; prefer >=300-500.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", type=str, nargs="*", default=None,
                         help="Subset of registered methods to run "
                              "(default: Grayscale + CAJ + FDA + RGPR + TMS + MAA).")
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    transform_py = resolve_module_path(args.transform_py, "transform.py", script_dir)
    caj_py = resolve_module_path(args.caj_py, "caj.py", script_dir)
    fda_py = resolve_module_path(args.fda_py, "fda.py", script_dir)
    rgpr_py = resolve_module_path(args.rgpr_py, "rgpr.py", script_dir)
    maa_py = resolve_module_path(args.maa_py, "maa.py", script_dir)

    run_analysis(
        sysu_root=args.sysu_root,
        transform_py=transform_py,
        caj_py=caj_py,
        fda_py=fda_py,
        rgpr_py=rgpr_py,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        device=args.device,
        methods=args.methods,
        seed=args.seed,
        maa_py=maa_py,
    )