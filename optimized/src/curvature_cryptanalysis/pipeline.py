"""
Attack implementation on trained CIFAR-10 transformer FFN blocks:

  raw-output queries
      -> shared vector-valued finite-difference stencils
      -> many projected Hessians constructed offline per probe location
      -> multi-restart joint Adam shared-dictionary recovery
      -> optional rank-one pursuit initialization ablation
      -> optional fixed-direction surrogate completion


Scope
    Smooth activations: GELU and SiLU.



Examples

Trained 10 class CIFAR-10 checkpoint:

python -m curvature_cryptanalysis extract \
  --ckpt ./runs/cifar10_gelu/best.pt \
  --out-dir ./runs_unified/cifar10_gelu_T16_P1_seed0 \
  --target-block 3 --T 16 --probe-locations 1 \
  --hessian-mode finite_diff --fd-eps 1e-2 \
  --dict-init random --dict-restarts 3 \
  --dict-steps 3000 --fit-steps 1500 \
  --max-fit-samples 12000 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


SMOOTH_ACTIVATIONS = ("gelu", "silu")

CIFAR10_CLASSES = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}
CIFAR10_CLASS_ORDER = list(CIFAR10_CLASSES.keys())
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465) # Channel means: (Red, Green, Blue)
CIFAR10_STD = (0.2470, 0.2435, 0.2616) # Channel standard deviations: (Red, Green, Blue)



def train(*_args, **_kwargs):
    raise RuntimeError("Checkpoint compatibility shim only.")


def run_extract(*_args, **_kwargs):
    raise RuntimeError("Checkpoint compatibility shim only.")


_main = sys.modules.get("__main__")
if _main is not None:
    setattr(_main, "train", train)
    setattr(_main, "run_extract", run_extract)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_result(out_dir: str | Path, result: Dict, arrays: Optional[Dict[str, np.ndarray]] = None) -> None:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    clean = {k: json_safe(v) for k, v in result.items()}
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    with open(out_dir / "result.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean.keys()))
        writer.writeheader()
        writer.writerow(clean)
    if arrays:
        np.savez(out_dir / "recovery_arrays.npz", **arrays)


def activation_forward(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu":
        return F.gelu(z)
    if activation == "silu":
        return F.silu(z)
    raise ValueError(f"Unsupported activation {activation!r}; use one of {SMOOTH_ACTIVATIONS}.")


def activation_second_derivative(z: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu":
        phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
        return phi * (2.0 - z.square())
    if activation == "silu":
        s = torch.sigmoid(z)
        return s * (1.0 - s) * (2.0 + z * (1.0 - 2.0 * s))
    raise ValueError(f"Unsupported activation {activation!r}; use one of {SMOOTH_ACTIVATIONS}.")


def normalize_columns(U: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return U / (torch.linalg.norm(U, dim=0, keepdim=True) + eps)


def normalize_rows_np(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return U / (np.linalg.norm(U, axis=1, keepdims=True) + eps)


def symmetrize(H: torch.Tensor) -> torch.Tensor:
    return 0.5 * (H + H.transpose(-1, -2))


def finite_difference_query_budget(probe_locations: int, d: int) -> int:
    return int(probe_locations * (2 * d * d + 1))


def resolve_probe_layout(T: int, probe_locations: Optional[int]) -> Tuple[int, int]:
    if T < 1:
        raise ValueError("T must be at least 1")
    P = T if probe_locations is None else int(probe_locations)
    if P < 1:
        raise ValueError("probe_locations must be at least 1")
    if P > T:
        raise ValueError(f"probe_locations={P} cannot exceed T={T}")
    if T % P != 0:
        raise ValueError(
            f"T={T} must be divisible by probe_locations={P}; "
            "T is the total number of Hessian mixtures."
        )
    return P, T // P


def sample_output_projections(k: int, count: int, generator: torch.Generator, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "random":
        return torch.randn(count, k, generator=generator, device=device)
    if mode == "coordinates":
        
        rows: List[torch.Tensor] = []
        remaining = count
        eye = torch.eye(k, device=device)
        while remaining > 0:
            perm = torch.randperm(k, generator=generator, device=device)
            take = min(remaining, k)
            rows.append(eye[perm[:take]])
            remaining -= take
        return torch.cat(rows, dim=0)
    raise ValueError(f"Unsupported projection_mode={mode!r}")


class ResidualFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        if activation not in SMOOTH_ACTIVATIONS:
            raise ValueError(f"This pipeline supports only {SMOOTH_ACTIVATIONS}.")
        self.d_model = d_model
        self.hidden = hidden
        self.activation = activation
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        return activation_forward(self.fc1(x), self.activation)

    def branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.hidden_features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


class ControlledSmoothFFN(ResidualFFN):
    def __init__(self, d: int, m: int, activation: str, seed: int):
        super().__init__(d_model=d, hidden=m, activation=activation, dropout=0.0)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        with torch.no_grad():
            W1 = torch.randn(m, d, generator=gen)
            W1 = F.normalize(W1, dim=1)
            self.fc1.weight.copy_(W1)
            self.fc1.bias.copy_(0.3 * torch.randn(m, generator=gen))
            self.fc2.weight.copy_(torch.randn(d, m, generator=gen) / math.sqrt(m))
            self.fc2.bias.copy_(0.2 * torch.randn(d, generator=gen))


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, d_model: int = 64):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_hidden: int, activation: str, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ResidualFFN(d_model, ffn_hidden, activation, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, need_weights=False)
        x = x + self.drop(a)
        f_in = self.ln2(x)
        return x + self.drop(self.ffn.branch(f_in))


class CifarTinyTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        patch_size: int = 4,
        d_model: int = 64,
        ffn_hidden: int = 128,
        depth: int = 4,
        heads: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation not in SMOOTH_ACTIVATIONS:
            raise ValueError(
                f"Checkpoint activation {activation!r} is outside this smooth-curvature pipeline. "
                f"Use GELU or SiLU."
            )
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.depth = depth
        self.heads = heads
        self.activation = activation
        self.dropout = dropout
        self.patch = PatchEmbed(32, patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.patch.num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, ffn_hidden, activation, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward_features(self, x: torch.Tensor, target_block: Optional[int] = None, return_ffn: bool = False):
        B = x.shape[0]
        x = self.patch(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        info: Dict[str, torch.Tensor] = {}
        for i, blk in enumerate(self.blocks):
            if return_ffn and target_block == i:
                a = blk.ln1(x)
                a, _ = blk.attn(a, a, a, need_weights=False)
                x = x + blk.drop(a)
                f_in = blk.ln2(x)
                hidden = blk.ffn.hidden_features(f_in)
                branch = blk.ffn.branch(f_in)
                x = x + blk.drop(branch)
                info = {
                    "ffn_input": f_in,
                    "ffn_hidden": hidden,
                    "ffn_branch": branch,
                }
            else:
                x = blk(x)
        x = self.norm(x)
        pooled = x[:, 0]
        return (pooled, info) if return_ffn else pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model_from_config(config: Dict) -> CifarTinyTransformer:
    return CifarTinyTransformer(
        num_classes=int(config["num_classes"]),
        patch_size=int(config["patch_size"]),
        d_model=int(config["d_model"]),
        ffn_hidden=int(config["ffn_hidden"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        activation=str(config["activation"]),
        dropout=float(config.get("dropout", 0.0)),
    )


def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[CifarTinyTransformer, Dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if "config" not in ckpt or "model" not in ckpt:
        raise ValueError("Checkpoint must contain 'config' and 'model'.")
    config = dict(ckpt["config"])
    activation = str(config.get("activation", ""))
    if activation not in SMOOTH_ACTIVATIONS:
        raise ValueError(
            f"Checkpoint uses {activation!r}. This pipeline is intentionally restricted to "
            f"smooth activations {SMOOTH_ACTIVATIONS}."
        )
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


#eval
def make_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class RemappedCifar10(Dataset):
    def __init__(self, root: str, train: bool, class_names: Sequence[str]):
        self.base = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=make_eval_transform(),
        )
        class_ids = [CIFAR10_CLASSES[name] for name in class_names]
        self.id_to_new = {old: new for new, old in enumerate(class_ids)}
        self.indices = [i for i, y in enumerate(self.base.targets) if y in self.id_to_new]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        return x, self.id_to_new[int(y)]

    def __getitems__(self, indices: Sequence[int]):
        return [self[i] for i in indices]


def make_eval_loaders(
    data_root: str,
    class_names: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = RemappedCifar10(data_root, True, class_names)
    test_ds = RemappedCifar10(data_root, False, class_names)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, test_loader



# Hessian

@dataclass
class OutputChannel:
    round_decimals: Optional[int] = None
    noise_std: float = 0.0

    def apply(self, y: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.noise_std > 0:
            noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)
            y = y + self.noise_std * noise
        if self.round_decimals is not None:
            scale = float(10 ** self.round_decimals)
            y = torch.round(y * scale) / scale
        return y


@torch.no_grad()
def query_branch_batch(
    branch_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    channel: OutputChannel,
    noise_generator: Optional[torch.Generator],
) -> torch.Tensor:
    y = branch_fn(x)
    if y.ndim != 2:
        raise ValueError(f"branch_fn must map [B,d] to [B,k], received shape {tuple(y.shape)}")
    return channel.apply(y, noise_generator)


@torch.no_grad()
def collect_hessians_finite_diff(
    branch_fn: Callable[[torch.Tensor], torch.Tensor],
    d: int,
    k: int,
    T: int,
    probe_locations: Optional[int],
    projection_mode: str,
    probe_scale: float,
    fd_eps: float,
    seed: int,
    device: torch.device,
    query_batch_size: int,
    channel: OutputChannel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if fd_eps <= 0:
        raise ValueError("fd_eps must be positive")
    P, projections_per_probe = resolve_probe_layout(T, probe_locations)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(seed + 7919)
    eye = torch.eye(d, device=device)
    Hs: List[torch.Tensor] = []
    Xs: List[torch.Tensor] = []
    Cs: List[torch.Tensor] = []
    query_count = 0

    for p in range(P):
        x = probe_scale * torch.randn(d, generator=gen, device=device)
        C = sample_output_projections(
            k, projections_per_probe, gen, device, projection_mode
        )
        H = torch.zeros(projections_per_probe, d, d, device=device)

        y0 = query_branch_batch(branch_fn, x.unsqueeze(0), channel, noise_gen)[0]
        s0 = C @ y0
        query_count += 1

        diag_q = torch.cat([x + fd_eps * eye, x - fd_eps * eye], dim=0)
        diag_y = query_branch_batch(branch_fn, diag_q, channel, noise_gen)
        diag_s = diag_y @ C.T
        plus = diag_s[:d]
        minus = diag_s[d:]
        diagonal = (plus - 2.0 * s0.unsqueeze(0) + minus) / (fd_eps ** 2)
        diag_idx = torch.arange(d, device=device)
        H[:, diag_idx, diag_idx] = diagonal.T
        query_count += 2 * d

        specs: List[Tuple[int, int, int, int]] = []
        queries: List[torch.Tensor] = []
        for i in range(d):
            ei = eye[i]
            for j in range(i + 1, d):
                ej = eye[j]
                start = len(queries)
                queries.extend([
                    x + fd_eps * ei + fd_eps * ej,
                    x + fd_eps * ei - fd_eps * ej,
                    x - fd_eps * ei + fd_eps * ej,
                    x - fd_eps * ei - fd_eps * ej,
                ])
                specs.append((i, j, start, start + 4))

        if queries:
            Q = torch.stack(queries, dim=0)
            scalar_parts: List[torch.Tensor] = []
            for start in range(0, Q.shape[0], query_batch_size):
                qb = Q[start:start + query_batch_size]
                yb = query_branch_batch(branch_fn, qb, channel, noise_gen)
                scalar_parts.append(yb @ C.T)
            S = torch.cat(scalar_parts, dim=0)
            for i, j, start, end in specs:
                vals = S[start:end]
                val = (vals[0] - vals[1] - vals[2] + vals[3]) / (4.0 * fd_eps ** 2)
                H[:, i, j] = val
                H[:, j, i] = val
            query_count += Q.shape[0]

        Hs.append(symmetrize(H))
        Xs.append(x.unsqueeze(0).expand(projections_per_probe, -1).clone())
        Cs.append(C)
        print(
            f"  shared finite-difference probe {p + 1}/{P}: "
            f"constructed {projections_per_probe} Hessians offline"
        )

    expected = finite_difference_query_budget(P, d)
    if query_count != expected:
        raise RuntimeError(f"Query-accounting bug: counted {query_count}, expected {expected}")
    H_all = torch.cat(Hs, dim=0)
    X_all = torch.cat(Xs, dim=0)
    C_all = torch.cat(Cs, dim=0)
    if H_all.shape[0] != T:
        raise RuntimeError(f"Constructed {H_all.shape[0]} Hessians, expected T={T}")
    return H_all, X_all, C_all, query_count


def collect_hessians_autograd(
    branch_fn: Callable[[torch.Tensor], torch.Tensor],
    d: int,
    k: int,
    T: int,
    probe_locations: Optional[int],
    projection_mode: str,
    probe_scale: float,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    P, projections_per_probe = resolve_probe_layout(T, probe_locations)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    Hs, Xs, Cs = [], [], []
    for p in range(P):
        x = probe_scale * torch.randn(d, generator=gen, device=device)
        C = sample_output_projections(
            k, projections_per_probe, gen, device, projection_mode
        )
        for q in range(projections_per_probe):
            c = C[q]

            def scalar_fn(inp: torch.Tensor, projection: torch.Tensor = c) -> torch.Tensor:
                return torch.dot(projection, branch_fn(inp.unsqueeze(0)).squeeze(0))

            H = torch.autograd.functional.hessian(scalar_fn, x)
            Hs.append(symmetrize(H.detach()))
            Xs.append(x.detach().clone())
            Cs.append(c.detach())
        print(
            f"  autograd probe {p + 1}/{P}: "
            f"constructed {projections_per_probe} Hessians"
        )
    return torch.stack(Hs), torch.stack(Xs), torch.stack(Cs)


@torch.no_grad()
def collect_hessians_exact_formula(
    ffn: ResidualFFN,
    T: int,
    probe_locations: Optional[int],
    projection_mode: str,
    probe_scale: float,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    P, projections_per_probe = resolve_probe_layout(T, probe_locations)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    W1 = ffn.fc1.weight.detach().to(device)
    b1 = ffn.fc1.bias.detach().to(device)
    W2 = ffn.fc2.weight.detach().to(device)
    m, d = W1.shape
    k = W2.shape[0]
    X_parts: List[torch.Tensor] = []
    C_parts: List[torch.Tensor] = []
    for p in range(P):
        x = probe_scale * torch.randn(d, generator=gen, device=device)
        X_parts.append(
            x.unsqueeze(0).expand(projections_per_probe, -1).clone()
        )
        C_parts.append(
            sample_output_projections(
                k, projections_per_probe, gen, device, projection_mode
            )
        )
    X = torch.cat(X_parts, dim=0)
    C = torch.cat(C_parts, dim=0)
    pre = X @ W1.T + b1
    beta = (C @ W2) * activation_second_derivative(pre, ffn.activation)
    H = torch.einsum("tm,md,me->tde", beta, W1, W1)
    return symmetrize(H), X, C


# init
@torch.no_grad()
def build_hessian_subspace(H: torch.Tensor, subspace_dim: Optional[int]) -> torch.Tensor:
    H = symmetrize(H)
    T, d, _ = H.shape
    X = H.reshape(T, d * d)
    X = X - X.mean(dim=0, keepdim=True)
    max_rank = min(T - 1 if T > 1 else 1, d * (d + 1) // 2)
    if subspace_dim is None or subspace_dim <= 0:
        L = min(max_rank, 3 * d)
    else:
        L = min(int(subspace_dim), max_rank)
    if L < 1:
        
        B = H[:1]
    else:
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        B = Vh[:L].reshape(L, d, d)
    return symmetrize(B)


@torch.no_grad()
def rank_one_pursuit_initialization(
    H: torch.Tensor,
    m_atoms: int,
    num_combos: int,
    top_keep: int,
    seed: int,
    device: torch.device,
    combo_batch_size: int = 256,
    subspace_dim: Optional[int] = None,
    power_iters: int = 12,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    if num_combos < m_atoms:
        raise ValueError("num_combos must be at least m_atoms")
    H = H.to(device)
    H_scale = torch.sqrt(torch.mean(H.square())) + 1e-12
    Hn = H / H_scale
    basis = build_hessian_subspace(Hn, subspace_dim)
    L, d, _ = basis.shape
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    all_scores: List[torch.Tensor] = []
    all_dirs: List[torch.Tensor] = []
    for start in range(0, num_combos, combo_batch_size):
        B = min(combo_batch_size, num_combos - start)
        coeff = torch.randn(B, L, generator=gen, device=device)
        coeff = F.normalize(coeff, dim=1)
        M = torch.einsum("bl,lij->bij", coeff, basis)
        M = symmetrize(M)

        v = torch.randn(B, d, generator=gen, device=device)
        v = F.normalize(v, dim=1)
        for _ in range(max(power_iters, 1)):
            v = torch.einsum("bij,bj->bi", M, v)
            v = F.normalize(v, dim=1)
        Mv = torch.einsum("bij,bj->bi", M, v)
        rayleigh = torch.sum(v * Mv, dim=1)
        scores = rayleigh.square() / (torch.sum(M.square(), dim=(1, 2)) + 1e-12)
        all_scores.append(scores.cpu())
        all_dirs.append(v.cpu())

    scores = torch.cat(all_scores).numpy()
    dirs = torch.cat(all_dirs).numpy()
    keep_n = min(top_keep, len(scores))
    if keep_n < m_atoms:
        raise ValueError(
            f"pursuit_top_keep must retain at least m_atoms candidates; "
            f"got keep_n={keep_n}, m_atoms={m_atoms}"
        )
    keep = np.argsort(scores)[::-1][:keep_n]
    kept_dirs = normalize_rows_np(dirs[keep])
    kept_scores = scores[keep]

    
    score_norm = kept_scores / (np.max(kept_scores) + 1e-12)
    selected = [0]
    chosen = np.zeros(keep_n, dtype=bool)
    chosen[0] = True
    while len(selected) < m_atoms:
        S = np.abs(kept_dirs @ kept_dirs[selected].T)
        max_cos = np.max(S, axis=1)
        utility = score_norm * np.sqrt(np.clip(1.0 - max_cos, 0.0, 1.0) + 1e-12)
        utility[chosen] = -np.inf
        next_idx = int(np.argmax(utility))
        if not np.isfinite(utility[next_idx]):
            remaining = np.flatnonzero(~chosen)
            if len(remaining) == 0:
                raise RuntimeError("Not enough pursuit candidates to initialize all atoms")
            next_idx = int(remaining[0])
        selected.append(next_idx)
        chosen[next_idx] = True

    U_rows = normalize_rows_np(kept_dirs[np.asarray(selected)])
    U_cols = torch.tensor(U_rows.T, dtype=H.dtype, device=device)
    stats = {
        "pursuit_subspace_dim": int(L),
        "pursuit_combos": int(num_combos),
        "pursuit_top_keep": int(keep_n),
        "pursuit_power_iters": int(power_iters),
        "pursuit_rank1_score_mean_kept": float(np.mean(kept_scores)),
        "pursuit_rank1_score_median_kept": float(np.median(kept_scores)),
        "pursuit_rank1_score_min_kept": float(np.min(kept_scores)),
    }
    return normalize_columns(U_cols), stats

@torch.no_grad()
def initialize_coefficients_ridge(Hn: torch.Tensor, U: torch.Tensor, ridge: float) -> torch.Tensor:
    T, d, _ = Hn.shape
    m = U.shape[1]
    X = Hn.reshape(T, d * d)
    B = torch.einsum("dm,em->mde", U, U).reshape(m, d * d)
    gram = B @ B.T + ridge * torch.eye(m, device=Hn.device, dtype=Hn.dtype)
    rhs = X @ B.T
    return torch.linalg.solve(gram, rhs.T).T


def fit_hessian_dictionary_adam(
    H: torch.Tensor,
    U_init: torch.Tensor,
    steps: int,
    lr: float,
    coeff_reg: float,
    coherence_reg: float,
    coeff_ridge_init: float,
    seed: int,
    device: torch.device,
    verbose_every: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], float]:
    set_seed(seed)
    H = H.to(device)
    H_scale = torch.sqrt(torch.mean(H.square())) + 1e-12
    Hn = H / H_scale
    U0 = normalize_columns(U_init.to(device))
    A0 = initialize_coefficients_ridge(Hn, U0, ridge=coeff_ridge_init)
    U = nn.Parameter(U0.clone())
    A = nn.Parameter(A0.clone())
    opt = torch.optim.Adam([U, A], lr=lr)
    losses: List[float] = []
    m = U.shape[1]

    for step in range(1, steps + 1):
        U_norm = normalize_columns(U)
        H_hat = torch.einsum("tm,dm,em->tde", A, U_norm, U_norm)
        loss_recon = torch.mean((H_hat - Hn).square())
        gram = U_norm.T @ U_norm
        offdiag = gram - torch.eye(m, device=device, dtype=gram.dtype)
        loss = (
            loss_recon
            + coeff_reg * torch.mean(A.square())
            + coherence_reg * torch.mean(offdiag.square())
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            U.copy_(normalize_columns(U))
        losses.append(float(loss_recon.detach().cpu()))
        if verbose_every and (step % verbose_every == 0 or step == steps):
            print(f"  dictionary step {step:05d}/{steps} | recon {loss_recon.item():.6e}")

    return normalize_columns(U.detach()), A.detach(), losses, float(H_scale.detach().cpu())

def evaluate_direction_recovery(U_hat_cols: torch.Tensor, W1_true: torch.Tensor) -> Dict:
    U = U_hat_cols.detach().cpu().numpy()
    W = W1_true.detach().cpu().numpy()
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    S = np.abs(W @ U)
    row_ind, col_ind = linear_sum_assignment(-S)
    matched = S[row_ind, col_ind]
    return {
        "dir_matched_mean": float(np.mean(matched)),
        "dir_matched_median": float(np.median(matched)),
        "dir_matched_min": float(np.min(matched)),
        "dir_matched_max": float(np.max(matched)),
        "dir_frac_gt_090": float(np.mean(matched >= 0.90)),
        "match_row_ind": row_ind,
        "match_col_ind": col_ind,
        "matched_scores": matched,
    }


def random_direction_baseline(W1_true: torch.Tensor, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    W = W1_true.detach().cpu().numpy()
    m, d = W.shape
    U = rng.normal(size=(d, m))
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    S = np.abs(W @ U)
    rows, cols = linear_sum_assignment(-S)
    matched = S[rows, cols]
    return {
        "random_dir_matched_mean": float(np.mean(matched)),
        "random_dir_matched_median": float(np.median(matched)),
        "random_dir_frac_gt_090": float(np.mean(matched >= 0.90)),
    }


def branch_metrics(y: torch.Tensor, y_hat: torch.Tensor) -> Dict:
    y = y.float()
    y_hat = y_hat.float()
    residual = torch.sum((y - y_hat).square())
    centered = y - y.mean(dim=0, keepdim=True)
    total = torch.sum(centered.square())
    mse = torch.mean((y - y_hat).square())
    rel_rmse = torch.sqrt(torch.mean((y - y_hat).square())) / (torch.sqrt(torch.mean(y.square())) + 1e-12)
    return {
        "branch_mse": float(mse.cpu()),
        "branch_r2": float((1.0 - residual / (total + 1e-12)).cpu()),
        "branch_rel_rmse": float(rel_rmse.cpu()),
    }


class ExtractedFFN(nn.Module):
    def __init__(self, W_dirs_rows: torch.Tensor, d_model: int, activation: str):
        super().__init__()
        self.register_buffer("W_dirs", F.normalize(W_dirs_rows.detach().clone(), dim=1))
        m, d = self.W_dirs.shape
        if d != d_model:
            raise ValueError("Direction dimension and d_model do not match")
        self.activation = activation
        self.scales = nn.Parameter(torch.ones(m))  #signed
        self.b1 = nn.Parameter(torch.zeros(m))
        self.fc2 = nn.Linear(m, d_model)
        nn.init.normal_(self.fc2.weight, std=1.0 / math.sqrt(m))
        nn.init.zeros_(self.fc2.bias)

    def effective_W1(self) -> torch.Tensor:
        return self.W_dirs * self.scales.unsqueeze(1)

    def hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.effective_W1().T + self.b1
        return activation_forward(z, self.activation)

    def branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.hidden_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


def fit_extracted_ffn(
    W_dirs_rows: torch.Tensor,
    X_train: torch.Tensor,
    Y_branch: torch.Tensor,
    activation: str,
    fit_steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tuple[ExtractedFFN, List[float]]:
    set_seed(seed)
    X_train = X_train.to(device)
    Y_branch = Y_branch.to(device)
    model = ExtractedFFN(W_dirs_rows.to(device), X_train.shape[1], activation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n = X_train.shape[0]
    losses: List[float] = []
    for step in range(1, fit_steps + 1):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=gen, device=device)
        loss = F.mse_loss(model.branch(X_train[idx]), Y_branch[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % 250 == 0 or step == fit_steps:
            print(f"  surrogate step {step:05d}/{fit_steps} | branch mse {loss.item():.6e}")
    return model.eval(), losses


@torch.no_grad()
def collect_ffn_pairs(
    model: CifarTinyTransformer,
    loader: DataLoader,
    target_block: int,
    device: torch.device,
    max_samples: int,
    channel: OutputChannel,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval().to(device)
    Xs, Ys = [], []
    count = 0
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(seed)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _, info = model.forward_features(x, target_block=target_block, return_ffn=True)
        f_in = info["ffn_input"].reshape(-1, model.d_model)
        branch = info["ffn_branch"].reshape(-1, model.d_model)
        branch = channel.apply(branch, noise_gen)
        Xs.append(f_in.cpu())
        Ys.append(branch.cpu())
        count += f_in.shape[0]
        if max_samples and count >= max_samples:
            break
    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)
    if max_samples and X.shape[0] > max_samples:
        X = X[:max_samples]
        Y = Y[:max_samples]
    return X, Y


def replace_target_ffn(victim: CifarTinyTransformer, extracted: ExtractedFFN, target_block: int) -> CifarTinyTransformer:
    surrogate = copy.deepcopy(victim).cpu().eval()
    surrogate.blocks[target_block].ffn = extracted.cpu().eval()
    return surrogate


@torch.no_grad()
def evaluate_full_model_fidelity(
    victim: CifarTinyTransformer,
    surrogate: CifarTinyTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Dict:
    victim = victim.to(device).eval()
    surrogate = surrogate.to(device).eval()
    total = victim_correct = surrogate_correct = agree = 0
    kl_sum = mse_sum = 0.0
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        lv = victim(x)
        ls = surrogate(x)
        pv = lv.argmax(dim=1)
        ps = ls.argmax(dim=1)
        total += y.numel()
        victim_correct += int((pv == y).sum())
        surrogate_correct += int((ps == y).sum())
        agree += int((pv == ps).sum())
        p = F.softmax(lv, dim=1)
        kl = torch.sum(p * (F.log_softmax(lv, dim=1) - F.log_softmax(ls, dim=1)), dim=1)
        kl_sum += float(kl.sum())
        lvc = lv - lv.mean(dim=1, keepdim=True)
        lsc = ls - ls.mean(dim=1, keepdim=True)
        mse_sum += float(torch.sum((lvc - lsc).square(), dim=1).sum())
    return {
        "victim_acc": victim_correct / max(total, 1),
        "surrogate_acc": surrogate_correct / max(total, 1),
        "agreement": agree / max(total, 1),
        "kl": kl_sum / max(total, 1),
        "centered_logit_mse": mse_sum / max(total, 1),
        "eval_examples": int(total),
    }


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    X = X.float() - X.float().mean(dim=0, keepdim=True)
    Y = Y.float() - Y.float().mean(dim=0, keepdim=True)
    xy = torch.linalg.norm(X.T @ Y, ord="fro").square()
    xx = torch.linalg.norm(X.T @ X, ord="fro").square()
    yy = torch.linalg.norm(Y.T @ Y, ord="fro").square()
    return float((xy / (torch.sqrt(xx * yy) + eps)).cpu())


@torch.no_grad()
def evaluate_representation(
    victim: CifarTinyTransformer,
    surrogate: CifarTinyTransformer,
    loader: DataLoader,
    target_block: int,
    device: torch.device,
    max_samples: int,
) -> Dict:
    Zv_all, Zs_all = [], []
    count = 0
    victim = victim.to(device).eval()
    surrogate = surrogate.to(device).eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _, iv = victim.forward_features(x, target_block=target_block, return_ffn=True)
        _, is_ = surrogate.forward_features(x, target_block=target_block, return_ffn=True)
        Zv = iv["ffn_hidden"].reshape(-1, victim.ffn_hidden).cpu()
        Zs = is_["ffn_hidden"].reshape(-1, surrogate.ffn_hidden).cpu()
        Zv_all.append(Zv)
        Zs_all.append(Zs)
        count += Zv.shape[0]
        if max_samples and count >= max_samples:
            break
    Zv = torch.cat(Zv_all)[:max_samples or None]
    Zs = torch.cat(Zs_all)[:max_samples or None]
    Zv_std = (Zv - Zv.mean(0, keepdim=True)) / (Zv.std(0, keepdim=True) + 1e-8)
    Zs_std = (Zs - Zs.mean(0, keepdim=True)) / (Zs.std(0, keepdim=True) + 1e-8)
    C = torch.abs((Zv_std.T @ Zs_std) / max(Zv.shape[0] - 1, 1)).numpy()
    rows, cols = linear_sum_assignment(-C)
    matched = C[rows, cols]
    return {
        "ffn_cka": linear_cka(Zv, Zs),
        "feature_corr_matched_mean": float(np.mean(matched)),
        "feature_corr_matched_median": float(np.median(matched)),
        "feature_corr_frac_gt_090": float(np.mean(matched >= 0.90)),
        "rep_samples": int(Zv.shape[0]),
    }

# runner
def run_unified_attack(
    ffn: ResidualFFN,
    T: int,
    probe_locations: Optional[int],
    projection_mode: str,
    hessian_mode: str,
    probe_scale: float,
    fd_eps: float,
    query_batch_size: int,
    channel: OutputChannel,
    pursuit_combos: int,
    pursuit_top_keep: int,
    pursuit_batch_size: int,
    pursuit_subspace_dim: Optional[int],
    pursuit_power_iters: int,
    dict_steps: int,
    dict_lr: float,
    coeff_reg: float,
    coherence_reg: float,
    coeff_ridge_init: float,
    dict_init: str,
    dict_restarts: int,
    seed: int,
    device: torch.device,
    verbose_every: int,
) -> Tuple[Dict, Dict[str, np.ndarray], torch.Tensor]:
    if dict_restarts < 1:
        raise ValueError("dict_restarts must be at least 1")
    if dict_init not in {"random", "pursuit"}:
        raise ValueError(f"Unsupported dict_init={dict_init!r}")

    ffn = ffn.to(device).eval()
    W1_true = ffn.fc1.weight.detach().to(device)
    m, d = W1_true.shape
    k = ffn.fc2.weight.shape[0]
    P, projections_per_probe = resolve_probe_layout(T, probe_locations)
    if hessian_mode == "finite_diff":
        shared_budget = finite_difference_query_budget(P, d)
        unshared_budget = finite_difference_query_budget(T, d)
        print(
            "Structural vector-query budget: "
            f"{shared_budget:,} = P({P}) * (2*d^2+1); "
            f"unshared baseline={unshared_budget:,}; "
            f"reduction={projections_per_probe:.1f}x"
        )
    if projections_per_probe > k:
        print(
            "Warning: projections_per_probe exceeds output dimension k; "
            "additional projections are linearly dependent in the exact "
            "vector-output model."
        )

    t0 = time.time()
    if hessian_mode == "finite_diff":
        H, X_probe, C_probe, query_count = collect_hessians_finite_diff(
            ffn.branch, d, k, T, P, projection_mode, probe_scale, fd_eps,
            seed, device, query_batch_size, channel,
        )
    elif hessian_mode == "autograd":
        H, X_probe, C_probe = collect_hessians_autograd(
            ffn.branch, d, k, T, P, projection_mode, probe_scale, seed, device,
        )
        query_count = 0
    elif hessian_mode == "exact":
        if channel.noise_std > 0 or channel.round_decimals is not None:
            raise ValueError("Output noise/rounding applies only to finite_diff mode.")
        H, X_probe, C_probe = collect_hessians_exact_formula(
            ffn, T, P, projection_mode, probe_scale, seed, device
        )
        query_count = 0
    else:
        raise ValueError(f"Unknown hessian_mode {hessian_mode!r}")
    hessian_seconds = time.time() - t0

    pursuit_seconds = 0.0
    pursuit_stats: Dict = {}
    pursuit_metrics: Dict = {}
    U_pursuit: Optional[torch.Tensor] = None

    if dict_init == "pursuit":
        t0 = time.time()
        U_pursuit, pursuit_stats = rank_one_pursuit_initialization(
            H,
            m_atoms=m,
            num_combos=pursuit_combos,
            top_keep=pursuit_top_keep,
            seed=seed + 1000,
            device=device,
            combo_batch_size=pursuit_batch_size,
            subspace_dim=pursuit_subspace_dim,
            power_iters=pursuit_power_iters,
        )
        pursuit_seconds = time.time() - t0
        pursuit_metrics = evaluate_direction_recovery(U_pursuit, W1_true)

    restart_records = []
    best = None
    dictionary_seconds = 0.0

    for restart in range(dict_restarts):
        restart_seed = seed + 2000 + restart
        if dict_init == "random":
            gen = torch.Generator(device=device)
            gen.manual_seed(restart_seed)
            U_init = torch.randn(d, m, generator=gen, device=device)
            U_init = normalize_columns(U_init)
        else:
            assert U_pursuit is not None
            U_init = U_pursuit.clone()
            if restart > 0:
                gen = torch.Generator(device=device)
                gen.manual_seed(restart_seed)
                U_init = normalize_columns(
                    U_init + 0.02 * torch.randn(
                        U_init.shape,
                        generator=gen,
                        device=device,
                        dtype=U_init.dtype,
                    )
                )

        print(
            f"Dictionary restart {restart + 1}/{dict_restarts} "
            f"(init={dict_init}, seed={restart_seed})"
        )
        t0 = time.time()
        U_r, A_r, losses_r, H_scale_r = fit_hessian_dictionary_adam(
            H,
            U_init=U_init,
            steps=dict_steps,
            lr=dict_lr,
            coeff_reg=coeff_reg,
            coherence_reg=coherence_reg,
            coeff_ridge_init=coeff_ridge_init,
            seed=restart_seed,
            device=device,
            verbose_every=verbose_every,
        )
        elapsed = time.time() - t0
        dictionary_seconds += elapsed
        final_loss = float(losses_r[-1])
        restart_records.append({
            "restart": restart,
            "seed": restart_seed,
            "final_recon_loss": final_loss,
            "seconds": elapsed,
        })
        if best is None or final_loss < best["loss"]:
            best = {
                "loss": final_loss,
                "restart": restart,
                "U": U_r,
                "A": A_r,
                "losses": losses_r,
                "H_scale": H_scale_r,
            }

    assert best is not None
    U_hat = best["U"]
    A_hat = best["A"]
    losses = best["losses"]
    H_scale = best["H_scale"]
    final_metrics = evaluate_direction_recovery(U_hat, W1_true)
    random_metrics = random_direction_baseline(W1_true, seed + 3000)

    result: Dict = {
        "activation": ffn.activation,
        "d": d,
        "m": m,
        "k": k,
        "T": T,
        "probe_locations": P,
        "projections_per_probe": projections_per_probe,
        "projection_mode": projection_mode,
        "projection_rank_upper_bound_per_probe": min(projections_per_probe, k),
        "hessian_mode": hessian_mode,
        "probe_scale": probe_scale,
        "fd_eps": fd_eps,
        "round_decimals": (
            channel.round_decimals
            if channel.round_decimals is not None
            else "full"
        ),
        "noise_std": channel.noise_std,
        "structural_query_count": int(query_count),
        "equivalent_fd_query_count": finite_difference_query_budget(P, d),
        "unshared_fd_query_count": finite_difference_query_budget(T, d),
        "queries_per_probe_stencil": finite_difference_query_budget(1, d),
        "structural_query_reduction_factor": float(projections_per_probe),
        "dict_init": dict_init,
        "dict_restarts": int(dict_restarts),
        "dict_selected_restart": int(best["restart"]),
        "dir_matched_mean": final_metrics["dir_matched_mean"],
        "dir_matched_median": final_metrics["dir_matched_median"],
        "dir_matched_min": final_metrics["dir_matched_min"],
        "dir_frac_gt_090": final_metrics["dir_frac_gt_090"],
        **random_metrics,
        "dict_steps": dict_steps,
        "dict_lr": dict_lr,
        "dict_final_recon_loss": losses[-1],
        "hessian_rms_scale": H_scale,
        "hessian_seconds": hessian_seconds,
        "pursuit_seconds": pursuit_seconds,
        "dictionary_seconds": dictionary_seconds,
        "seed": seed,
    }

    if dict_init == "pursuit":
        result.update({
            "pursuit_dir_matched_mean": pursuit_metrics["dir_matched_mean"],
            "pursuit_dir_matched_median": pursuit_metrics["dir_matched_median"],
            "pursuit_dir_frac_gt_090": pursuit_metrics["dir_frac_gt_090"],
            **pursuit_stats,
        })

    arrays = {
        "H": H.detach().cpu().numpy(),
        "X_probe": X_probe.detach().cpu().numpy(),
        "C_probe": C_probe.detach().cpu().numpy(),
        "probe_group_index": np.repeat(
            np.arange(P, dtype=np.int64), projections_per_probe
        ),
        "X_unique_probe": X_probe[::projections_per_probe].detach().cpu().numpy(),
        "U_hat": U_hat.detach().cpu().numpy(),
        "A_hat": A_hat.detach().cpu().numpy(),
        "W1_true": W1_true.detach().cpu().numpy(),
        "dict_losses": np.asarray(losses),
        "matched_scores": final_metrics["matched_scores"],
        "restart_final_losses": np.asarray(
            [r["final_recon_loss"] for r in restart_records],
            dtype=np.float64,
        ),
        "restart_seeds": np.asarray(
            [r["seed"] for r in restart_records],
            dtype=np.int64,
        ),
    }
    if U_pursuit is not None:
        arrays["U_pursuit"] = U_pursuit.detach().cpu().numpy()

    result["dict_restart_summary"] = json.dumps(restart_records)
    return result, arrays, U_hat

def controlled(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=not args.nondeterministic)
    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    victim = ControlledSmoothFFN(args.d, args.m, args.activation, args.seed).to(device).eval()
    channel = OutputChannel(args.round_decimals, args.noise_std)

    print("=" * 80)
    print("Unified controlled smooth-FFN attack")
    P, Q = resolve_probe_layout(args.T, args.probe_locations)
    print(
        f"activation={args.activation} d={args.d} m={args.m} "
        f"T={args.T} P={P} Q={Q} mode={args.hessian_mode}"
    )
    print("=" * 80)

    result, arrays, U_hat = run_unified_attack(
        victim,
        T=args.T,
        probe_locations=args.probe_locations,
        projection_mode=args.projection_mode,
        hessian_mode=args.hessian_mode,
        probe_scale=args.probe_scale,
        fd_eps=args.fd_eps,
        query_batch_size=args.query_batch_size,
        channel=channel,
        pursuit_combos=args.pursuit_combos,
        pursuit_top_keep=args.pursuit_top_keep,
        pursuit_batch_size=args.pursuit_batch_size,
        pursuit_subspace_dim=args.pursuit_subspace_dim,
        pursuit_power_iters=args.pursuit_power_iters,
        dict_steps=args.dict_steps,
        dict_lr=args.dict_lr,
        coeff_reg=args.coeff_reg,
        coherence_reg=args.coherence_reg,
        coeff_ridge_init=args.coeff_ridge_init,
        dict_init=args.dict_init,
        dict_restarts=args.dict_restarts,
        seed=args.seed,
        device=device,
        verbose_every=args.verbose_every,
    )
    result["experiment"] = "controlled_residual"

    if args.fit_steps > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + 4000)
        X_fit = torch.randn(args.fit_samples, args.d, generator=gen, device=device)
        with torch.no_grad():
            completion_noise_gen = torch.Generator(device=device)
            completion_noise_gen.manual_seed(args.seed + 4500)
            Y_fit = channel.apply(victim.branch(X_fit), completion_noise_gen)
        extracted, fit_losses = fit_extracted_ffn(
            U_hat.T.cpu(), X_fit.cpu(), Y_fit.cpu(), args.activation,
            args.fit_steps, args.fit_lr, args.fit_batch_size, device, args.seed + 5000,
        )
        X_test = torch.randn(args.test_samples, args.d, generator=gen, device=device)
        with torch.no_grad():
            Y_test = victim.branch(X_test)
            Y_hat = extracted.branch(X_test)
        result.update(branch_metrics(Y_test.cpu(), Y_hat.cpu()))
        result["completion_query_count"] = int(args.fit_samples)
        result["total_query_count"] = int(result["structural_query_count"] + args.fit_samples)
        result["fit_steps"] = args.fit_steps
        result["fit_final_loss"] = fit_losses[-1]
        arrays["surrogate_scales"] = extracted.scales.detach().cpu().numpy()
        arrays["surrogate_b1"] = extracted.b1.detach().cpu().numpy()
        arrays["surrogate_W2"] = extracted.fc2.weight.detach().cpu().numpy()
        arrays["surrogate_b2"] = extracted.fc2.bias.detach().cpu().numpy()
        torch.save({
            "state_dict": extracted.cpu().state_dict(),
            "activation": args.activation,
            "d": args.d,
            "m": args.m,
        }, Path(args.out_dir) / "extracted_ffn.pt")
    else:
        result["completion_query_count"] = 0
        result["total_query_count"] = int(result["structural_query_count"])

    save_result(args.out_dir, result, arrays)
    print(json.dumps({k: v for k, v in result.items() if isinstance(v, (int, float, str))}, indent=2))


def extract(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=not args.nondeterministic)
    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    victim, config = load_checkpoint(args.ckpt, device)
    target_block = args.target_block if args.target_block >= 0 else victim.depth - 1
    if not (0 <= target_block < victim.depth):
        raise ValueError(f"target_block must be in [0,{victim.depth - 1}]")
    target_ffn = victim.blocks[target_block].ffn
    class_names = config.get("class_names", CIFAR10_CLASS_ORDER[: int(config["num_classes"])])
    train_loader, test_loader = make_eval_loaders(
        args.data_root, class_names, args.batch_size, args.num_workers,
    )
    channel = OutputChannel(args.round_decimals, args.noise_std)

    print("=" * 80)
    print("Unified trained-transformer FFN attack")
    print(f"checkpoint={args.ckpt}")
    P, Q = resolve_probe_layout(args.T, args.probe_locations)
    print(
        f"activation={target_ffn.activation} block={target_block} "
        f"T={args.T} P={P} Q={Q} mode={args.hessian_mode}"
    )
    print("=" * 80)

    result, arrays, U_hat = run_unified_attack(
        target_ffn,
        T=args.T,
        probe_locations=args.probe_locations,
        projection_mode=args.projection_mode,
        hessian_mode=args.hessian_mode,
        probe_scale=args.probe_scale,
        fd_eps=args.fd_eps,
        query_batch_size=args.query_batch_size,
        channel=channel,
        pursuit_combos=args.pursuit_combos,
        pursuit_top_keep=args.pursuit_top_keep,
        pursuit_batch_size=args.pursuit_batch_size,
        pursuit_subspace_dim=args.pursuit_subspace_dim,
        pursuit_power_iters=args.pursuit_power_iters,
        dict_steps=args.dict_steps,
        dict_lr=args.dict_lr,
        coeff_reg=args.coeff_reg,
        coherence_reg=args.coherence_reg,
        coeff_ridge_init=args.coeff_ridge_init,
        dict_init=args.dict_init,
        dict_restarts=args.dict_restarts,
        seed=args.seed,
        device=device,
        verbose_every=args.verbose_every,
    )
    result.update({
        "experiment": "trained_cifar10_transformer",
        # Record only the filename so saved artifacts do not disclose a local path.
        "ckpt": Path(args.ckpt).name,
        "target_block": target_block,
        "num_classes": int(config["num_classes"]),
        "class_names": ",".join(class_names),
    })

    if args.fit_steps > 0:
        X_fit, Y_fit = collect_ffn_pairs(
            victim, train_loader, target_block, device, args.max_fit_samples,
            channel=channel, seed=args.seed + 4500,
        )
        extracted, fit_losses = fit_extracted_ffn(
            U_hat.T.cpu(), X_fit, Y_fit, target_ffn.activation,
            args.fit_steps, args.fit_lr, args.fit_batch_size, device, args.seed + 5000,
        )
        surrogate = replace_target_ffn(victim.cpu(), extracted.cpu(), target_block).to(device).eval()
        fidelity = evaluate_full_model_fidelity(
            victim, surrogate, test_loader, device, args.eval_max_batches,
        )
        reps = evaluate_representation(
            victim, surrogate, test_loader, target_block, device, args.rep_max_samples,
        )
        result.update(fidelity)
        result.update(reps)
        result["completion_query_count"] = int(X_fit.shape[0])
        result["total_query_count"] = int(result["structural_query_count"] + X_fit.shape[0])
        result["fit_steps"] = args.fit_steps
        result["fit_final_loss"] = fit_losses[-1]
        torch.save({
            "surrogate_model": surrogate.cpu().state_dict(),
            "victim_config": config,
            "target_block": target_block,
            "result": {k: json_safe(v) for k, v in result.items()},
        }, Path(args.out_dir) / "surrogate_replaced_block.pt")
        arrays["surrogate_scales"] = extracted.scales.detach().cpu().numpy()
        arrays["surrogate_b1"] = extracted.b1.detach().cpu().numpy()
        arrays["surrogate_W2"] = extracted.fc2.weight.detach().cpu().numpy()
        arrays["surrogate_b2"] = extracted.fc2.bias.detach().cpu().numpy()
    else:
        result["completion_query_count"] = 0
        result["total_query_count"] = int(result["structural_query_count"])

    save_result(args.out_dir, result, arrays)
    print(json.dumps({k: v for k, v in result.items() if isinstance(v, (int, float, str))}, indent=2))

def add_attack_args(p: argparse.ArgumentParser, *, default_fd_eps: float, default_T: int) -> None:
    p.add_argument(
        "--T", type=int, default=default_T,
        help="Total number of scalar projected Hessian mixtures (T=P*Q).",
    )
    p.add_argument(
        "--probe-locations", type=int, default=None,
        help=(
            "Number P of distinct chosen-input probe locations. Each location "
            "uses one shared vector-valued stencil; T must be divisible by P. "
            "Omit to preserve legacy P=T, Q=1 behavior."
        ),
    )
    p.add_argument(
        "--projection-mode", choices=["random", "coordinates"], default="random",
        help="Offline output projections applied to each cached vector stencil.",
    )
    p.add_argument("--hessian-mode", choices=["finite_diff", "exact", "autograd"], default="finite_diff")
    p.add_argument("--probe-scale", type=float, default=1.0)
    p.add_argument("--fd-eps", type=float, default=default_fd_eps)
    p.add_argument("--query-batch-size", type=int, default=1024)
    p.add_argument("--round-decimals", type=int, default=None)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--pursuit-combos", type=int, default=10000)
    p.add_argument("--pursuit-top-keep", type=int, default=3000)
    p.add_argument("--pursuit-batch-size", type=int, default=256)
    p.add_argument("--pursuit-subspace-dim", type=int, default=0, help="0 selects min(T-1,3d)")
    p.add_argument("--pursuit-power-iters", type=int, default=12)
    p.add_argument("--dict-steps", type=int, default=3000)
    p.add_argument("--dict-lr", type=float, default=2e-2)
    p.add_argument("--coeff-reg", type=float, default=1e-5)
    p.add_argument("--coherence-reg", type=float, default=1e-4)
    p.add_argument("--coeff-ridge-init", type=float, default=1e-5)
    p.add_argument("--dict-init", choices=["random", "pursuit"], default="random",
                   help="Dictionary initialization. Random is the default for low-T extraction.")
    p.add_argument("--dict-restarts", type=int, default=3,
                   help="Number of independent Adam dictionary restarts; the lowest reconstruction loss is selected.")
    p.add_argument("--fit-steps", type=int, default=1500, help="0 disables functional completion")
    p.add_argument("--fit-lr", type=float, default=1e-3)
    p.add_argument("--fit-batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--nondeterministic", action="store_true")
    p.add_argument("--verbose-every", type=int, default=250)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified smooth-FFN curvature extraction with random multi-restart Adam dictionary recovery"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("controlled", help="run the unified attack on a controlled residual smooth FFN")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--activation", choices=list(SMOOTH_ACTIVATIONS), default="gelu")
    c.add_argument("--d", type=int, default=64)
    c.add_argument("--m", type=int, default=128)
    c.add_argument("--fit-samples", type=int, default=2000)
    c.add_argument("--test-samples", type=int, default=3000)
    add_attack_args(c, default_fd_eps=1e-2, default_T=32)
    c.set_defaults(func=controlled)

    e = sub.add_parser("extract", help="run the unified attack on a trained CIFAR transformer FFN")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--data-root", default="./data")
    e.add_argument("--target-block", type=int, default=-1)
    e.add_argument("--max-fit-samples", type=int, default=12000)
    e.add_argument("--batch-size", type=int, default=128)
    e.add_argument("--num-workers", type=int, default=4)
    e.add_argument("--eval-max-batches", type=int, default=0)
    e.add_argument("--rep-max-samples", type=int, default=20000)
    add_attack_args(e, default_fd_eps=1e-2, default_T=16)
    e.set_defaults(func=extract)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "pursuit_subspace_dim", 0) == 0:
        args.pursuit_subspace_dim = None
    args.func(args)

# main
if __name__ == "__main__":
    main()
