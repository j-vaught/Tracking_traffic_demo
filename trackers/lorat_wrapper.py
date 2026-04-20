"""
Thin, batched wrapper around LoRAT-g-378 for multi-object single-object-tracking.

The upstream LoRAT codebase is evaluator-shaped (torchrun / distributed / VOT
toolkit), but the actual per-frame work is small: crop template and search
regions, run one ViT forward, post-process. This module replays that logic
with a clean Python API so we can drive N concurrent tracks per forward pass.

Usage:
    from lorat_wrapper import BatchedLoRAT
    engine = BatchedLoRAT(
        weights="/abs/path/weights/lorat/giant-378.bin",
        device="cuda:0", dtype=torch.float16, max_batch=32)
    tid = engine.init(frame_bgr, [x1, y1, x2, y2])   # returns track id
    out = engine.track(frame_bgr)                    # advance all alive tracks
    # out = {track_id: (bbox_xyxy_float64, score_float)}
    engine.kill(tid)                                 # drop a track
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Make the LoRAT trackit package importable as a side-loaded module.
_LORAT_REPO = Path(__file__).resolve().parent / "lorat"
if str(_LORAT_REPO) not in sys.path:
    sys.path.insert(0, str(_LORAT_REPO))

from safetensors.torch import load_file as _safetensors_load

# Reuse LoRAT's own cropping + normalization utilities so pixel-level semantics
# match exactly what the model was trained with.
from trackit.core.utils.siamfc_cropping import (  # noqa: E402
    apply_siamfc_cropping,
    apply_siamfc_cropping_to_boxes,
    reverse_siamfc_cropping_params,
    get_siamfc_cropping_params,
)
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform  # noqa: E402
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_, bbox_clip_to_image_boundary  # noqa: E402
from trackit.models.backbone.dinov2.builder import build_dino_v2_backbone  # noqa: E402
from trackit.models.methods.LoRAT.lorat import LoRAT_DINOv2  # noqa: E402
from trackit.models.methods.LoRAT.funcs.vit_lora_utils import enable_lora_  # noqa: E402
from trackit.models.methods.LoRAT.funcs.vit_backbone_freeze import freeze_vit_backbone_  # noqa: E402


# Variant configs — pulled from config/LoRAT/_mixin.
_VARIANTS = {
    "base-224":  dict(vit="ViT-B/14", drop_path=0.1,
                      template_size=(112, 112), template_feat=(8, 8),
                      search_size=(224, 224), search_feat=(16, 16),
                      search_area_factor=4.0),
    "base-378":  dict(vit="ViT-B/14", drop_path=0.1,
                      template_size=(196, 196), template_feat=(14, 14),
                      search_size=(378, 378), search_feat=(27, 27),
                      search_area_factor=5.0),
    "large-224": dict(vit="ViT-L/14", drop_path=0.1,
                      template_size=(112, 112), template_feat=(8, 8),
                      search_size=(224, 224), search_feat=(16, 16),
                      search_area_factor=4.0),
    "large-378": dict(vit="ViT-L/14", drop_path=0.1,
                      template_size=(196, 196), template_feat=(14, 14),
                      search_size=(378, 378), search_feat=(27, 27),
                      search_area_factor=5.0),
    "giant-224": dict(vit="ViT-g/14", drop_path=0.4,
                      template_size=(112, 112), template_feat=(8, 8),
                      search_size=(224, 224), search_feat=(16, 16),
                      search_area_factor=4.0),
    "giant-378": dict(vit="ViT-g/14", drop_path=0.4,
                      template_size=(196, 196), template_feat=(14, 14),
                      search_size=(378, 378), search_feat=(27, 27),
                      search_area_factor=5.0),
}
_TEMPLATE_AREA_FACTOR = 2.0
_LORA_R = 64
_LORA_ALPHA = 64
_NORM = "imagenet"


def _xyxy_to_center_wh(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)


def _bbox_is_valid(box):
    return box[2] > box[0] and box[3] > box[1]


def _score_map_to_boxes(score_map, bbox_map, search_size):
    """Replicates PostProcessing_BoxWithScoreMap.__call__ but torch-native."""
    # score_map: (N, H, W) logits ; bbox_map: (N, H, W, 4) already in [0,1]
    N, H, W = score_map.shape
    probs = score_map.float().sigmoid().view(N, H * W)
    conf, idx = probs.max(dim=1, keepdim=True)         # (N, 1)
    conf = conf.squeeze(1)
    boxes = bbox_map.view(N, H * W, 4).float()
    boxes = torch.gather(boxes, 1, idx.unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)  # (N, 4)
    # Coords are in [0,1]; scale to search-region pixels
    scale = torch.tensor([search_size[0], search_size[1], search_size[0], search_size[1]],
                         dtype=boxes.dtype, device=boxes.device)
    return boxes * scale, conf


# ---------------------------------------------------------------------------
class _Track:
    __slots__ = ("id", "z", "z_mean", "z_mask", "last_box", "alive")

    def __init__(self, tid: int, z: torch.Tensor, z_mean: torch.Tensor,
                 z_mask: torch.Tensor, init_box: np.ndarray):
        self.id = tid
        self.z = z              # (3, Ht, Wt) normalized template tensor on device
        self.z_mean = z_mean    # (3,) mean color for search-region padding
        self.z_mask = z_mask    # (Ht_feat, Wt_feat) int64 mask on device
        self.last_box = init_box.astype(np.float64)
        self.alive = True


class BatchedLoRAT:
    def __init__(self, weights: str, variant: str = "large-224",
                 device: str | torch.device = "cuda:0",
                 dtype: torch.dtype = torch.float16, max_batch: int = 32,
                 interpolation_mode: str = "bilinear",
                 interpolation_align_corners: bool = False):
        if variant not in _VARIANTS:
            raise ValueError(f"unknown variant {variant}; choose from {list(_VARIANTS)}")
        cfg = _VARIANTS[variant]
        self.variant = variant
        self.template_size = cfg["template_size"]
        self.template_feat = cfg["template_feat"]
        self.search_size = cfg["search_size"]
        self.search_feat = cfg["search_feat"]
        self.search_area_factor = cfg["search_area_factor"]
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch = max_batch
        self.interp_mode = interpolation_mode
        self.interp_align = interpolation_align_corners

        # Build DINOv2 backbone (downloads once from FB public URL).
        vit = build_dino_v2_backbone(cfg["vit"], load_pretrained=True,
                                     drop_path_rate=cfg["drop_path"])
        model = LoRAT_DINOv2(vit, self.template_feat, self.search_feat)
        freeze_vit_backbone_(model)
        enable_lora_(model, _LORA_R, _LORA_ALPHA, 0.0, False)

        # Load the LoRA deltas + head weights from the safetensors checkpoint.
        sd = _safetensors_load(weights)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # The head and token_type_embed are trained task-specific params; they
        # MUST be in the checkpoint. pos_embed and norm.* are reused from the
        # DINOv2 backbone (pos_embed is re-interpolated in LoRAT.__init__) and
        # are expected to be absent from the LoRAT checkpoint.
        required_prefixes = ("head.", "token_type_embed")
        fatal = [k for k in missing if k.startswith(required_prefixes)]
        if fatal:
            raise RuntimeError(f"checkpoint missing required keys: {fatal[:5]}...")

        model.eval().to(self.device, dtype=self.dtype)
        self.model = model

        # Image normalization (imagenet mean/std) — LoRAT applies it inplace to /255 tensors.
        self._normalize_ = get_dataset_norm_stats_transform(_NORM, inplace=True)

        # Allocate pinned input buffers once.
        Wx, Hx = self.search_size
        Wt, Ht = self.template_size
        self._x_buf = torch.empty((max_batch, 3, Hx, Wx), dtype=torch.float32, device=self.device)
        self._z_buf = torch.empty((max_batch, 3, Ht, Wt), dtype=self.dtype, device=self.device)
        self._z_mask_buf = torch.empty((max_batch, self.template_feat[1], self.template_feat[0]),
                                       dtype=torch.long, device=self.device)

        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0

    # --- helpers -----------------------------------------------------------
    def _frame_to_tensor(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # BGR uint8 (H,W,3) -> RGB float32 (3,H,W) on CPU (cropping happens CPU-side)
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.float32)
        return t

    def _make_template(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray):
        img = self._frame_to_tensor(frame_bgr)
        image_mean = img.reshape(3, -1).mean(dim=1)          # (3,)
        out_image = torch.empty((3, self.template_size[1], self.template_size[0]), dtype=torch.float32)
        cropping_params = get_siamfc_cropping_params(
            bbox_xyxy.astype(float), _TEMPLATE_AREA_FACTOR, np.array(self.template_size))
        _, _, cropping_params = apply_siamfc_cropping(
            img, np.array(self.template_size), cropping_params,
            self.interp_mode, self.interp_align, image_mean,
            out_image=out_image)
        # Build foreground mask for template tokens.
        stride = (self.template_size[0] / self.template_feat[0],
                  self.template_size[1] / self.template_feat[1])
        from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.utils.bbox_mask_gen import (
            get_foreground_bounding_box,
        )
        fg = get_foreground_bounding_box(bbox_xyxy.astype(float),
                                         cropping_params, stride)
        bbox_clip_to_image_boundary_(fg, np.array(self.template_feat))
        z_mask_np = np.zeros((self.template_feat[1], self.template_feat[0]), dtype=np.int64)
        z_mask_np[int(fg[1]):int(fg[3]), int(fg[0]):int(fg[2])] = 1
        # Normalize template to the model's input format.
        z = out_image / 255.0
        self._normalize_(z)
        z = z.to(self.device, dtype=self.dtype)
        return z, image_mean.to(self.device), torch.from_numpy(z_mask_np).to(self.device)

    # --- public API --------------------------------------------------------
    def init(self, frame_bgr: np.ndarray, bbox_xyxy) -> int:
        bbox = np.asarray(bbox_xyxy, dtype=np.float64)
        if not _bbox_is_valid(bbox):
            raise ValueError(f"invalid bbox {bbox}")
        z, z_mean, z_mask = self._make_template(frame_bgr, bbox)
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = _Track(tid, z, z_mean, z_mask, bbox)
        return tid

    def kill(self, tid: int) -> None:
        self._tracks.pop(tid, None)

    def alive_ids(self) -> List[int]:
        return [t.id for t in self._tracks.values() if t.alive]

    @torch.inference_mode()
    def track(self, frame_bgr: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        ids = self.alive_ids()
        if not ids:
            return {}
        img = self._frame_to_tensor(frame_bgr)
        H, W = img.shape[-2:]
        image_size = np.array((W, H), dtype=np.int32)

        results: Dict[int, Tuple[np.ndarray, float]] = {}

        for start in range(0, len(ids), self.max_batch):
            chunk = ids[start:start + self.max_batch]
            n = len(chunk)

            cropping_params_batch = np.empty((n, 2, 2), dtype=np.float64)
            for i, tid in enumerate(chunk):
                tr = self._tracks[tid]
                cp = get_siamfc_cropping_params(
                    tr.last_box, self.search_area_factor, np.array(self.search_size))
                # Write directly into the float32 scratch buffer.
                _, _, cp_final = apply_siamfc_cropping(
                    img, np.array(self.search_size), cp,
                    self.interp_mode, self.interp_align, tr.z_mean.cpu(),
                    out_image=self._x_buf[i])
                cropping_params_batch[i] = cp_final
                self._z_buf[i].copy_(tr.z)
                self._z_mask_buf[i].copy_(tr.z_mask)

            x = self._x_buf[:n].clone() / 255.0
            self._normalize_(x)
            x = x.to(self.dtype)

            out = self.model(self._z_buf[:n], x, self._z_mask_buf[:n])
            boxes_sr, conf = _score_map_to_boxes(
                out["score_map"], out["boxes"], self.search_size)
            boxes_sr = boxes_sr.float().cpu().numpy().astype(np.float64)
            conf_np = conf.float().cpu().numpy()

            boxes_full = apply_siamfc_cropping_to_boxes(
                boxes_sr, reverse_siamfc_cropping_params(cropping_params_batch))
            for i, tid in enumerate(chunk):
                tr = self._tracks[tid]
                b = bbox_clip_to_image_boundary(boxes_full[i], image_size)
                s = float(conf_np[i])
                if _bbox_is_valid(b):
                    tr.last_box = b.astype(np.float64)
                results[tid] = (b.astype(np.float64), s)
        return results

    def n_alive(self) -> int:
        return len(self._tracks)
