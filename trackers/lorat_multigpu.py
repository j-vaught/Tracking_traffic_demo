"""
Thread-fan-out BatchedLoRAT across multiple GPUs.

Track state is sequential per object, so we can't split frames across GPUs.
We split *tracks*: each new init lands on the least-loaded GPU, and every
frame we fan-out to all GPUs concurrently via threading. CUDA kernel launches
and the ViT forward release the GIL, so Python threads give real parallelism.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch

from lorat_wrapper import BatchedLoRAT


class MultiGPULoRAT:
    def __init__(self, weights: str, variant: str, gpus: List[int],
                 dtype: torch.dtype = torch.float16, max_batch_per_gpu: int = 16):
        self.gpus = list(gpus)
        self.engines: List[BatchedLoRAT] = []
        for g in self.gpus:
            self.engines.append(BatchedLoRAT(
                weights=weights, variant=variant,
                device=f"cuda:{g}", dtype=dtype, max_batch=max_batch_per_gpu))
        # global track id -> (engine_idx, local_tid)
        self._routing: Dict[int, Tuple[int, int]] = {}
        self._next_tid = 0
        self._pool = ThreadPoolExecutor(max_workers=len(self.engines))

    def _counts(self) -> list[int]:
        c = [0] * len(self.engines)
        for g_idx, _ in self._routing.values():
            c[g_idx] += 1
        return c

    def init(self, frame_bgr: np.ndarray, bbox_xyxy) -> int:
        # Place new track on the least-loaded GPU.
        counts = self._counts()
        g_idx = counts.index(min(counts))
        local_tid = self.engines[g_idx].init(frame_bgr, bbox_xyxy)
        gid = self._next_tid
        self._next_tid += 1
        self._routing[gid] = (g_idx, local_tid)
        return gid

    def kill(self, gid: int) -> None:
        route = self._routing.pop(gid, None)
        if route is not None:
            g_idx, local_tid = route
            self.engines[g_idx].kill(local_tid)

    def alive_ids(self) -> List[int]:
        return list(self._routing.keys())

    def n_alive(self) -> int:
        return len(self._routing)

    def track(self, frame_bgr: np.ndarray) -> Dict[int, Tuple[np.ndarray, float]]:
        if not self._routing:
            return {}
        # Fan out: each engine advances its own tracks concurrently.
        futures = [self._pool.submit(e.track, frame_bgr) for e in self.engines]
        per_engine = [f.result() for f in futures]
        # Remap local tids -> global tids.
        out: Dict[int, Tuple[np.ndarray, float]] = {}
        for gid, (g_idx, local_tid) in self._routing.items():
            r = per_engine[g_idx].get(local_tid)
            if r is not None:
                out[gid] = r
        return out

    # Expose the per-engine `_tracks` so the orchestrator can re-anchor.
    def set_last_box(self, gid: int, box_xyxy: np.ndarray) -> None:
        g_idx, local_tid = self._routing[gid]
        self.engines[g_idx]._tracks[local_tid].last_box = box_xyxy.astype(np.float64)
