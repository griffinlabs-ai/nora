# Training Pipeline Bottlenecks

## Observed symptoms

| Config | GPU util | Observed bottleneck |
|---|---|---|
| `per_device_batch_size=1` (original) | GPU ~60% | Main thread at ~100% CPU |
| `num_frames=3`, `per_device_batch_size=2` | GPU ~60% | Worker threads ~80–90% CPU |
| Two-node setup | GPU ~60% | Possibly additional node sync overhead |

GPU sitting at ~60% means it is idle roughly a third of the time — waiting on CPU to prepare batches or on cross-rank synchronization after backward passes.

The bottlenecks fall into three categories: CPU compute (main thread and workers), GPU VRAM, and GPU utilization (time the GPU actually spends on forward/backward vs. idle).

---

## CPU compute bottlenecks

These are what push CPU threads toward 100% and starve the GPU of work.

### 1 — Per-step grad-norm: thousands of GPU→CPU syncs on the main thread

**Location:** [lerobot_training/lerobot_training.py:424-435](../lerobot_training/lerobot_training.py#L424-L435)

```python
if completed_steps % config.logging_frequency == 0:   # logging_frequency defaults to 1
    if accelerator.is_main_process:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2   # GPU→CPU sync per tensor
```

`logging_frequency=1` means this runs every optimizer step. Each `.item()` call forces a GPU→CPU copy and sync. For Gemma-4-E4B this is thousands of syncs per step — the main thread is doing almost nothing else while the GPU idles.

**Fix:** One fused norm + one `.item()`, and log less often:
```python
grads = [p.grad for p in model.parameters() if p.grad is not None]
total_norm = torch.stack([g.detach().norm(2) for g in grads]).norm(2).item()
```
Set `logging_frequency` to 10–50.

### 2 — Video frame decoding scales with `num_frames × num_cameras` (workers)

**Location:** [utils/data_loading.py:233-248](../utils/data_loading.py#L233-L248)

With 3 cameras and `delta_timestamps` set to `num_frames` timestamps, every `__getitem__` call triggers:

| `num_frames` | Seeks/decodes per sample |
|---|---|
| 6 | 18 |
| 3 | 9 |

lerobot decodes via PyAV/FFmpeg. With `shuffle=True` + `ConcatDataset`, each fetch is a random seek into an arbitrary video file — no read-ahead possible. This is the dominant per-sample CPU cost in workers, and explains why reducing `num_frames` from 6 to 3 visibly lowered worker load.

**Tactical fix:** Increase `num_workers` beyond 4 (workers aren't hard-pegged so there is headroom), add `prefetch_factor=4`, `persistent_workers=True`:
```python
DataLoader(..., num_workers=8, prefetch_factor=4, persistent_workers=True)
```

**Structural fix:** Pre-process datasets into stacked per-sample frame tensors stored as `.pt` files. Workers do one file read instead of 18 video seeks. Biggest win but requires a one-off preprocessing script.

### 3 — Image transforms applied per-frame, per-camera, in a Python loop (workers)

**Location:** [lerobot_training/lerobot_training.py:152-158](../lerobot_training/lerobot_training.py#L152-L158)

```python
for t in range(T):
    for k in self.IMAGE_KEYS:
        frame = img_tensor[t] if img_tensor.dim() == 4 else img_tensor
        transformed_frame = self.nora_image_transform(frame)   # one call per (t, camera)
```

With `num_frames=3` × 3 cameras × `batch_size=2` = 18 individual torchvision calls per batch. Each call has Python overhead.

**Fix:** Stack all frames into `(N, C, H, W)` and call the transform once. torchvision v2 transforms natively support batched tensors.

### 4 — `pin_memory=True` with image lists (main process pin-memory thread)

**Location:** [lerobot_training/lerobot_training.py:363](../lerobot_training/lerobot_training.py#L363), [utils/data_loading.py:313-336](../utils/data_loading.py#L313-L336)

`collate_with_observation_image_lists` keeps `observation.images.*` as Python lists of tensors. The DataLoader's pin-memory thread (main process) has to walk each list and pin tensors one by one, which is much slower than pinning a single stacked tensor.

**Fix:** Within a batch, images for a given embodiment share the same shape — stack them per camera key in `collate_with_observation_image_lists` and only fall back to a list when shapes genuinely differ. Or try `pin_memory=False` and measure — the overhead may outweigh the gain.

### 5 — `SkipEpisodesLeRobotDataset.__getitem__` linear scan (workers)

**Location:** [utils/data_loading.py:75-80](../utils/data_loading.py#L75-L80)

```python
for new_range, old_offset in self.new_ranges_to_old_offsets.items():
    if idx in new_range:
        return self.lerobot_ds[idx + old_offset]
```

O(n) scan per sample fetch, where n is the number of contiguous non-dirty episode runs. Not the primary bottleneck, but adds up at scale.

**Fix:** Replace with `bisect` on a sorted list of `(range_start, old_offset)` pairs — O(log n).

### 6 — CubicSpline constructed per sample (workers, conditional)

**Location:** [utils/data_loading.py:120-131](../utils/data_loading.py#L120-L131)

When `load_action_chunk_size` is not a clean multiple of `canonical_action_chunk_size`, a new `scipy.CubicSpline` is fit and evaluated on every sample.

**Fix:** Verify the divisible branch (`action[step_size-1::step_size]`) is always being taken. If not, precompute the interpolation matrix once — the old/new time grids are fixed per dataset — and apply it as a matrix multiply per sample instead of re-fitting a spline.

### 7 — wandb logging on every step (main process)

**Location:** [lerobot_training/lerobot_training.py:435](../lerobot_training/lerobot_training.py#L435)

`accelerator.log()` every step adds Python dict creation, W&B client overhead, and occasional network activity on the main thread. Minor in isolation but compounds with bottleneck #1.

**Fix:** Covered by the `logging_frequency` fix in #1.

---

## GPU VRAM bottlenecks

These limit `per_device_batch_size` and `num_frames`, which in turn force more data-loading calls per GPU-hour.

### 8 — Image history multiplies activation memory

Each additional frame adds one full forward pass of Gemma-4's vision encoder to the activation graph. At `num_frames=6`, you're holding 6× the vision activations in VRAM during the backward pass. Going from 6 → 3 frames is what allows `per_device_batch_size` to go from 1 to 2.

**Mitigations:**
- **Gradient checkpointing** on the vision encoder: `model.gradient_checkpointing_enable()` recomputes activations during backward instead of storing them. Trades ~30% extra compute for large VRAM savings — often worth it when batch size is VRAM-constrained.
- **Reduce `max_tokens_per_image`** (currently 70): fewer soft tokens per image = smaller attention context = less KV cache in VRAM.
- **Freeze the vision encoder** during early training: if only the LM and action token embeddings need updating, the vision encoder gradients (and therefore activations) don't need to be kept alive through backward.

### 9 — Action chunk size drives sequence length

`action_chunk_size=50` produces a long assistant turn in the chat template. Longer sequences = larger attention matrices = more VRAM.

**Mitigation:** If 50 is not empirically necessary, try 25 or 10 first. The attention memory cost scales quadratically with sequence length for standard attention, or linearly with flash-attention.

### 10 — Flash attention is not enabled

**Location:** [lerobot_training/lerobot_training.py:262-266](../lerobot_training/lerobot_training.py#L262-L266)

```python
model = AutoModelClass.from_pretrained(
    config.model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
    # attn_implementation="flash_attention_2" not set
)
```

Flash Attention 2 (or SDPA) can cut attention VRAM by up to 10× for long sequences compared to eager attention, and usually speeds up the attention kernel. If `flash-attn` is installed in the environment, add `attn_implementation="flash_attention_2"`.

---

## GPU utilization bottlenecks

These are things that reduce the fraction of time the GPU spends on actual compute (forward + backward) vs. idle / synchronizing.

### 11 — `find_unused_parameters=True` adds a graph traversal per backward

**Location:** [lerobot_training/lerobot_training.py:326](../lerobot_training/lerobot_training.py#L326)

```python
DistributedDataParallelKwargs(find_unused_parameters=True)
```

DDP walks the entire autograd graph at the end of every backward to find parameters that received no gradient. This adds latency before the all-reduce can start, reducing the overlap between compute and communication.

**Fix:** After stabilizing training, profile which params have `grad=None` after a backward pass. Either freeze them (`requires_grad=False`) or verify they are in fact all used and flip to `find_unused_parameters=False`.

### 12 — Straggler waits at all-reduce from uneven data loading

**Location:** [lerobot_training/lerobot_training.py:329](../lerobot_training/lerobot_training.py#L329) (`dispatch_batches=False`)

Each rank loads its own batches independently. Because `shuffle=True` + `ConcatDataset` causes random video seeks, per-batch latency varies significantly. The faster rank finishes its backward and waits at the NCCL all-reduce for the slower rank. This appears as periodic GPU idle spikes, especially visible in two-node setups where network latency adds to the wait.

**Mitigation:** `prefetch_factor` (see #2) helps smooth out latency spikes. The structural pre-decode fix (#2) removes variance at the source. You can also set a sampler that assigns a fixed embodiment to each rank, so each rank's video reads are confined to one dataset and get better OS-level caching.

### 13 — Large gradient accumulation = infrequent all-reduces but long idle stretches

`gradient_accumulation_steps=128` means the GPU runs 128 micro-steps before any optimizer/all-reduce work. This is good for throughput (fewer all-reduces) but if any micro-step stalls (e.g. waiting on a slow batch), the whole accumulation window is delayed.

This is mostly fine but interacts badly with the data-loading variance in #12.

---

## Priority order (by expected impact, not effort)

| Rank | Fix | Category | Why it's high impact |
|---|---|---|---|
| 1 | Pre-decode video clips to `.pt` frame stacks | CPU (workers) | Eliminates the single largest per-sample cost: 18 random video seeks → 1 file read. Directly addresses the worker bottleneck that limits batch size and throughput. |
| 2 | Fused grad-norm + `logging_frequency=25` | CPU (main) | Removes thousands of GPU→CPU syncs per step from the main thread. With batch=1 this is what drives main thread to 100% and starves the GPU. |
| 3 | Gradient checkpointing on vision encoder | GPU VRAM | Allows `per_device_batch_size` to grow (currently VRAM-limited). Larger batches = fewer DataLoader calls per GPU-hour = less relative data-loading overhead. |
| 4 | `attn_implementation="flash_attention_2"` | GPU VRAM + GPU util | Reduces attention VRAM (enabling larger batches) and speeds up attention kernels. Particularly valuable with long sequences from large action chunks and multi-image inputs. |
| 5 | Batched image transforms (stack frames before transform) | CPU (workers) | Cuts 18 per-frame Python transform calls to 1 batched call per sample. Reduces per-sample worker CPU time, helping keep workers from gating GPU. |
| 6 | `find_unused_parameters=False` | GPU util | Removes per-backward autograd graph traversal and unblocks all-reduce earlier. More impactful with two nodes where all-reduce latency is already high. |
| 7 | `num_workers=8`, `prefetch_factor=4`, `persistent_workers=True` | CPU (workers) | Workers aren't hard-pegged (80–90%), so more workers directly increase batch prefetch depth and absorb latency spikes from random video seeks until fix #1 is done. |
| 8 | Stack images in collate to fix `pin_memory` overhead | CPU (main) | Removes per-tensor pinning loop on main process, but lower impact than the grad-norm sync which is proportionally larger. |
| 9 | `bisect`-based index in `SkipEpisodesLeRobotDataset` | CPU (workers) | Correctness/scalability fix more than a bottleneck at current dataset sizes. |
