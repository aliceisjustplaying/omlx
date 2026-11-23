# Inkling 2 Mini Optimizations for Apple Silicon (M5 Pro)

This fork extends [`jundot/omlx`](https://github.com/jundot/omlx) and [`masterfasa/omlx`](https://github.com/masterfasa/omlx) (`feat/inkling2-basalt-combined`, commit `e1bffb9`) with five targeted Python and Metal kernel changes (`+2,057 / -47` lines) for **Inkling 2 Mini** (`oQ4e` mixed 4-bit NVFP4/Affine-4 experts with its bundled **DSpark** recurrent draft head) on Apple Silicon.

On an **Apple M5 Pro (48 GB unified memory)**, this branch reaches up to **1,376 tok/s** prefill at 27K context and **1,103 tok/s** at 54K (+62% faster long-context prefill vs before the M5 Metal 4 `matmul2d` kernels), **86–115 tok/s** on short coding/editing tasks and **46–51 tok/s** sampled decode from 6.7K to 54K context (**0.26–0.63s** warm TTFT on full-prefix retries).

> **Default tradeoff**: This branch defaults `OMLX_INKLING2_FP8_KV` to `"all"` (lossy unit-scale FP8 E4M3 KV cache across all attention layers) for long-context throughput and memory headroom. Set `OMLX_INKLING2_FP8_KV=0` for BF16 KV cache.

## What This Branch Changes

Every change sits in five files layered cleanly on top of `masterfasa/omlx@e1bffb9`. Below, `L` denotes the number of token rows evaluated in a single decode or speculative-verification forward pass (`1..16`), and `predraft` denotes overlapping the next speculative draft round with the current verification step.

| File | Delta | Summary |
| :--- | :---: | :--- |
| [`omlx/patches/inkling2/kernels.py`](omlx/patches/inkling2/kernels.py) | `+1,534 / -6` | Exact FP32/BF16 vectorized router (`fast_router_logits`), fused `static_conv4_fused` + residual add, packed NVFP4/Affine-4 MoE for `L=1..2`, bit-exact pre-planned grouped-16 NVFP4 MoE for `L=3..16`, M5 Metal 4 `matmul2d` FlashAttention (`flash_banded_attention_nax`) and batched MoE prefill (`moe_prefill_nax`) |
| [`omlx/patches/inkling2/dspark.py`](omlx/patches/inkling2/dspark.py) | `+383 / -8` | Fused in-kernel 2-bit screened top-64 sampled DSpark head (`_DSPARK_HEAD_CANDS_SOURCE`), emitting `SparseDraftQ` + calibrated confidence with adaptive step slicing (`4` prose, up to `7` code/JSON) |
| [`omlx/patches/inkling2/model.py`](omlx/patches/inkling2/model.py) | `+103 / -15` | Wires `fast_router_logits` (with bit-exact BF16 router cache), `moe_fused_packed`, `moe_prefill_nax` and fused conv+residual into `Inkling2Layer`/`Inkling2MoE`/`Inkling2Experts`, raises `_omlx_mtp_max_verify_rows = 16` and defaults `OMLX_INKLING2_FP8_KV` to `"all"` |
| [`omlx/patches/mlx_lm_mtp/batch_generator.py`](omlx/patches/mlx_lm_mtp/batch_generator.py) | `+33 / -14` | Sparse top-512 nucleus verify (`allow_nucleus=True`) alongside `predraft` with clean fallback, vectorized `SparseDraftQ` scatter, `OMLX_MTP_DRAFT_POLICY="1"` default and preventing `_park_mtp_to_standard` from parking `DraftPolicy` sessions |
| [`omlx/speculative/draft_policy.py`](omlx/speculative/draft_policy.py) | `+4 / -4` | Calibrates initial `_SHAPE` cost curve and `draft_ratio = 0.04` for M5 Pro fused MoE timings |

### 1. M5 Metal 4 `matmul2d` Prefill FlashAttention & Batched MoE (`kernels.py`, `model.py`)

- **Banded-sink FlashAttention (`flash_banded_attention_nax`)**: Uses Metal 4 `mpp::tensor_ops::matmul2d` (`AQ=32, AK=64, D=128`, 8 simdgroups, 12 KB threadgroup memory) streaming BF16 K/V directly from device memory (dequantizing FP8 KV once per layer instead of per query tile), adding relative bias and per-head sinks in-kernel with zero dense mask allocation. For long full-attention contexts (`>4K` keys), swaps the dispatch grid so Metal's Y-sweep keeps one KV head resident in M5 Pro's 24–32 MB SLC across all query tiles (`3.3x` faster attention at 46.7K keys; `mean_diff = 0.000001`).
- **Batched MoE prefill (`moe_prefill_nax`)**: Uses GPU counting-sort routing (`_NAX_MOE_ROUTE_SOURCE`) and double-buffered 64×32 BF16 weight tiles fed to per-simdgroup `matmul2d(16, 64, 32)` with fused SiTU register epilogues for both NVFP4 (`group_size=16`) and Affine-4 (`group_size=64`) experts (`L >= 64`). Both Metal 4 paths run a one-time accumulator-layout probe (`can_use_nax_attn()` / `can_use_nax_moe()`) and fall back automatically on M1–M4 or older macOS.

### 2. Exact BF16/FP32 Router, Fused Conv+Residual & Grouped-16 NVFP4 MoE (`kernels.py`, `model.py`)

- **Vectorized router (`fast_router_logits`)**: For `L <= 16` token rows, replaces the small-batch `x @ router_weight.T` matmul with a vectorized exact FP32-accumulating Metal dot product. Lazily caches `router_weight` in BF16 when bit-exact (verified `0.000000` max diff across all 24 layers of `oQ4e`), halving router memory traffic.
- **Fused conv+residual (`static_conv4_fused`)**: Fuses the `attn_conv` and `mlp_conv` residual additions directly into the width-4 causal conv kernel (`0.000000` max diff), eliminating ~90 elementwise kernels per decode/verify step.
- **Fused packed MoE (`moe_fused_packed`)**: Handles `L <= 16` verification batches for 4-bit `nvfp4` and `affine` experts (up to 8 active experts per token) in fused Metal kernels (gate/up projection, softcap, SiLU and weighted down projection):
  - **`L=1..2`**: Direct per-row packed kernels (`_UPS_NVFP4_SOURCE`, `_DOWNS_NVFP4_SOURCE`, `_UPS_AFF4_SOURCE`, `_DOWNS_AFF4_SOURCE`).
  - **`L=3..16` (NVFP4)**: A four-kernel pre-planned grouped pipeline (`_MOE_PLAN16_SOURCE`, `_UPS_NVFP4_GRP_SOURCE`, `_DOWNS_NVFP4_GRP_SOURCE`, `_MOE_COMB_SOURCE`) that groups rows routed to the same expert (up to 4 rows per work unit) so each shared expert weight is read from memory once instead of once per row. BF16 output matches single-row (`L=1`) decode to `0.000000` max difference across `L=3..16`.
- **Wider verify window**: Sets `_omlx_mtp_max_verify_rows = 16` so speculative verification can confirm up to 15 drafted tokens in one pass.

### 3. Fused 2-Bit Screened Sampled DSpark Head (`dspark.py`)

Inkling 2's DSpark module is a lightweight recurrent Markov draft head. Upstream `masterfasa/omlx` uses a 2-bit approximate screen + exact top-candidate evaluation for greedy decoding, while falling back to a full dense LM-head projection for sampled decoding (`temp > 0`).

This branch adds `_DSPARK_HEAD_CANDS_SOURCE` (`_dspark_screened_candidates` / `apply_step_candidates_with_conf`) to run the same 2-bit screen for sampled decoding:

- Evaluates the top 64 exact candidates per step and performs `temp` / `top_p` / `top_k` / `min_p` sampling inside a single Metal kernel launch per Markov step.
- Emits a compact `SparseDraftQ` (top-64 token IDs and log-probabilities) plus a calibrated per-step confidence (`0.5 + 0.5 * p_max`) for `DraftPolicy`.
- Slices `draft_hidden` before `prepare_block` and `screen_base` (`4` steps for prose, scaling up to `7` for high-acceptance code/JSON).
- Preserves the target model's exact sampling distribution through standard speculative rejection sampling.

### 4. Sparse Nucleus Verify, `predraft` & `DraftPolicy` Tuning (`batch_generator.py`, `draft_policy.py`)

- **Top-512 nucleus verify with `predraft`**: Enables fast top-512 sparse nucleus verification (`allow_nucleus=True`) while `predraft` is active. If any verified token's target nucleus exceeds 512 tokens, the queued predraft is discarded cleanly via `drafter.discard_predraft()` and verification falls back to exact dense rejection sampling.
- **Vectorized `SparseDraftQ` scatter**: Uses `mx.put_along_axis` instead of per-row Python stacking when dense fallback runs.
- **`DraftPolicy` defaults & M5 Pro curve**: Defaults `OMLX_MTP_DRAFT_POLICY` to `"1"`, prevents `_park_mtp_to_standard` from disabling speculative decoding while `DraftPolicy` is active and updates `_SHAPE` and `draft_ratio = 0.04` to reflect the lower draft and multi-row MoE cost.

### 5. Base Optimizations Inherited from `masterfasa/omlx` (`e1bffb9`)

This branch builds directly on `masterfasa/omlx`'s `feat/inkling2-basalt-combined` (PRs #5505–#5511), which already provides:

- Unit-scale FP8 E4M3 KV cache (this branch flips `OMLX_INKLING2_FP8_KV` default to `"all"`).
- GQA-packed Split-K FlashDecoding.
- Fused Metal BF16 loop controller.
- 2-bit screened exact greedy DSpark head and global layer-1 draft KV cache.
- Adaptive `DraftPolicy` and `SuffixDrafter`.

## Measured Performance (Apple M5 Pro, 48 GB)

**Test environment**: Apple M5 Pro, 48 GB unified memory, macOS 27.3, `mlx==0.32.4`, single-request decode.
This branch ran [`mlx-community/Inkling-2-Mini-oQ4e-mtp`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp) (4-bit NVFP4/Affine-4 routed experts, 8-bit attention/embeddings, BF16 DSpark head, FP8 E4M3 KV cache, prefix cache enabled).

> **Baseline isolation note**: Unmodified `masterfasa/omlx@e1bffb9` was not separately benchmarked on this exact prompt suite. The numbers below measure the combined stack (`masterfasa/omlx@e1bffb9` + this branch's 5-file patch) end-to-end rather than isolating the 5-file delta alone.

### Short & Medium Tasks (Warm Prefix Cache)

Sampled runs use `temp=1.0, top_p=0.95`; greedy runs use `temp=0`. Each prompt ran with its prefix already warm in oMLX's KV cache (`prompt_tokens - 1` cached tokens), isolating steady-state decode throughput. Workloads: `code` (JavaScript function generation), `json` (structured record extraction), `real` (targeted code edit) and `edit` (multi-line file rewrite).

| Workload | Prompt Tokens | Output Tokens (Sampled / Greedy) | This Fork (Sampled) | This Fork (Greedy) |
| :--- | :---: | :---: | :---: | :---: |
| `code` | 83 | 956 / 1,039 | **86.0 tok/s** (`0.18s` TTFT) | **79.1 tok/s** (`0.18s` TTFT) |
| `json` | 166 | 633 / 599 | **88.6 tok/s** (`0.20s` TTFT) | **97.8 tok/s** (`0.19s` TTFT) |
| `real` | 516 | 687 / 611 | **94.2 tok/s** (`0.27s` TTFT) | **108.6 tok/s** (`0.18s` TTFT) |
| `edit` | 818 | 905 / 884 | **102.6 tok/s** (`0.22s` TTFT) | **128.9 tok/s** (`0.18s` TTFT) |

- On `code`, sampled decode (`86.0 tok/s`) finished faster than greedy (`79.1 tok/s`) because the sampled trajectory produced slightly shorter reasoning and higher average speculative acceptance on that prompt.
- A separate OpenAI tool-call check (`read` `/etc/hosts`) completed in **1.23s** total with valid `finish_reason="tool_calls"`.

### Long-Context Prefill & Sampled Decode (`temp=1.0, top_p=0.95`, 512 Output Tokens)

Measured head-to-head on identical 6.7K → 27.2K → 53.9K multi-turn prompts before and after adding the Metal 4 `matmul2d` FlashAttention (`flash_banded_attention_nax`), `matmul2d` MoE prefill (`moe_prefill_nax`), BF16 router and fused conv+residual kernels:

| Turn / Context Length | Cached Prefix | Before (`80b1b4d`) Prefill / TTFT | After (`matmul2d` NAX) Prefill / TTFT | Prefill Gain | Decode Speed (After) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turn 1 — 6,709 tokens (cold) | 0 | 953 tok/s (`7.23s`) | **1,206 tok/s (`5.71s`)** | **+26.6%** | **51.4 tok/s** |
| Turn 2 — 27,179 tokens | 6,144 | 981 tok/s (`21.62s`) | **1,376 tok/s (`15.48s`)** | **+40.2%** | **47.9 tok/s** |
| Turn 3 — 53,859 tokens | 26,624 | 681 tok/s (`40.19s`) | **1,103 tok/s (`24.91s`)** | **+62.0%** | **43.3 tok/s** |
| Turn 3 warm retry — 53,859 tokens (2 seeds) | 53,858 | `0.25–0.71s` (`42.6–43.5 tok/s`) | **`0.26–0.63s`** | — | **46.1–46.8 tok/s** |

*(Earlier `80b1b4d` multi-turn measurements on a separate 5.8K → 23.5K → 46.7K prompt suite for the `mlx-serve` comparison below: 977 / 977 / 753 tok/s prefill and 55.4 / 51.4 / 47.7 tok/s decode, with 43.8–45.3 tok/s and 0.32–0.65s TTFT on warm 46.7K retries.)*

### Reference Comparison on the Same M5 Pro

For context, here are measurements taken on the same M5 Pro against `ddalcu/mlx-serve v26.11.4-pre-release.1` (`--os-reserve-gib 12 --ctx-size 262144 --kv-quant 8`, running [`ddalcu/Inkling-2-Mini-MLX-8bit`](https://huggingface.co/ddalcu/Inkling-2-Mini-MLX-8bit) where the hot prompt cache clamped to `0 MB` under the 12 GiB reserve):

| Scenario | This Fork (`oQ4e` + FP8 KV) | `mlx-serve` (`MLX-8bit` + 8-bit KV) |
| :--- | :---: | :---: |
| Short sampled (`code / json / real / edit`) | **86.0 / 88.6 / 94.2 / 102.6 tok/s** | 66.8 / 88.8 / 78.6 / 89.0 tok/s |
| Short greedy (`code / json / real / edit`) | **79.1 / 97.8 / 108.6 / 128.9 tok/s** | 63.0 / 89.9 / 90.4 / 94.1 tok/s |
| 5.8K sampled decode (cold) | **55.4 tok/s** (`5.90s` TTFT) | 41.3 tok/s (`6.18s` TTFT) |
| 23.5K sampled decode (turn 2) | **51.4 tok/s** (`18.21s` TTFT) | 31.5 tok/s (`26.09s` TTFT) |
| 46.7K sampled decode (turn 3, 23.5K cached on oMLX) | **47.7 tok/s** (`30.75s` TTFT) | 22.2 tok/s (`61.82s` TTFT) |
| 46.7K sampled decode (full-prefix warm retry, 2 seeds) | **43.8–45.3 tok/s** (`0.32–0.65s` TTFT) | 22.2–22.3 tok/s (`61.78–62.27s` TTFT) |

> **Comparison note**: This is not a weight-for-weight apples-to-apples benchmark. This fork ran a 4-bit mixed NVFP4/Affine-4 checkpoint (`oQ4e`) with FP8 KV and prefix caching enabled, while `mlx-serve` ran an 8-bit MLX checkpoint with 8-bit KV and no prefix cache reuse across turns. Both rows reflect the practical single-user setup tested for each server on a 48 GB M5 Pro.

## Precision & Losslessness

- **MoE & router kernels**: Exact relative to the loaded `oQ4e` checkpoint (`0.000000` max difference vs single-row decode for grouped NVFP4 MoE).
- **Sampled DSpark speculation**: Exact relative to the loaded checkpoint's target distribution via speculative rejection sampling.
- **KV cache (`OMLX_INKLING2_FP8_KV`)**: Defaults to `"all"` (unit-scale FP8 E4M3 across all attention layers) for long-context speed and memory savings. This is lossy KV quantization; set `OMLX_INKLING2_FP8_KV=full` to apply FP8 KV only to full-attention layers or `OMLX_INKLING2_FP8_KV=0` for BF16 KV.

## Install & Run

Requires macOS 15.0+, Python 3.11–3.13, Apple Silicon and a 4-bit Inkling 2 Mini MTP checkpoint such as [`mlx-community/Inkling-2-Mini-oQ4e-mtp`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp) (~18 GB loaded in memory; quantized from [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini), so 32 GB+ unified memory is recommended for comfortable headroom with long context).

```bash
git clone -b inkling2-m5-pro https://github.com/aliceisjustplaying/omlx.git
cd omlx

# Activate your Python 3.11–3.13 virtual environment first (make runs `python3 -m pip install -e .`)
python3 -m venv .venv && source .venv/bin/activate

# Full editable install (needs full Xcode for optional C++/Metal kernels used by other model families)
make install

# Or without full Xcode — Inkling 2's Metal kernels compile at runtime via mx.fast.metal_kernel
make install-no-kernels
```

Download the checkpoint into `~/models` (`hf` is installed with `huggingface_hub` inside `.venv`) and start the server (default endpoint `http://127.0.0.1:8000/v1`; DSpark MTP speculation engages automatically for `-mtp` checkpoints without extra server flags):

```bash
hf download mlx-community/Inkling-2-Mini-oQ4e-mtp --local-dir ~/models/Inkling-2-Mini-oQ4e-mtp
omlx serve --model-dir ~/models
```

Verify the server and check server logs for `MTP[...]` lines after your first request to confirm speculative decoding is engaged:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

### Configuration Environment Variables

| Variable | Default in This Branch | Options / Effect |
| :--- | :---: | :--- |
| `OMLX_INKLING2_FP8_KV` | `"all"` | `"all"` (FP8 E4M3 KV on all layers), `"full"` (full-attention layers only) or `"0"` (BF16 KV) |
| `OMLX_MTP_DRAFT_POLICY` | `"1"` | `"1"` (adaptive `DraftPolicy` enabled) or `"0"` (disable `DraftPolicy`) |

## Known Limits

- **Hardware coverage**: The decode Metal kernels use standard Metal SIMD operations and are expected to compile on M1–M4 Macs; the Metal 4 `matmul2d` prefill kernels (`flash_banded_attention_nax` and `moe_prefill_nax`) require Apple Silicon with Metal 4 Neural Accelerators (M5+) and automatically probe layout support (`can_use_nax_attn()` / `can_use_nax_moe()`), falling back cleanly on M1–M4 or older macOS. All kernels were developed and tested on an Apple M5 Pro (48 GB, macOS 27.3, `mlx==0.32.4`). `DraftPolicy` adapts online to measured round timings, while its initial `_SHAPE` prior is tuned for M5 Pro.
- **FP8 KV quality**: `OMLX_INKLING2_FP8_KV="all"` was smoke-tested on coding, JSON, tool-calling and long-context prompts rather than evaluated on a formal perplexity benchmark. Switch to `OMLX_INKLING2_FP8_KV=full` or `0` if you need exact BF16 KV.
- **Checkpoint scope**: `moe_fused_packed` targets 4-bit `nvfp4` and `affine` MoE weights (`L <= 16`, up to 8 experts per token). Other bit widths automatically fall back to MLX's standard path.
- **Concurrency**: Benchmarks cover single-request decode. High-concurrency multi-user batching was not benchmarked.
- **Upstream sync**: `jundot/omlx` and `masterfasa/omlx` evolve rapidly; this branch is a snapshot based on `masterfasa/omlx@e1bffb9`.

## Credits & Attribution

- **[`jundot/omlx`](https://github.com/jundot/omlx)** — upstream oMLX server, tiered KV caching and Inkling 2 integration.
- **[`masterfasa/omlx`](https://github.com/masterfasa/omlx)** (`feat/inkling2-basalt-combined`, PRs #5505–#5511) — unit-scale FP8 E4M3 KV cache, GQA Split-K FlashDecoding, fused Metal BF16 loop controller, 2-bit screened greedy DSpark head, `DraftPolicy` and `SuffixDrafter`.
- **[`ddalcu/mlx-serve`](https://github.com/ddalcu/mlx-serve)** — Metal kernel design inspiration for `fast_router_logits` and `L=1..2` packed NVFP4/Affine-4 MoE.
- **[`ShinMegamiBoson/lookatmyquant`](https://github.com/ShinMegamiBoson/lookatmyquant)** — pre-planned expert-grouping design inspiration for shared MoE experts across `L=3..16`.
- **[`Eriskii/erisinky`](https://github.com/Eriskii/erisinky)** — Metal 4 `mpp::tensor_ops::matmul2d` FlashAttention (`k_attn_h`), per-simdgroup `matmul2d` MoE prefill (`k_moe_sg`), Y-axis cache-locality dispatch order, bit-exact BF16 router weights and fused post-conv+residual epilogues.
