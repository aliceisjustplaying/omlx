# SPDX-License-Identifier: Apache-2.0
"""Attention and attention-residual primitives for Inkling 2."""

from __future__ import annotations

import mlx.core as mx

from ...custom_kernels.nax import is_nax_available

# Upper bound on one additive mask block (elements) in the general attention
# path; queries are split into sub-blocks so the mask stays below it.
_MASK_BUDGET = 1 << 29


_ATTN_RES_TG = 256
_attn_res_kernels: dict = {}
_attn_res_fused_kernels: dict = {}


def _row0_view(a: mx.array) -> mx.array:
    """Return ``a[:1]`` when it provably aliases ``a``'s buffer at offset 0.

    MLX counts each bound input array's full byte size against its 50 MB
    per-command-buffer cap, so binding stacked expert weights or lm_head
    fragments every forward into ~100+ command buffers. Custom kernels that
    index from buffer offset 0 can safely bind this 1-row alias while the
    owning module keeps the full parameter alive.
    """
    if a is None or a.ndim < 2 or a.shape[0] <= 1:
        return a
    import numpy as np

    view = a[:1]
    mx.eval(view)
    u8_whole = a.view(mx.uint8)
    u8_first = view.view(mx.uint8)
    mx.eval(u8_whole, u8_first)
    mv_w, mv_f = memoryview(u8_whole), memoryview(u8_first)
    addr_w = np.frombuffer(mv_w, dtype=np.uint8).ctypes.data
    addr_f = np.frombuffer(mv_f, dtype=np.uint8).ctypes.data
    if mv_w.c_contiguous and mv_f.c_contiguous and addr_f == addr_w:
        return view
    return a


def prepare_moe_packed_views(mod):
    """Prepare cached 1-expert offset-0 views outside ``mx.compile``."""
    w = mod.weight
    cached = getattr(mod, "_omlx_row0_views", None)
    if cached is not None and cached[0] == id(w):
        return cached[1], cached[2], cached[3]
    wv = _row0_view(w)
    sv = _row0_view(mod.scales)
    b = getattr(mod, "biases", None)
    bv = _row0_view(b) if b is not None else None
    mod._omlx_row0_views = (id(w), wv, sv, bv)
    return wv, sv, bv


def _moe_packed_views(mod):
    """Read cached 1-expert views (compile-safe; no eval/memoryview)."""
    w = mod.weight
    cached = getattr(mod, "_omlx_row0_views", None)
    if cached is not None and cached[0] == id(w):
        return cached[1], cached[2], cached[3]
    return w, mod.scales, getattr(mod, "biases", None)


def _attn_res_fused_kernel(
    blocks: int, queries: int, with_norm: bool, emit_pn: bool
):
    """Fused unit-RMS(partial) + depth-residual softmax mix + optional RMSNorm."""
    key = (blocks, queries, with_norm, emit_pn)
    kernel = _attn_res_fused_kernels.get(key)
    if kernel is not None:
        return kernel
    srcs = [f"b{i}" for i in range(blocks)]
    b_loads = "\n".join(
        f"        float s{i} = b{i}[base + h];" for i in range(blocks)
    )
    b_dots = "\n".join(
        f"            acc[{i} * Q + qi] += s{i} * qv;" for i in range(blocks)
    )
    mix_terms = [f"w[{i} * Q + qi] * s{i}" for i in range(blocks)] + [
        f"w[{blocks} * Q + qi] * sp"
    ]
    mix_expr = " + ".join(mix_terms)
    pn_store = "        pn[base + h] = sp;" if emit_pn else ""

    if with_norm:
        nw_names = [f"nw{qi}" for qi in range(queries)]
        out_names = [f"out{qi}" for qi in range(queries)]
        v_sq_acc = "\n".join(
            f"        {{\n"
            f"            uint qi = {qi}u;\n"
            f"            float mv = {mix_expr};\n"
            f"            vreg[{qi}][it] = mv;\n"
            f"            v_sq[{qi}] += mv * mv;\n"
            f"        }}"
            for qi in range(queries)
        )
        out_stores = "\n".join(
            f"        out{qi}[base + h] = OT(vreg[{qi}][it] * inv_v[{qi}] * float(nw{qi}[h]));"
            for qi in range(queries)
        )
        pass2 = f"""
    float vreg[Q][MAX_IT];
    float v_sq[Q];
    for (uint qi = 0; qi < Q; ++qi) v_sq[qi] = 0.0f;
    uint it = 0u;
    for (uint h = tid; h < H; h += TG, ++it) {{
{b_loads}
        float sp = float(p[base + h]) * inv_p;
{pn_store}
{v_sq_acc}
    }}
    threadgroup float vred[Q][TG / 32];
    threadgroup float vfold[Q];
    for (uint qi = 0; qi < Q; ++qi) {{
        float sv = simd_sum(v_sq[qi]);
        if (lane == 0) vred[qi][sg] = sv;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float eps_n = scale[1];
    if (tid < Q) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += vred[tid][j];
        vfold[tid] = metal::rsqrt(t / float(H) + eps_n);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_v[Q];
    for (uint qi = 0; qi < Q; ++qi) inv_v[qi] = vfold[qi];
    it = 0u;
    for (uint h = tid; h < H; h += TG, ++it) {{
{out_stores}
    }}
"""
    else:
        nw_names = []
        out_names = ["out"]
        pass2 = f"""
    for (uint h = tid; h < H; h += TG) {{
{b_loads}
        float sp = float(p[base + h]) * inv_p;
{pn_store}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix_expr};
        }}
    }}
"""
    if emit_pn:
        out_names = [*out_names, "pn"]

    source = f"""
    constexpr uint NS = {blocks + 1};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    float p_sq = 0.0f;
    for (uint h = tid; h < H; h += TG) {{
{b_loads}
        float pv = float(p[base + h]);
        p_sq += pv * pv;
        for (uint qi = 0; qi < Q; ++qi) {{
            float qv = query[qi * H + h];
{b_dots}
            acc[{blocks} * Q + qi] += pv * qv;
        }}
    }}
    threadgroup float red[NS * Q + 1][TG / 32];
    threadgroup float fold[NS * Q + 1];
    threadgroup float sw[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) red[i][sg] = v;
    }}
    {{
        float ps = simd_sum(p_sq);
        if (lane == 0) red[NS * Q][sg] = ps;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i <= NS * Q; i += TG) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];
        fold[i] = t;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_p = metal::rsqrt(fold[NS * Q] / float(H) + 1e-6f);
    float sc = scale[0];
    if (tid < Q) {{
        float logits[NS];
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) {{
            float t = fold[n * Q + tid];
            if (n == {blocks}u) t *= inv_p;
            logits[n] = t * sc;
            m = max(m, logits[n]);
        }}
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            logits[n] = metal::exp(logits[n] - m);
            z += logits[n];
        }}
        for (uint n = 0; n < NS; ++n) sw[n * Q + tid] = logits[n] / z;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) w[i] = sw[i];
{pass2}
"""
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_f_{blocks}_{queries}_{int(with_norm)}_{int(emit_pn)}",
        input_names=[*srcs, "p", "query", "scale", *nw_names],
        output_names=out_names,
        source=source,
    )
    _attn_res_fused_kernels[key] = kernel
    return kernel


def attn_res_mix_raw(
    blocks: list,
    partial: mx.array,
    query: mx.array,
    scale_eps: mx.array,
    *,
    norm_weights: tuple[mx.array, ...] | None = None,
    out_dtype=None,
    emit_partial_n: bool = False,
):
    """Fused unit-RMS(partial) + attn_res_mix (+ optional RMSNorm + cast).

    When ``norm_weights`` is provided, returns ``(out0, ...)`` (or ``out0`` when
    Q=1) in ``out_dtype`` (plus ``partial_n`` if ``emit_partial_n``). Otherwise
    returns FP32 ``[B, L, Q, H]`` (plus ``partial_n`` if ``emit_partial_n``).
    """
    b, length, hidden = partial.shape
    nq = query.shape[0]
    rows = b * length
    with_norm = norm_weights is not None
    max_it = (hidden + _ATTN_RES_TG - 1) // _ATTN_RES_TG
    if (
        mx.default_device() != mx.gpu
        or (with_norm and max_it > 16)
    ):
        pf = partial.astype(mx.float32)
        pn = pf * mx.rsqrt(mx.mean(pf * pf, axis=-1, keepdims=True) + 1e-6)
        mixed = attn_res_mix(blocks, pn, query, float(scale_eps[0].item()))
        if with_norm:
            eps_n = float(scale_eps[1].item())
            dt = partial.dtype if out_dtype is None else out_dtype
            outs = tuple(
                mx.fast.rms_norm(mixed[:, :, qi], norm_weights[qi], eps_n).astype(dt)
                for qi in range(nq)
            )
            res = outs[0] if nq == 1 else outs
        else:
            res = mixed
        return (res, pn) if emit_partial_n else res

    kernel = _attn_res_fused_kernel(len(blocks), nq, with_norm, emit_partial_n)
    inputs = [*blocks, partial, query, scale_eps]
    template = [("H", hidden), ("Q", nq), ("TG", _ATTN_RES_TG)]
    if with_norm:
        dt = partial.dtype if out_dtype is None else out_dtype
        inputs.extend(norm_weights)
        template.extend([("MAX_IT", max_it), ("OT", dt)])
        out_shapes = [(b, length, hidden)] * nq
        out_dtypes = [dt] * nq
    else:
        out_shapes = [(b, length, nq, hidden)]
        out_dtypes = [mx.float32]
    if emit_partial_n:
        out_shapes.append((b, length, hidden))
        out_dtypes.append(mx.float32)
    outs = kernel(
        inputs=inputs,
        template=template,
        grid=(rows * _ATTN_RES_TG, 1, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=out_shapes,
        output_dtypes=out_dtypes,
    )
    if emit_partial_n:
        pn = outs[-1]
        main = outs[0] if (not with_norm or nq == 1) else tuple(outs[:-1])
        return main, pn
    return outs[0] if (not with_norm or nq == 1) else tuple(outs)


def _attn_res_kernel(blocks: int, queries: int):
    """Fused depth mixing for ``blocks`` completed blocks plus the partial sum.

    One threadgroup per token row: dot products of every source with every
    query, softmax over sources, then the weighted sum, without stacking the
    sources.
    """
    key = (blocks, queries)
    kernel = _attn_res_kernels.get(key)
    if kernel is not None:
        return kernel
    srcs = [f"b{i}" for i in range(blocks)] + ["p"]
    loads = "\n".join(
        f"        float s{i} = {name}[base + h];" for i, name in enumerate(srcs)
    )
    dots = "\n".join(
        f"        acc[{i} * Q + qi] += s{i} * qv;" for i in range(len(srcs))
    )
    mix = " + ".join(f"w[{i} * Q + qi] * s{i}" for i in range(len(srcs)))
    source = f"""
    constexpr uint NS = {len(srcs)};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            float qv = query[qi * H + h];
{dots}
        }}
    }}
    threadgroup float red[NS * Q][TG / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix};
        }}
    }}
"""
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_{blocks}_{queries}",
        input_names=[*srcs, "query", "scale"],
        output_names=["out"],
        source=source,
    )
    _attn_res_kernels[key] = kernel
    return kernel


def attn_res_mix(blocks: list, partial_n: mx.array, query: mx.array, scale: float):
    """Mix unit-RMS depth sources with learned queries.

    ``blocks`` holds [B, L, H] FP32 sources, ``partial_n`` is the current
    partial sum (also unit-RMS) and ``query`` is [Q, H]. Returns [B, L, Q, H].
    """
    if mx.default_device() != mx.gpu:
        src = mx.stack([*blocks, partial_n], axis=-2)
        logits = (src @ query.T) * scale
        w = mx.softmax(logits, axis=-2, precise=True)
        return w.swapaxes(-1, -2) @ src
    b, length, hidden = partial_n.shape
    nq = query.shape[0]
    rows = b * length
    kernel = _attn_res_kernel(len(blocks), nq)
    out = kernel(
        inputs=[
            *[x.astype(mx.float32) for x in blocks],
            partial_n.astype(mx.float32),
            query.astype(mx.float32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("H", hidden), ("Q", nq), ("TG", _ATTN_RES_TG)],
        grid=(rows * _ATTN_RES_TG, 1, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=[(b, length, nq, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def _decode_attention(q, k, v, rel, sink, scale, window):
    s = k.shape[2]
    lo = max(0, s - window) if window else 0
    if lo:
        k = k[:, :, lo:]
        v = v[:, :, lo:]
    sv = s - lo
    r = rel.shape[-1]
    n = min(r, sv)
    hq = q.shape[1]
    tail = rel[0, 0, :, :n][:, ::-1]
    if n < sv:
        tail = mx.concatenate([mx.zeros((hq, sv - n), dtype=tail.dtype), tail], -1)
    mask = tail.reshape(1, hq, 1, sv).astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)
    )


def _build_fp8_e4m3_lut_header() -> str:
    """256-entry exact OCP FP8 E4M3 lookup table in Metal constant memory."""
    vals = []
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            v = 0.0
        else:
            s = -1.0 if (c & 0x80) else 1.0
            e = (c >> 3) & 0xF
            m = c & 0x7
            v = s * (
                (m / 8.0) * (2.0**-6)
                if e == 0
                else (1.0 + m / 8.0) * (2.0 ** (e - 7))
            )
        vals.append(f"half({v:.8e}f)")
    return (
        "constant half FP8_E4M3_LUT[256] = {"
        + ",".join(vals)
        + "};\n"
        + """
inline half fp8_e4m3_to_half(uchar code) {
    return FP8_E4M3_LUT[code];
}

inline uchar float_to_fp8_e4m3(float x) {
    uint u = as_type<uint>(x);
    uint sign = (u >> 24) & 0x80u;
    float ax = metal::fabs(x);
    if (!(ax < 448.0f)) {
        return (ax >= 448.0f) ? uchar(sign | 0x7Eu) : uchar(0u);
    }
    if (ax < 0.015625f) {
        uint q = uint(metal::rint(ax * 512.0f));
        return uchar(sign | q);
    }
    uint ua = as_type<uint>(ax);
    uint lsb = (ua >> 20) & 1u;
    uint rounded = ua + 0x7FFFFu + lsb;
    uint e = (rounded >> 23) - 120u;
    uint m = (rounded >> 20) & 0x7u;
    uint code = (e << 3) | m;
    if (code > 0x7Eu) code = 0x7Eu;
    return uchar(sign | code);
}
"""
    )


# Unit-scale IEEE/OCP FP8 E4M3 (1 sign, 4 exp bias 7, 3 mantissa; max +-448.0,
# no inf; overflow clamps to +-448.0 so NaN codes 0x7F/0xFF are never emitted).
_FP8_E4M3_HEADER = _build_fp8_e4m3_lut_header()

_FP8_QUANTIZE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (VEC4) {
        uint base = idx * 4;
        if (base + 3 < N) {
            uchar4 out4;
            out4.x = float_to_fp8_e4m3(float(x[base + 0]));
            out4.y = float_to_fp8_e4m3(float(x[base + 1]));
            out4.z = float_to_fp8_e4m3(float(x[base + 2]));
            out4.w = float_to_fp8_e4m3(float(x[base + 3]));
            *((device uchar4*)(out + base)) = out4;
        }
    } else if (idx < N) {
        out[idx] = float_to_fp8_e4m3(float(x[idx]));
    }
"""

_FP8_DEQUANTIZE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx < N) {
        out[idx] = OT(fp8_e4m3_to_half(x[idx]));
    }
"""

_fp8_quantize_kernel = None
_fp8_dequantize_kernel = None


def fp8_e4m3_quantize(x: mx.array) -> mx.array:
    """Quantize ``x`` to unit-scale OCP FP8 E4M3 codes (``uint8``, clamped to +-448)."""
    global _fp8_quantize_kernel
    if mx.default_device() == mx.gpu and x.size > 0:
        if _fp8_quantize_kernel is None:
            _fp8_quantize_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_fp8_quantize",
                input_names=["x"],
                output_names=["out"],
                source=_FP8_QUANTIZE_SOURCE,
                header=_FP8_E4M3_HEADER,
            )
        xc = mx.contiguous(x)
        n = xc.size
        vec4 = 1 if (n % 4 == 0) else 0
        grid_x = (n // 4) if vec4 else n
        return _fp8_quantize_kernel(
            inputs=[xc],
            template=[("T", xc.dtype), ("N", n), ("VEC4", vec4)],
            grid=(grid_x, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[xc.shape],
            output_dtypes=[mx.uint8],
        )[0]
    # CPU reference: exact IEEE/OCP FP8 E4M3 round-to-nearest-even.
    u32 = lambda val: mx.array(val, dtype=mx.uint32)
    xf = x.astype(mx.float32)
    u = xf.view(mx.uint32)
    sign = (u >> u32(24)) & u32(0x80)
    ax = mx.abs(xf)
    is_nan = mx.isnan(xf)
    ax_safe = mx.where(is_nan, mx.array(0.0, dtype=mx.float32), ax)
    sub = mx.round(mx.minimum(ax_safe, 0.015625) * 512.0).astype(mx.uint32)
    ua = mx.maximum(ax_safe, 0.015625).view(mx.uint32)
    lsb = (ua >> u32(20)) & u32(1)
    rounded = ua + u32(0x7FFFF) + lsb
    e = (rounded >> u32(23)) - u32(120)
    m = (rounded >> u32(20)) & u32(0x7)
    norm = mx.minimum((e << u32(3)) | m, u32(0x7E))
    mag = mx.where(ax_safe < 0.015625, sub, norm)
    mag = mx.where(ax_safe >= 448.0, u32(0x7E), mag)
    code = mx.where(is_nan, u32(0), sign | mag)
    return code.astype(mx.uint8)


def fp8_e4m3_dequantize(x: mx.array, dtype=mx.bfloat16) -> mx.array:
    """Dequantize unit-scale OCP FP8 E4M3 ``uint8`` codes to ``dtype``."""
    global _fp8_dequantize_kernel
    if mx.default_device() == mx.gpu and x.size > 0:
        if _fp8_dequantize_kernel is None:
            _fp8_dequantize_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_fp8_dequantize",
                input_names=["x"],
                output_names=["out"],
                source=_FP8_DEQUANTIZE_SOURCE,
                header=_FP8_E4M3_HEADER,
            )
        xc = mx.contiguous(x)
        return _fp8_dequantize_kernel(
            inputs=[xc],
            template=[("OT", dtype), ("N", xc.size)],
            grid=(xc.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[xc.shape],
            output_dtypes=[dtype],
        )[0]
    u32 = lambda val: mx.array(val, dtype=mx.uint32)
    code = x.astype(mx.uint32)
    sgn = (code & u32(0x80)) << u32(8)
    mag = code & u32(0x7F)
    norm_h = (sgn | ((mag + u32(0x40)) << u32(7))).astype(mx.uint16).view(mx.float16)
    zero_h = sgn.astype(mx.uint16).view(mx.float16)
    sub_h = (
        (sgn | u32(0x2400) | (mag << u32(7))).astype(mx.uint16).view(mx.float16)
        - (sgn | u32(0x2400)).astype(mx.uint16).view(mx.float16)
    )
    return mx.where(mag >= 8, norm_h, mx.where(mag == 0, zero_h, sub_h)).astype(dtype)


# Flash attention with the relative-bias band and per-query sinks, for
# prefill-sized BF16 query blocks. One threadgroup runs 32 queries of one head
# in four simdgroups (8 rows each) over 16-key tiles staged in half. Scores
# live in the log2 domain; each row's online softmax starts from its sink
# logit, so the zero-value sink is exact without an extra key.
_FLASH_SOURCE = """
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int NT = 128;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int qt = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int lp = dims[2 + b];
    const int hk = h / (HQ / HKV);
    const int q0 = qt * BQ;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int a_row = S - L + row;
    const float qscale = scale[0] * LOG2E;

    // Q, K and V are BF16 (or unit-scale FP8 E4M3) values within half's range
    // (L2-clipped q/k, unit-RMS v), so staging them in half is exact; scores
    // accumulate in FP32 and the softmax scale is applied there. K is stored
    // transposed and both tiles are padded by 16 bytes per row against bank
    // conflicts.
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + row * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = row_ok ? float(sink[(b * L + row) * HQ + h]) * LOG2E : 0.0f;
    float l = 1.0f;
    const device T* relp = rel + ((size_t)(b * L + (row_ok ? row : 0)) * HQ + h) * R;

    const int a_first = S - L + q0;
    const int a_last = S - L + min(q0 + BQ, L) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int k_begin = (lo / BK) * BK;
    const device KT* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device KT* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const bool vec_kv = FP8_KV
        ? (k_strides[3] == 1 && v_strides[3] == 1
           && (k_strides[2] % 16 == 0) && (v_strides[2] % 16 == 0))
        : (k_strides[3] == 1 && v_strides[3] == 1);

    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (FP8_KV) {
            if (vec_kv) {
                // 16 FP8 E4M3 values (16 bytes) per load.
                for (int c = tid; c < BK * D / 16; c += NT) {
                    const int r = c / (D / 16);
                    const int col = (c % (D / 16)) * 16;
                    const int key = k0 + r;
                    uint4 kw = uint4(0);
                    uint4 vw = uint4(0);
                    if (key < S) {
                        kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                        vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                    }
                    threadgroup half* kd = Ks + col * LDK + r;
                    threadgroup half* vd = Vs + r * LDV + col;
                    for (int j = 0; j < 4; ++j) {
                        uint kword = kw[j];
                        uint vword = vw[j];
                        kd[(4 * j + 0) * LDK] = fp8_e4m3_to_half(uchar(kword & 0xFFu));
                        kd[(4 * j + 1) * LDK] = fp8_e4m3_to_half(uchar((kword >> 8) & 0xFFu));
                        kd[(4 * j + 2) * LDK] = fp8_e4m3_to_half(uchar((kword >> 16) & 0xFFu));
                        kd[(4 * j + 3) * LDK] = fp8_e4m3_to_half(uchar((kword >> 24) & 0xFFu));
                        vd[4 * j + 0] = fp8_e4m3_to_half(uchar(vword & 0xFFu));
                        vd[4 * j + 1] = fp8_e4m3_to_half(uchar((vword >> 8) & 0xFFu));
                        vd[4 * j + 2] = fp8_e4m3_to_half(uchar((vword >> 16) & 0xFFu));
                        vd[4 * j + 3] = fp8_e4m3_to_half(uchar((vword >> 24) & 0xFFu));
                    }
                }
            } else {
                for (int c = tid; c < BK * D; c += NT) {
                    const int r = c / D;
                    const int col = c % D;
                    const int key = k0 + r;
                    half kv = half(0.0h);
                    half vv = half(0.0h);
                    if (key < S) {
                        kv = fp8_e4m3_to_half(uchar(kb_ptr[key * k_strides[2] + col * k_strides[3]]));
                        vv = fp8_e4m3_to_half(uchar(vb_ptr[key * v_strides[2] + col * v_strides[3]]));
                    }
                    Ks[col * LDK + r] = kv;
                    Vs[r * LDV + col] = vv;
                }
            }
        } else if (vec_kv) {
            // 8 BF16 values (16 bytes) per load.
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < S) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = half(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < S) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        if (WINDOW == 0 && k0 >= lp && k0 + BK <= a_first - R + 1) {
            // Full-attention tile completely beyond the relative-bias band:
            // every key is strictly visible with zero relative bias.
            for (int kb = 0; kb < BK / 8; ++kb) {
                float s0 = Sf[kb].thread_elements()[0] * qscale;
                float s1 = Sf[kb].thread_elements()[1] * qscale;
                Sf[kb].thread_elements()[0] = s0;
                Sf[kb].thread_elements()[1] = s1;
                rmax = max(rmax, max(s0, s1));
            }
        } else {
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = a_row - key;
                    const bool vis = row_ok && d >= 0 && key >= lp
                        && (WINDOW == 0 || d < WINDOW);
                    float s = Sf[kb].thread_elements()[e] * qscale;
                    if (vis) {
                        if (d < R) {
                            s += float(relp[d]) * LOG2E;
                        }
                    } else {
                        s = -INFINITY;
                    }
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(Sf[kb].thread_elements()[e] - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + ((size_t)(b * L + row) * HQ + h) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"""
_flash_kernel = None
# Query blocks shorter than this keep the decode/verify path.
_FLASH_MIN_QUERIES = 32


# GQA-packed Split-K FlashDecoding kernel for short query blocks (1 <= L < 32:
# speculative verify and multi-stream decode). Packs all ``G * L`` queries
# sharing one KV head into rows ``0 .. G*L - 1`` so K and V are read once per
# KV head via 16-byte loads without copying K/V for a dummy sink row or
# materializing an additive mask.
_GQA_DECODE_PART_SOURCE = """
    constexpr int BK = 16;
    constexpr int NT = NSG * 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv_qt = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int L = dims[0];
    const int S = dims[1];
    const int NS = dims[2];
    const int lp = dims[3 + b];

    const int n_rows = G * L;
    const int BQ = NSG * 8;
    const int nqt = (n_rows + BQ - 1) / BQ;
    const int hkv = hkv_qt / nqt;
    const int qt = hkv_qt % nqt;

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int packed_row = qt * BQ + int(sg) * 8 + fm;
    const bool row_ok = packed_row < n_rows;
    const int g_idx = row_ok ? (packed_row / L) : 0;
    const int q_pos = row_ok ? (packed_row % L) : 0;
    const int h = hkv * G + g_idx;
    const int a_row = S - L + q_pos;
    const float qscale = scale[0] * LOG2E;

    const int a_first = S - L;
    const int a_last = S - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_last + 1 - lo);
    const int per_sp = ((nk + NS - 1) / NS + BK - 1) / BK * BK;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_last + 1, k_start + per_sp);

    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + q_pos * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    #pragma clang loop unroll(full)
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = (SINGLE && row_ok) ? float(sink[(b * L + q_pos) * HQ + h]) * LOG2E : -INFINITY;
    float l = SINGLE ? 1.0f : 0.0f;
    const device T* relp = rel + ((size_t)(b * L + q_pos) * HQ + h) * R;

    const device KT* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device KT* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const bool vec_kv = FP8_KV
        ? (k_strides[3] == 1 && v_strides[3] == 1
           && (k_strides[2] % 16 == 0) && (v_strides[2] % 16 == 0))
        : (k_strides[3] == 1 && v_strides[3] == 1
           && (k_strides[2] % 8 == 0) && (v_strides[2] % 8 == 0));

    for (int k0 = k_start; k0 < k_end; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (FP8_KV) {
            if (vec_kv) {
                for (int c = int(tid); c < BK * D / 16; c += NT) {
                    const int r = c / (D / 16);
                    const int col = (c % (D / 16)) * 16;
                    const int key = k0 + r;
                    uint4 kw = uint4(0);
                    uint4 vw = uint4(0);
                    if (key < k_end) {
                        kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                        vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                    }
                    threadgroup half* kd = Ks + col * LDK + r;
                    threadgroup half* vd = Vs + r * LDV + col;
                    for (int j = 0; j < 4; ++j) {
                        uint kword = kw[j];
                        uint vword = vw[j];
                        kd[(4 * j + 0) * LDK] = fp8_e4m3_to_half(uchar(kword & 0xFFu));
                        kd[(4 * j + 1) * LDK] = fp8_e4m3_to_half(uchar((kword >> 8) & 0xFFu));
                        kd[(4 * j + 2) * LDK] = fp8_e4m3_to_half(uchar((kword >> 16) & 0xFFu));
                        kd[(4 * j + 3) * LDK] = fp8_e4m3_to_half(uchar((kword >> 24) & 0xFFu));
                        vd[4 * j + 0] = fp8_e4m3_to_half(uchar(vword & 0xFFu));
                        vd[4 * j + 1] = fp8_e4m3_to_half(uchar((vword >> 8) & 0xFFu));
                        vd[4 * j + 2] = fp8_e4m3_to_half(uchar((vword >> 16) & 0xFFu));
                        vd[4 * j + 3] = fp8_e4m3_to_half(uchar((vword >> 24) & 0xFFu));
                    }
                }
            } else {
                for (int c = int(tid); c < BK * D; c += NT) {
                    const int r = c / D;
                    const int col = c % D;
                    const int key = k0 + r;
                    half kv = half(0.0h);
                    half vv = half(0.0h);
                    if (key < k_end) {
                        kv = fp8_e4m3_to_half(uchar(kb_ptr[key * k_strides[2] + col * k_strides[3]]));
                        vv = fp8_e4m3_to_half(uchar(vb_ptr[key * v_strides[2] + col * v_strides[3]]));
                    }
                    Ks[col * LDK + r] = kv;
                    Vs[r * LDV + col] = vv;
                }
            }
        } else if (vec_kv) {
            for (int c = int(tid); c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < k_end) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = half(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = int(tid); c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < k_end) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qt * BQ + int(sg) * 8 >= n_rows) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        if (WINDOW == 0 && k0 >= lp && k0 + BK <= k_end && k0 + BK <= a_first - R + 1) {
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK / 8; ++kb) {
                float s0 = row_ok ? Sf[kb].thread_elements()[0] * qscale : -INFINITY;
                float s1 = row_ok ? Sf[kb].thread_elements()[1] * qscale : -INFINITY;
                Sf[kb].thread_elements()[0] = s0;
                Sf[kb].thread_elements()[1] = s1;
                rmax = max(rmax, max(s0, s1));
            }
        } else {
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK / 8; ++kb) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = a_row - key;
                    const bool vis = row_ok && key < k_end && d >= 0 && key >= lp
                        && (WINDOW == 0 || d < WINDOW);
                    float s = Sf[kb].thread_elements()[e] * qscale;
                    if (vis) {
                        if (d < R) {
                            s += float(relp[d]) * LOG2E;
                        }
                    } else {
                        s = -INFINITY;
                    }
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = (m == -INFINITY || m_new == -INFINITY)
            ? 0.0f : fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        #pragma clang loop unroll(full)
        for (int kb = 0; kb < BK / 8; ++kb) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const float sv = Sf[kb].thread_elements()[e];
                const float p = (sv == -INFINITY) ? 0.0f : fast::exp2(sv - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        #pragma clang loop unroll(full)
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            #pragma clang loop unroll(full)
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        if (SINGLE) {
            const float inv = 1.0f / l;
            size_t base_o = (((size_t)b * HQ + h) * L + q_pos) * D;
            #pragma clang loop unroll(full)
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0] * inv);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1] * inv);
            }
        } else {
            size_t base_ml = (((size_t)b * HQ + h) * L + q_pos) * NS + sp;
            if (fn == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            #pragma clang loop unroll(full)
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0]);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1]);
            }
        }
    }
"""

_GQA_DECODE_RED_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const uint lane = thread_index_in_simdgroup;
    const int idx = threadgroup_position_in_grid.x;
    const int q_pos = idx % L;
    const int bh = idx / L;
    const int h = bh % HQ;
    const int b = bh / HQ;

    constexpr int D4 = D / 32;
    float m = float(sink[(b * L + q_pos) * HQ + h]) * LOG2E;
    float l = 1.0f;
    float ov[D4];
    for (int i = 0; i < D4; ++i) ov[i] = 0.0f;

    size_t ml_base = (size_t)idx * NS;
    for (int sp = 0; sp < NS; ++sp) {
        float lp = part_l[ml_base + sp];
        if (lp == 0.0f) continue;
        float mp = part_m[ml_base + sp];
        float m_new = max(m, mp);
        float a0 = fast::exp2(m - m_new);
        float a1 = fast::exp2(mp - m_new);
        l = l * a0 + lp * a1;
        m = m_new;
        const device float* pp = part_o + (ml_base + sp) * D;
        for (int i = 0; i < D4; ++i) {
            ov[i] = ov[i] * a0 + pp[int(lane) * D4 + i] * a1;
        }
    }
    float inv = 1.0f / l;
    device T* dst = out + (size_t)idx * D;
    for (int i = 0; i < D4; ++i) {
        dst[int(lane) * D4 + i] = T(ov[i] * inv);
    }
"""

# Register-streaming Split-K GQA decode kernel for L == 1 (1..6 query heads per
# KV head): zero threadgroup KV staging and zero barriers inside the key loop,
# reading each KV vector once across all G query heads via coalesced 4-byte
# FP8 E4M3 loads (or BF16 loads).
_GQA_DIRECT_DECODE_PART_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int D4 = D / 32;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int S = dims[0];
    const int NS = dims[1];
    const int lp = dims[2 + b];
    const float qscale = scale[0] * LOG2E;

    const int a_row = S - 1;
    int lo = WINDOW > 0 ? max(0, a_row - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_row + 1 - lo);
    const int per_sp = (nk + NS - 1) / NS;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_row + 1, k_start + per_sp);

    float qv[G][D4];
    float ov[G][D4];
    float mv[G];
    float lv[G];
    const device T* relp[G];
    const int d_off = int(lane) * D4;
    for (int g_idx = 0; g_idx < G; ++g_idx) {
        int h = hkv * G + g_idx;
        const device T* qp = q + b * q_strides[0] + h * q_strides[1]
            + d_off * q_strides[3];
        for (int i = 0; i < D4; ++i) {
            qv[g_idx][i] = float(qp[i * q_strides[3]]) * qscale;
            ov[g_idx][i] = 0.0f;
        }
        mv[g_idx] = -INFINITY;
        lv[g_idx] = 0.0f;
        relp[g_idx] = rel + ((size_t)b * HQ + h) * R;
    }

    const device KT* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device KT* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const size_t ks2 = k_strides[2];
    const size_t vs2 = v_strides[2];
    const size_t ks3 = k_strides[3];
    const size_t vs3 = v_strides[3];
    const bool vec_kv = FP8_KV && ks3 == 1 && vs3 == 1 && (D4 == 4)
        && (ks2 % 4 == 0) && (vs2 % 4 == 0);

    for (int key = k_start + int(sg); key < k_end; key += NSG) {
        float kv[D4];
        float vv[D4];
        if (FP8_KV && vec_kv) {
            uchar4 kw = *(const device uchar4*)(kb_ptr + key * ks2 + d_off);
            uchar4 vw = *(const device uchar4*)(vb_ptr + key * vs2 + d_off);
            kv[0] = float(FP8_E4M3_LUT[kw.x]);
            kv[1] = float(FP8_E4M3_LUT[kw.y]);
            kv[2] = float(FP8_E4M3_LUT[kw.z]);
            kv[3] = float(FP8_E4M3_LUT[kw.w]);
            vv[0] = float(FP8_E4M3_LUT[vw.x]);
            vv[1] = float(FP8_E4M3_LUT[vw.y]);
            vv[2] = float(FP8_E4M3_LUT[vw.z]);
            vv[3] = float(FP8_E4M3_LUT[vw.w]);
        } else if (FP8_KV) {
            const device KT* kp = kb_ptr + key * ks2;
            const device KT* vp = vb_ptr + key * vs2;
            for (int i = 0; i < D4; ++i) {
                kv[i] = float(FP8_E4M3_LUT[uchar(kp[(d_off + i) * ks3])]);
                vv[i] = float(FP8_E4M3_LUT[uchar(vp[(d_off + i) * vs3])]);
            }
        } else {
            const device KT* kp = kb_ptr + key * ks2;
            const device KT* vp = vb_ptr + key * vs2;
            for (int i = 0; i < D4; ++i) {
                kv[i] = float(kp[(d_off + i) * ks3]);
                vv[i] = float(vp[(d_off + i) * vs3]);
            }
        }
        const int d = a_row - key;
        const bool has_rel = d < R;
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            float dot = 0.0f;
            for (int i = 0; i < D4; ++i) dot += kv[i] * qv[g_idx][i];
            dot = simd_sum(dot);
            if (has_rel) dot += float(relp[g_idx][d]) * LOG2E;
            float m_new = max(mv[g_idx], dot);
            float f_old = (mv[g_idx] == -INFINITY) ? 0.0f : fast::exp2(mv[g_idx] - m_new);
            float p = fast::exp2(dot - m_new);
            lv[g_idx] = lv[g_idx] * f_old + p;
            mv[g_idx] = m_new;
            for (int i = 0; i < D4; ++i) {
                ov[g_idx][i] = ov[g_idx][i] * f_old + p * vv[i];
            }
        }
    }

    threadgroup float tg_m[NSG][G];
    threadgroup float tg_l[NSG][G];
    threadgroup float tg_o[NSG][G][D];
    for (int g_idx = 0; g_idx < G; ++g_idx) {
        if (lane == 0) {
            tg_m[sg][g_idx] = mv[g_idx];
            tg_l[sg][g_idx] = lv[g_idx];
        }
        for (int i = 0; i < D4; ++i) {
            tg_o[sg][g_idx][d_off + i] = ov[g_idx][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            int h = hkv * G + g_idx;
            float m = -INFINITY;
            float l = 0.0f;
            float oo[D4];
            for (int i = 0; i < D4; ++i) oo[i] = 0.0f;
            for (int s = 0; s < NSG; ++s) {
                float ls = tg_l[s][g_idx];
                if (ls == 0.0f) continue;
                float ms = tg_m[s][g_idx];
                float mn = max(m, ms);
                float a0 = (m == -INFINITY) ? 0.0f : fast::exp2(m - mn);
                float a1 = fast::exp2(ms - mn);
                l = l * a0 + ls * a1;
                m = mn;
                for (int i = 0; i < D4; ++i) {
                    oo[i] = oo[i] * a0 + tg_o[s][g_idx][d_off + i] * a1;
                }
            }
            size_t base_ml = ((size_t)b * HQ + h) * NS + sp;
            if (lane == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            for (int i = 0; i < D4; ++i) {
                out_o[base_o + d_off + i] = oo[i];
            }
        }
    }
"""

_gqa_decode_part_kernel = None
_gqa_direct_part_kernel = None
_gqa_decode_red_kernel = None


def gqa_flash_decode_attention(
    q, k, v, rel, sink, *, scale, window, left_padding=None
):
    """Fused Split-K GQA-packed banded-sink attention for BF16 verify/decode."""
    global _gqa_decode_part_kernel, _gqa_direct_part_kernel, _gqa_decode_red_kernel
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    g = hq // hkv
    r = rel.shape[-1]
    fp8_kv = k.dtype == mx.uint8
    if _gqa_decode_red_kernel is None:
        _gqa_decode_red_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_gqa_decode_red",
            input_names=["part_o", "part_m", "part_l", "sink"],
            output_names=["out"],
            source=_GQA_DECODE_RED_SOURCE,
        )
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    sc = mx.array([scale], dtype=mx.float32)
    sink_f32 = mx.contiguous(sink.astype(mx.float32))
    rel_c = mx.contiguous(rel)
    nk_est = min(s, window + length - 1) if window else s
    if length == 1 and g <= 6 and d <= 128 and (fp8_kv or nk_est >= 2048):
        if _gqa_direct_part_kernel is None:
            _gqa_direct_part_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_gqa_direct_part",
                input_names=["q", "k", "v", "rel", "dims", "scale"],
                output_names=["out_o", "out_m", "out_l"],
                source=_GQA_DIRECT_DECODE_PART_SOURCE,
                header=_FP8_E4M3_HEADER,
                ensure_row_contiguous=False,
            )
        nsg = 4
        base_tgs = b * hkv
        ns = (
            1
            if nk_est <= 256
            else min(32, max(1, (nk_est + 255) // 256))
        )
        if nk_est >= 2048 and ns < 16 and base_tgs <= 16:
            ns = 16
        if nk_est >= 8192 and ns < 32 and base_tgs <= 16:
            ns = 32
        dims = mx.concatenate([mx.array([s, ns], dtype=mx.int32), lp])
        po, pm, pl = _gqa_direct_part_kernel(
            inputs=[q, k, v, rel_c, dims, sc],
            template=[
                ("T", q.dtype),
                ("KT", k.dtype),
                ("D", d),
                ("HQ", hq),
                ("G", g),
                ("R", r),
                ("WINDOW", int(window)),
                ("NSG", nsg),
                ("FP8_KV", 1 if fp8_kv else 0),
            ],
            grid=(ns * nsg * 32, hkv, b),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[
                (b, hq, 1, ns, d),
                (b, hq, 1, ns),
                (b, hq, 1, ns),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return _gqa_decode_red_kernel(
            inputs=[po, pm, pl, sink_f32],
            template=[("T", q.dtype), ("D", d), ("HQ", hq), ("L", 1), ("NS", ns)],
            grid=(b * hq * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(b, hq, 1, d)],
            output_dtypes=[q.dtype],
        )[0]
    if _gqa_decode_part_kernel is None:
        _gqa_decode_part_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_gqa_decode_part",
            input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
            output_names=["out_o", "out_m", "out_l"],
            source=_GQA_DECODE_PART_SOURCE,
            header=_FP8_E4M3_HEADER,
            ensure_row_contiguous=False,
        )
    n_rows = g * length
    nsg = min(4, max(1, (n_rows + 7) // 8))
    bq = nsg * 8
    nqt = (n_rows + bq - 1) // bq
    # Match bounded vs unbounded sliding caches identically: once keys drop,
    # s >= window + length - 1 so nk_est is invariant to dropped prefix keys.
    base_tgs = b * hkv * nqt
    ns = (
        1
        if nk_est <= 128
        else min(
            32,
            max(1, min((nk_est + 63) // 64, (64 + base_tgs - 1) // base_tgs)),
        )
    )
    if nk_est >= 512 and ns < 8 and base_tgs <= 16:
        ns = 8
    if nk_est >= 2048 and ns < 16 and base_tgs <= 16:
        ns = 16
    if nk_est >= 8192 and ns < 32 and base_tgs <= 16:
        ns = 32
    dims = mx.concatenate([mx.array([length, s, ns], dtype=mx.int32), lp])
    tpl = [
        ("T", q.dtype),
        ("KT", k.dtype),
        ("OT", q.dtype if ns == 1 else mx.float32),
        ("D", d),
        ("HQ", hq),
        ("G", g),
        ("R", r),
        ("WINDOW", int(window)),
        ("NSG", nsg),
        ("SINGLE", 1 if ns == 1 else 0),
        ("FP8_KV", 1 if fp8_kv else 0),
    ]
    nt = nsg * 32
    if ns == 1:
        out, _, _ = _gqa_decode_part_kernel(
            inputs=[q, k, v, rel_c, sink_f32, dims, sc],
            template=tpl,
            grid=(nt, hkv * nqt, b),
            threadgroup=(nt, 1, 1),
            output_shapes=[(b, hq, length, d), (1,), (1,)],
            output_dtypes=[q.dtype, mx.float32, mx.float32],
        )
        return out
    po, pm, pl = _gqa_decode_part_kernel(
        inputs=[q, k, v, rel_c, sink_f32, dims, sc],
        template=tpl,
        grid=(ns * nt, hkv * nqt, b),
        threadgroup=(nt, 1, 1),
        output_shapes=[
            (b, hq, length, ns, d),
            (b, hq, length, ns),
            (b, hq, length, ns),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _gqa_decode_red_kernel(
        inputs=[po, pm, pl, sink_f32],
        template=[("T", q.dtype), ("D", d), ("HQ", hq), ("L", length), ("NS", ns)],
        grid=(b * hq * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]


def _use_flash(length: int, window: int) -> bool:
    """Prefill blocks take the flash kernel on the GPU.

    On hosts with tensor units (NAX) MLX's masked SDPA runs on them, which
    this kernel does not; there only sliding layers, whose cost is dominated
    by building the mask, switch to it.
    """
    if length < _FLASH_MIN_QUERIES or mx.default_device() != mx.gpu:
        return False
    return bool(window) or not is_nax_available()


def flash_banded_attention(q, k, v, rel, sink, *, scale, window, left_padding=None):
    """``banded_sink_attention`` for BF16 prefill blocks (BF16 or FP8 E4M3 K/V)."""
    global _flash_kernel
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    fp8_kv = k.dtype == mx.uint8
    if _flash_kernel is None:
        _flash_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_banded",
            input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
            output_names=["out"],
            source=_FLASH_SOURCE,
            header=_FP8_E4M3_HEADER,
            ensure_row_contiguous=False,
        )
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    dims = mx.concatenate([mx.array([length, s], dtype=mx.int32), lp])
    out = _flash_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sink.astype(mx.float32)),
            dims,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("KT", k.dtype),
            ("D", d),
            ("HQ", hq),
            ("HKV", hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
            ("FP8_KV", 1 if fp8_kv else 0),
        ],
        grid=(((length + 31) // 32) * 128, hq, b),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, length, hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


def banded_sink_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sink: mx.array,
    *,
    scale: float,
    window: int = 0,
    left_padding=None,
) -> mx.array:
    """Attention with a relative bias band and per-query sink logits.

    q: [B, Hq, L, D]; k, v: [B, Hkv, S, D] (queries are the last L keys; may be
    ``uint8`` unit-scale OCP FP8 E4M3 codes); rel: [B, L, Hq, R] bias by
    backward distance d (applied for d < R); sink: [B, L, Hq] logits of a
    zero-value sink. ``window`` > 0 limits keys to d < window (sliding layers
    use window == R).
    """
    b, hq, length, d = q.shape
    fp8_kv = k.dtype == mx.uint8
    if b == 1 and length == 1 and left_padding is None and not fp8_kv:
        return _decode_attention(q, k, v, rel, sink, scale, window)
    if q.dtype == mx.bfloat16 and (
        _use_flash(length, window)
        or (fp8_kv and length >= _FLASH_MIN_QUERIES and mx.default_device() == mx.gpu)
    ):
        if (
            d == 128
            and k.dtype in (mx.bfloat16, mx.uint8)
            and hq % k.shape[1] == 0
            and can_use_nax_attn()
        ):
            return flash_banded_attention_nax(
                q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
            )
        return flash_banded_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    if (
        q.dtype == mx.bfloat16
        and length < _FLASH_MIN_QUERIES
        and mx.default_device() == mx.gpu
        and d % 32 == 0
        and hq % k.shape[1] == 0
    ):
        return gqa_flash_decode_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    return _masked_attention(q, k, v, rel, sink, scale, window, left_padding)


def _masked_attention(q, k, v, rel, sink, scale, window, left_padding):
    """MLX SDPA over an additive mask that carries the bias band and sinks."""
    if k.dtype == mx.uint8:
        k = fp8_e4m3_dequantize(k, q.dtype)
        v = fp8_e4m3_dequantize(v, q.dtype)
    b, hq, length, d = q.shape
    s = k.shape[2]
    r = rel.shape[-1]
    if window:
        assert window == r, "sliding layers expect rel_extent == window"
    lo = max(0, s - length - window + 1) if window else 0
    hkv = k.shape[1]
    # A zero key/value row carries the sink: its logit comes from the mask.
    zero = mx.zeros((b, hkv, 1, d), dtype=k.dtype)
    kz = mx.concatenate([zero, k[:, :, lo:]], axis=2)
    vz = mx.concatenate([zero, v[:, :, lo:]], axis=2)
    nk = s - lo
    qb = max(16, min(length, _MASK_BUDGET // max(1, b * hq * (nk + 1))))
    neg = mx.array(-mx.inf, dtype=q.dtype)
    lp_ok = None
    if left_padding is not None:
        lp_ok = (lo + mx.arange(nk))[None, :] >= left_padding[:, None]
        lp_ok = mx.concatenate([mx.ones((b, 1), dtype=mx.bool_), lp_ok], axis=1)
        lp_ok = lp_ok[:, None, None, :]

    outs = []
    for i0 in range(0, length, qb):
        i1 = min(length, i0 + qb)
        nq = i1 - i0
        a0 = s - length + i0
        hi = s - length + i1
        f = max(lo, a0 - r + 1)
        near = hi - f
        dist = (a0 - f) + mx.arange(nq)[:, None] - mx.arange(near)[None, :]
        relq = rel[:, i0:i1].transpose(0, 2, 1, 3)
        gidx = mx.broadcast_to(mx.clip(dist, 0, r - 1), (b, hq, nq, near))
        bias = mx.take_along_axis(relq, gidx, axis=-1)
        vis = dist >= 0
        if window:
            vis = vis & (dist < window)
        near_mask = mx.where(vis, mx.where(dist < r, bias, 0), neg)
        parts = [sink[:, i0:i1].transpose(0, 2, 1)[..., None].astype(q.dtype)]
        if f > lo:
            fill = neg if window else mx.array(0, dtype=q.dtype)
            parts.append(mx.broadcast_to(fill, (b, hq, nq, f - lo)))
        parts.append(near_mask.astype(q.dtype))
        if hi < s:
            parts.append(mx.broadcast_to(neg, (b, hq, nq, s - hi)))
        mask = mx.concatenate(parts, axis=-1)
        if lp_ok is not None:
            mask = mx.where(lp_ok, mask, neg)
        out = mx.fast.scaled_dot_product_attention(
            q[:, :, i0:i1], kz, vz, scale=scale, mask=mask
        )
        if qb < length:
            # A lazy graph would hold every block's mask at once (12 GiB at
            # 64K keys); evaluated block by block, one mask is alive.
            mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


_ROUTER_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    constexpr uint PER = E / 32;
    float sc[PER];
    float key[PER];
    for (uint j = 0; j < PER; ++j) {
        uint e = lane + 32 * j;
        float s = 1.0f / (1.0f + metal::exp(-logits[row * E + e]));
        sc[j] = s;
        key[j] = s + bias[e];
    }
    float picked[K];
    uint chosen[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) {
                best = key[j];
                bj = j;
            }
        }
        float top = simd_max(best);
        uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
        uint win = simd_min(mine);
        float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
        s = simd_sum(s);
        if (lane == win % 32) key[win / 32] = -INFINITY;
        picked[k] = s;
        chosen[k] = win;
        total += s;
    }
    if (lane == 0) {
        float norm = route_scale[0] / (total + 1e-20f);
        for (uint k = 0; k < K; ++k) {
            inds[row * K + k] = chosen[k];
            weights[row * K + k] = picked[k] * norm;
        }
    }
"""
_router_kernel = None


def route_topk(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """Sigmoid scores, top-k by ``score + bias``, renormalized unbiased weights.

    ``logits``: [N, E] FP32. Returns ([N, K] uint32 indices, [N, K] FP32).
    """
    global _router_kernel
    rows, experts = logits.shape
    if mx.default_device() != mx.gpu or experts % 32:
        scores = mx.sigmoid(logits)
        inds = mx.argpartition(-(scores + bias), kth=top_k - 1, axis=-1)[..., :top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20) * route_scale
        return inds, w
    if _router_kernel is None:
        _router_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_route_topk",
            input_names=["logits", "bias", "route_scale"],
            output_names=["inds", "weights"],
            source=_ROUTER_SOURCE,
        )
    inds, weights = _router_kernel(
        inputs=[
            logits.astype(mx.float32),
            bias.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
        ],
        template=[("E", experts), ("K", top_k)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, weights


# Fused loop-controller kernels for ``_mix_messages``:
# - ``_NORM_PAIR_SOURCE`` computes unit-RMS of ``x`` and ``msg``, rounds to
#   BF16, and writes the stacked [N, 2, ROWS, H] array in either BF16 (small-M
#   fused path) or FP32 (large-M GEMM path) in one pass.
# - ``_LOOP_DOWN_SOURCE`` projects [N, 2, ROWS, H] BF16 against transposed
#   ``down_t`` [N, 2, R, H] (BF16 or FP32), accumulating both halves in FP32
#   and reading ``down_t`` once across all ``ROWS`` (exact chunk invariance).
# - ``_LOOP_GMIX_SOURCE`` projects ``low`` [N, ROWS, R] BF16 against transposed
#   ``gf_t`` [N, H, 2, R] (BF16 or FP32), applies ``tanh`` to gain/forget in
#   registers, and writes ``(1 - forget) * x + gain * msg`` directly without
#   materializing ``gates``.
# - ``_TANH_MIX_SOURCE`` fuses ``tanh(gain/forget)`` + residual mix for the
#   large-M GEMM path.
_NORM_PAIR_SOURCE = """
    uint row_idx = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint total_rows = N * ROWS;
    uint is_msg = row_idx >= total_rows ? 1u : 0u;
    uint nr = is_msg ? (row_idx - total_rows) : row_idx;
    uint n = nr / ROWS;
    uint r = nr % ROWS;
    const device float* src = (is_msg ? msg : x) + (size_t)nr * H;
    float ss = 0.0f;
    for (uint h = tid; h < H; h += TG) {
        float v = src[h];
        ss += v * v;
    }
    threadgroup float red[TG / 32];
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < TG / 32; ++i) total += red[i];
    float inv = metal::rsqrt(total / float(H) + 1e-6f);
    device OT* dst = out + ((size_t)(n * 2 + is_msg) * ROWS + r) * H;
    for (uint h = tid; h < H; h += TG) {
        bfloat bv = bfloat(src[h] * inv);
        dst[h] = OT(bv);
    }
"""

_LOOP_DOWN_SOURCE = """
    uint nr = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint n = nr / R;
    uint r = nr % R;
    float acc[ROWS];
    for (uint i = 0; i < ROWS; ++i) acc[i] = 0.0f;
    for (uint part = 0; part < 2; ++part) {
        const device WT* wp = down_t + ((size_t)(n * 2 + part) * R + r) * H;
        const device bfloat* zp = z + (size_t)(n * 2 + part) * ROWS * H;
        for (uint h = lane * 4; h < H; h += 128) {
            float w0 = float(wp[h + 0]);
            float w1 = float(wp[h + 1]);
            float w2 = float(wp[h + 2]);
            float w3 = float(wp[h + 3]);
            for (uint row = 0; row < ROWS; ++row) {
                const device bfloat* zr = zp + (size_t)row * H + h;
                acc[row] += float(zr[0]) * w0 + float(zr[1]) * w1
                    + float(zr[2]) * w2 + float(zr[3]) * w3;
            }
        }
    }
    for (uint row = 0; row < ROWS; ++row) {
        float s = simd_sum(acc[row]);
        if (lane == 0) {
            low[((size_t)n * ROWS + row) * R + r] = bfloat(s);
        }
    }
"""

_LOOP_GMIX_SOURCE = """
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint nh = threadgroup_position_in_grid.x * 4 + sg;
    if (nh >= N * H) return;
    uint n = nh / H;
    uint h = nh % H;
    constexpr uint PER = (R + 31) / 32;
    float wg[PER], wf[PER];
    const device WT* gp = gf_t + ((size_t)nh * 2 + 0) * R;
    const device WT* fp = gf_t + ((size_t)nh * 2 + 1) * R;
    for (uint j = 0; j < PER; ++j) {
        uint idx = lane + j * 32;
        wg[j] = idx < R ? float(gp[idx]) : 0.0f;
        wf[j] = idx < R ? float(fp[idx]) : 0.0f;
    }
    for (uint row = 0; row < ROWS; ++row) {
        const device bfloat* lp = low + ((size_t)n * ROWS + row) * R;
        float dg = 0.0f;
        float df = 0.0f;
        for (uint j = 0; j < PER; ++j) {
            uint idx = lane + j * 32;
            float lv = idx < R ? float(lp[idx]) : 0.0f;
            dg += lv * wg[j];
            df += lv * wf[j];
        }
        dg = simd_sum(dg);
        df = simd_sum(df);
        if (lane == 0) {
            float g = metal::tanh(metal::clamp(dg, -20.0f, 20.0f));
            float f = metal::tanh(metal::clamp(df, -20.0f, 20.0f));
            size_t pos = ((size_t)n * ROWS + row) * H + h;
            out[pos] = (1.0f - f) * x[pos] + g * msg[pos];
        }
    }
"""

_TANH_MIX_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= N * ROWS * H) return;
    uint nr = idx / H;
    uint h = idx % H;
    size_t gbase = (size_t)nr * (2 * H);
    float g = metal::tanh(metal::clamp(float(gates[gbase + h]), -20.0f, 20.0f));
    float f = metal::tanh(metal::clamp(float(gates[gbase + H + h]), -20.0f, 20.0f));
    out[idx] = (1.0f - f) * x[idx] + g * msg[idx];
"""

_norm_pair_kernel = None
_loop_down_kernel = None
_loop_gmix_kernel = None
_tanh_mix_kernel = None
_LOOP_FUSED_MAX_ROWS = 16


def pack_loop_controller(down: mx.array, gain_forget: mx.array):
    """Transpose loop controller weights for coalesced decode/verify reads.

    Real checkpoints widen BF16 controller weights to FP32 at load; when the
    values are exact BF16, pack as BF16 to halve memory bandwidth with zero
    numerical loss.
    """
    n, _, h, r = down.shape
    exact_bf16 = (
        mx.array_equal(down.astype(mx.bfloat16).astype(mx.float32), down).item()
        and mx.array_equal(
            gain_forget.astype(mx.bfloat16).astype(mx.float32), gain_forget
        ).item()
    )
    dtype = mx.bfloat16 if exact_bf16 else down.dtype
    down_t = mx.contiguous(down.astype(dtype).transpose(0, 1, 3, 2))
    gf_t = mx.contiguous(
        gain_forget.astype(dtype).reshape(n, r, 2, h).transpose(0, 3, 2, 1)
    )
    mx.eval(down_t, gf_t)
    return down_t, gf_t


def mix_messages_gpu(
    x: mx.array,
    message: mx.array,
    down: mx.array,
    gain_forget: mx.array,
    down_t: mx.array | None = None,
    gf_t: mx.array | None = None,
) -> mx.array:
    """Fused GPU loop-controller message mixing; exact FP32 accumulation."""
    global _norm_pair_kernel, _loop_down_kernel, _loop_gmix_kernel, _tanh_mix_kernel
    n, b, length, hidden = x.shape
    rows = b * length
    rank = down.shape[-1]
    if _norm_pair_kernel is None:
        _norm_pair_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_norm_pair",
            input_names=["x", "msg"],
            output_names=["out"],
            source=_NORM_PAIR_SOURCE,
        )
    if (
        rows <= _LOOP_FUSED_MAX_ROWS
        and hidden % 4 == 0
        and down_t is not None
        and gf_t is not None
    ):
        if _loop_down_kernel is None:
            _loop_down_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_loop_down",
                input_names=["z", "down_t"],
                output_names=["low"],
                source=_LOOP_DOWN_SOURCE,
            )
            _loop_gmix_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_loop_gmix",
                input_names=["x", "msg", "low", "gf_t"],
                output_names=["out"],
                source=_LOOP_GMIX_SOURCE,
            )
        z = _norm_pair_kernel(
            inputs=[x.astype(mx.float32), message.astype(mx.float32)],
            template=[
                ("OT", mx.bfloat16),
                ("N", n),
                ("ROWS", rows),
                ("H", hidden),
                ("TG", 256),
            ],
            grid=(2 * n * rows * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, 2, rows, hidden)],
            output_dtypes=[mx.bfloat16],
        )[0]
        low = _loop_down_kernel(
            inputs=[z, down_t],
            template=[
                ("WT", down_t.dtype),
                ("R", rank),
                ("H", hidden),
                ("ROWS", rows),
            ],
            grid=(n * rank * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(n, rows, rank)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return _loop_gmix_kernel(
            inputs=[x.astype(mx.float32), message.astype(mx.float32), low, gf_t],
            template=[
                ("WT", gf_t.dtype),
                ("N", n),
                ("H", hidden),
                ("R", rank),
                ("ROWS", rows),
            ],
            grid=(((n * hidden + 3) // 4) * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n, b, length, hidden)],
            output_dtypes=[mx.float32],
        )[0]

    # Large-M (prefill) path: keep exact FP32 GEMMs (chunk-invariant) while
    # fusing norm+BF16-round and tanh+residual-mix into single-pass kernels.
    if _tanh_mix_kernel is None:
        _tanh_mix_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_tanh_mix",
            input_names=["x", "msg", "gates"],
            output_names=["out"],
            source=_TANH_MIX_SOURCE,
        )
    xf = x.astype(mx.float32)
    mf = message.astype(mx.float32)
    z = _norm_pair_kernel(
        inputs=[xf, mf],
        template=[
            ("OT", mx.float32),
            ("N", n),
            ("ROWS", rows),
            ("H", hidden),
            ("TG", 256),
        ],
        grid=(2 * n * rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(2 * n, rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    low = (z @ down.reshape(2 * n, hidden, rank)).reshape(n, 2, rows, rank).sum(axis=1)
    low = low.astype(mx.bfloat16).astype(mx.float32)
    gates = low @ gain_forget
    return _tanh_mix_kernel(
        inputs=[xf, mf, gates],
        template=[("N", n), ("ROWS", rows), ("H", hidden)],
        grid=(n * rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, b, length, hidden)],
        output_dtypes=[mx.float32],
    )[0]


# Fused expert activation (optional NVFP4 post_scale + interleaved de-interleave
# + softcapped SwiGLU), weighted expert reduction (optional NVFP4 post_scale +
# FP32 sum over top-k experts), and static depthwise causal conv (K=4) with
# residual add and direct next-state output.
_EXPERT_SWIGLU_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= M * I) return;
    uint m = idx / I;
    uint i = idx % I;
    size_t base = (size_t)m * (2 * I) + 2 * i;
    float g = float(y[base]);
    float u = float(y[base + 1]);
    if (HAS_SCALE) {
        T sc = T(gscale[eidx[m]] * params[2]);
        g = float(T(g * float(sc)));
        u = float(T(u * float(sc)));
    }
    float gcap = params[0];
    float ucap = params[1];
    float eg = metal::exp(metal::clamp(-g, -80.0f, 80.0f));
    float sig = (g < -80.0f) ? 0.0f : (1.0f / (1.0f + eg));
    float tg = metal::tanh(metal::clamp(g / gcap, -20.0f, 20.0f)) * gcap;
    float tu = metal::tanh(metal::clamp(u / ucap, -20.0f, 20.0f)) * ucap;
    out[idx] = T(tg * sig * tu);
"""

_EXPERT_COMBINE_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= ROWS * H) return;
    uint row = idx / H;
    uint h = idx % H;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        size_t rk = (size_t)row * K + k;
        float w = weights[rk];
        if (HAS_SCALE) {
            w *= gscale[inds[rk]] * inv_denom[0];
        }
        acc += float(z[rk * H + h]) * w;
    }
    out[idx] = T(acc);
"""

_STATIC_CONV4_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= B * L * C) return;
    uint c = idx % C;
    uint bl = idx / C;
    uint l = bl % L;
    uint b = bl / L;
    size_t ss0 = state_strides[0], ss1 = state_strides[1], ss2 = state_strides[2];
    const device float* st = state + (size_t)b * ss0 + (size_t)c * ss2;
    size_t xs0 = x_strides[0], xs1 = x_strides[1], xs2 = x_strides[2];
    const device T* xb = x + (size_t)b * xs0 + (size_t)c * xs2;
    size_t ws0 = w_strides[0], ws1 = w_strides[1];
    const device WT* wp = w + (size_t)c * ws0;
    float x0 = (l >= 3) ? float(xb[(size_t)(l - 3) * xs1]) : st[(size_t)l * ss1];
    float x1 = (l >= 2) ? float(xb[(size_t)(l - 2) * xs1]) : st[(size_t)(l + 1) * ss1];
    float x2 = (l >= 1) ? float(xb[(size_t)(l - 1) * xs1]) : st[(size_t)(l + 2) * ss1];
    float x3 = float(xb[(size_t)l * xs1]);
    float conv = float(wp[0]) * x0 + float(wp[ws1]) * x1
        + float(wp[2 * ws1]) * x2 + float(wp[3 * ws1]) * x3;
    float y = float(T((float(WT(conv)) + x3) * scale[0]));
    if (HAS_RES) {
        size_t roff = (size_t)b * res_strides[0] + (size_t)l * res_strides[1]
            + (size_t)c * res_strides[2];
        out[idx] = OT(float(res[roff]) + y);
    } else {
        out[idx] = OT(y);
    }
    if (L >= 3) {
        if (l >= L - 3) {
            next_state[((size_t)b * 3 + (l - (L - 3))) * C + c] = x3;
        }
    } else if (l == L - 1) {
        device float* nst = next_state + (size_t)b * 3 * C;
        if (L == 1) {
            nst[0 * C + c] = st[1 * ss1];
            nst[1 * C + c] = st[2 * ss1];
            nst[2 * C + c] = x3;
        } else {
            nst[0 * C + c] = st[2 * ss1];
            nst[1 * C + c] = x2;
            nst[2 * C + c] = x3;
        }
    }
"""

_expert_swiglu_kernel = None
_expert_combine_kernel = None
_static_conv4_kernel = None
_DUMMY_F32 = None


def _dummy_f32() -> mx.array:
    global _DUMMY_F32
    if _DUMMY_F32 is None:
        _DUMMY_F32 = mx.zeros((1,), dtype=mx.float32)
    return _DUMMY_F32


def expert_softcapped_swiglu(
    y: mx.array,
    idx: mx.array,
    gscale: mx.array | None,
    params: mx.array,
) -> mx.array:
    """Fused optional NVFP4 post-scale, de-interleave, and softcapped SwiGLU."""
    global _expert_swiglu_kernel
    if _expert_swiglu_kernel is None:
        _expert_swiglu_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_expert_swiglu",
            input_names=["y", "eidx", "gscale", "params"],
            output_names=["out"],
            source=_EXPERT_SWIGLU_SOURCE,
        )
    inter = y.shape[-1] // 2
    m = idx.size
    has_scale = 1 if gscale is not None else 0
    return _expert_swiglu_kernel(
        inputs=[y, idx.astype(mx.uint32), gscale if has_scale else _dummy_f32(), params],
        template=[("T", y.dtype), ("M", m), ("I", inter), ("HAS_SCALE", has_scale)],
        grid=(m * inter, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*y.shape[:-1], inter)],
        output_dtypes=[y.dtype],
    )[0]


def expert_weighted_combine(
    z: mx.array,
    inds: mx.array,
    weights: mx.array,
    gscale: mx.array | None,
    inv_denom: mx.array,
) -> mx.array:
    """Fused optional NVFP4 post-scale and FP32 weighted reduction over top-k experts."""
    global _expert_combine_kernel
    if _expert_combine_kernel is None:
        _expert_combine_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_expert_combine",
            input_names=["z", "inds", "weights", "gscale", "inv_denom"],
            output_names=["out"],
            source=_EXPERT_COMBINE_SOURCE,
        )
    rows, top_k, hidden = z.shape
    has_scale = 1 if gscale is not None else 0
    return _expert_combine_kernel(
        inputs=[
            z,
            inds.astype(mx.uint32),
            weights.astype(mx.float32),
            gscale if has_scale else _dummy_f32(),
            inv_denom,
        ],
        template=[
            ("T", z.dtype),
            ("ROWS", rows),
            ("H", hidden),
            ("K", top_k),
            ("HAS_SCALE", has_scale),
        ],
        grid=(rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[z.dtype],
    )[0]


def static_conv4_fused(
    x: mx.array,
    w: mx.array,
    state: mx.array,
    scale: mx.array,
    residual: mx.array | None = None,
    out_dtype=None,
) -> tuple[mx.array, mx.array]:
    """Fused K=4 depthwise causal conv, scale, optional residual add and next state."""
    global _static_conv4_kernel
    if _static_conv4_kernel is None:
        _static_conv4_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_static_conv4",
            input_names=["x", "w", "state", "scale", "res"],
            output_names=["out", "next_state"],
            source=_STATIC_CONV4_SOURCE,
            ensure_row_contiguous=False,
        )
    b, length, channels = x.shape
    has_res = 1 if residual is not None else 0
    res_arr = residual if has_res else _dummy_f32()
    odtype = x.dtype if out_dtype is None else out_dtype
    out, next_state = _static_conv4_kernel(
        inputs=[x, w, state.astype(mx.float32), scale, res_arr],
        template=[
            ("T", x.dtype),
            ("WT", w.dtype),
            ("RT", res_arr.dtype),
            ("OT", odtype),
            ("HAS_RES", has_res),
            ("B", b),
            ("L", length),
            ("C", channels),
        ],
        grid=(b * length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, length, channels), (b, 3, channels)],
        output_dtypes=[odtype, mx.float32],
    )
    return out, next_state


_DYNAMIC_CONV_HEADER = r"""
template <typename T, typename S, typename X>
inline float _ink2_dconv_padded(
    const device S* srow,
    int64_t ss,
    const device X* xrow,
    int64_t xs,
    int p,
    int i
) {
    return i < p ? float(srow[i * ss]) : float(T(xrow[(i - p) * xs]));
}
"""

_DYNAMIC_CONV_SOURCE = r"""
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int y = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = x_shape[1];
    const int C = x_shape[2];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const device T* xrow = x + b * x_strides[0] + c * x_strides[2];
    const int64_t ss = state_strides[1];
    const int64_t xs = x_strides[1];
    const float own = _ink2_dconv_padded<T>(srow, ss, xrow, xs, P, y);
    xp[((size_t)b * (P + L) + y) * C + c] = own;
    if (y < P) return;
    const int t = y - P;
    const int head = c / (C / H);
    const auto drow = dyn + b * dyn_strides[0] + t * dyn_strides[1]
        + head * K * dyn_strides[2];
    float acc = 0.0f;
    for (int lag = 0; lag < K; ++lag) {
        const float sw = float(static_w[lag * static_w_strides[0]
                                        + c * static_w_strides[1]]);
        const float tap = sw + float(drow[lag * dyn_strides[2]]);
        const float term = _ink2_dconv_padded<T>(srow, ss, xrow, xs, P, y - lag)
            * tap;
        acc = lag == 0 ? term : acc + term;
    }
    const float res = acc + own;
    const float o = res * scale[0];
    out[((size_t)b * L + t) * C + c] = O(float(T(o)));
"""
_dynamic_conv_kernel = None


def dynamic_conv_fused(
    x: mx.array,
    dyn: mx.array,
    static_w: mx.array,
    state: mx.array,
    heads: int,
    scale_arr: mx.array,
    out_dtype=None,
) -> tuple[mx.array, mx.array]:
    """Fused strided dynamic causal conv for full-attention K/V."""
    global _dynamic_conv_kernel
    if _dynamic_conv_kernel is None:
        _dynamic_conv_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dynamic_conv",
            input_names=["x", "dyn", "static_w", "state", "scale"],
            output_names=["out", "xp"],
            header=_DYNAMIC_CONV_HEADER,
            source=_DYNAMIC_CONV_SOURCE,
            ensure_row_contiguous=False,
        )
    b, length, channels = x.shape
    k = static_w.shape[0]
    odtype = x.dtype if out_dtype is None else out_dtype
    return _dynamic_conv_kernel(
        inputs=[x, dyn, static_w, state, scale_arr],
        template=[
            ("T", x.dtype),
            ("S", state.dtype),
            ("O", odtype),
            ("K", k),
            ("H", heads),
        ],
        grid=(channels, length + k - 1, b),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, length, channels), (b, length + k - 1, channels)],
        output_dtypes=[odtype, mx.float32],
    )


# ---------------------------------------------------------------------------
# Fast exact FP32 router and fused multi-row packed MoE kernels (NVFP4 / Affine-4)
# ---------------------------------------------------------------------------

_FAST_ROUTER_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint e = thread_position_in_grid.y;
    const uint s = thread_position_in_grid.z;
    const device T* x = X + size_t(s) * D;
    const device RT_T* w = RT + size_t(e) * D;
    float acc = 0.0f;
    for (uint i = lane * 4u; i < D; i += 128u) {
        acc += float(x[i]) * float(w[i])
             + float(x[i + 1u]) * float(w[i + 1u])
             + float(x[i + 2u]) * float(w[i + 2u])
             + float(x[i + 3u]) * float(w[i + 3u]);
    }
    acc = simd_sum(acc);
    if (lane == 0u) SC[size_t(s) * E + e] = acc;
"""

_PACKED_MOE_HEADER = r"""
inline float2 _ink2_e2m1x2(uint w, int j) {
    const uint v = w >> (4 * j);
    return float2(as_type<half2>(((v & 0x70007u) << 9) | ((v & 0x80008u) << 12)));
}
inline float _ink2_fp4_dot16(uint2 w, thread const float* x) {
    float a = 0.0f, b = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const float2 p = _ink2_e2m1x2(w.x, j);
        const float2 q = _ink2_e2m1x2(w.y, j);
        a += x[j] * p.x + x[j + 4] * p.y;
        b += x[8 + j] * q.x + x[12 + j] * q.y;
    }
    return a + b;
}
inline float _ink2_fp4_scale(uint b) {
    return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8))));
}
inline float _ink2_nvfp4_chunk(
    const device uint* row,
    const device uchar* srow,
    uint c,
    thread const float* x
) {
    const float d = _ink2_fp4_dot16(((const device uint2*)row)[c], x);
    const float s = _ink2_fp4_scale(uint(srow[c]));
    return d * s * 4194304.0f;
}
template <typename S, int GS>
inline float _ink2_affine4_chunk(
    const device uchar* row,
    const device S* srow,
    const device S* brow,
    uint c,
    thread const float* x
) {
    const device uint* p = (const device uint*)(row + c * 8u);
    uint w0 = p[0], w1 = p[1];
    float qx = 0.0f, sx = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float xi = x[i];
        qx += float((w0 >> (4 * i)) & 0xFu) * xi;
        sx += xi;
    }
    for (int i = 0; i < 8; ++i) {
        float xi = x[8 + i];
        qx += float((w1 >> (4 * i)) & 0xFu) * xi;
        sx += xi;
    }
    const uint g = (c * 16u) / GS;
    return float(srow[g]) * qx + float(brow[g]) * sx;
}
"""

_UPS_NVFP4_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = thread_position_in_grid.y;
    const uint e = thread_position_in_grid.z;
    const device T* x = X + size_t(e / KSEL) * K;
    constexpr uint WPR = K / 8u;
    constexpr uint SPR = K / 16u;
    const size_t base = size_t(IDX[e]) * (2u * I) + 2u * row0;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* srow = WS + base * SPR;
    float accg = 0.0f, accu = 0.0f;
    for (uint k = lane * 16u; k < K; k += 512u) {
        float xr[16];
        for (uint i = 0u; i < 16u; ++i) xr[i] = float(x[k + i]);
        accg += _ink2_nvfp4_chunk(wrow, srow, k >> 4, xr);
        accu += _ink2_nvfp4_chunk(wrow + WPR, srow + SPR, k >> 4, xr);
    }
    const float sc = SC[IDX[e]] * PR[4];
    const float g = float(T(simd_sum(accg))) * sc;
    const float u = float(T(simd_sum(accu))) * sc;
    const float capped =
        (metal::precise::tanh(g * PR[0]) * PR[1])
        * (metal::precise::tanh(u * PR[2]) * PR[3]);
    if (lane == 0u) ACT[size_t(e) * I + row0] = T(capped / (1.0f + metal::precise::exp(-g)));
"""

_DOWNS_NVFP4_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint e = thread_position_in_grid.y;
    const uint srow = thread_position_in_grid.z / D;
    const uint row0 = thread_position_in_grid.z % D;
    const uint p = srow * KSEL + e;
    constexpr uint WPR = I / 8u;
    constexpr uint SPR = I / 16u;
    threadgroup float part[KSEL];
    const size_t base = size_t(IDX[p]) * D + row0;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* scales = WS + base * SPR;
    float acc = 0.0f;
    for (uint k = lane * 16u; k < I; k += 512u) {
        float xr[16];
        for (uint i = 0u; i < 16u; ++i) xr[i] = float(ACT[size_t(p) * I + k + i]);
        acc += _ink2_nvfp4_chunk(wrow, scales, k >> 4, xr);
    }
    const float v = float(T(simd_sum(acc))) * (SC[IDX[p]] * PR[0]) * WTS[p];
    if (lane == 0u) part[e] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u && lane == 0u) {
        float total = 0.0f;
        for (uint k = 0u; k < KSEL; ++k) total += part[k];
        OUT[size_t(srow) * D + row0] = T(total);
    }
"""

_UPS_AFF4_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = thread_position_in_grid.y;
    const uint e = thread_position_in_grid.z;
    const device T* x = X + size_t(e / KSEL) * K;
    constexpr uint WBR = K / 2u;
    constexpr uint SPR = K / GS;
    const size_t base = size_t(IDX[e]) * (2u * I) + 2u * row0;
    const device uchar* arow = (const device uchar*)WQ + base * WBR;
    const device S* sa = WS + base * SPR;
    const device S* ba = WB + base * SPR;
    float accg = 0.0f, accu = 0.0f;
    for (uint k = lane * 16u; k < K; k += 512u) {
        float xr[16];
        for (uint i = 0u; i < 16u; ++i) xr[i] = float(x[k + i]);
        accg += _ink2_affine4_chunk<S, GS>(arow, sa, ba, k >> 4, xr);
        accu += _ink2_affine4_chunk<S, GS>(arow + WBR, sa + SPR, ba + SPR, k >> 4, xr);
    }
    const float g = float(T(simd_sum(accg)));
    const float u = float(T(simd_sum(accu)));
    const float capped =
        (metal::precise::tanh(g * PR[0]) * PR[1])
        * (metal::precise::tanh(u * PR[2]) * PR[3]);
    if (lane == 0u) ACT[size_t(e) * I + row0] = T(capped / (1.0f + metal::precise::exp(-g)));
"""

_DOWNS_AFF4_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint e = thread_position_in_grid.y;
    const uint srow = thread_position_in_grid.z / D;
    const uint row0 = thread_position_in_grid.z % D;
    const uint p = srow * KSEL + e;
    constexpr uint WBR = I / 2u;
    constexpr uint SPR = I / GS;
    threadgroup float part[KSEL];
    const size_t base = size_t(IDX[p]) * D + row0;
    const device uchar* arow = (const device uchar*)WQ + base * WBR;
    const device S* sa = WS + base * SPR;
    const device S* ba = WB + base * SPR;
    float acc = 0.0f;
    for (uint k = lane * 16u; k < I; k += 512u) {
        float xr[16];
        for (uint i = 0u; i < 16u; ++i) xr[i] = float(ACT[size_t(p) * I + k + i]);
        acc += _ink2_affine4_chunk<S, GS>(arow, sa, ba, k >> 4, xr);
    }
    const float v = float(T(simd_sum(acc))) * WTS[p];
    if (lane == 0u) part[e] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0u && lane == 0u) {
        float total = 0.0f;
        for (uint k = 0u; k < KSEL; ++k) total += part[k];
        OUT[size_t(srow) * D + row0] = T(total);
    }
"""

_MOE_PLAN16_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n_slots = T_TOK * KSEL;
    threadgroup uint tg_eids[128];
    threadgroup uint tg_wsum[4];
    if (tid < n_slots) tg_eids[tid] = IDX[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint p[4] = {0u, 0u, 0u, 0u};
    uint my_cnt = 0u, my_nu = 0u;
    if (tid < 128u) {
        for (uint s = 0u; s < n_slots; ++s) {
            if (tg_eids[s] == tid && my_cnt < 16u) {
                p[my_cnt >> 2] |= (s << ((my_cnt & 3u) * 8u));
                my_cnt++;
            }
        }
        my_nu = (my_cnt + 3u) >> 2;
    }
    uint pfx = 0u;
    if (tid < 128u) {
        pfx = simd_prefix_exclusive_sum(my_nu);
        uint ws = simd_sum(my_nu);
        if (lane == 0u) tg_wsum[simd] = ws;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) PLAN[0] = tg_wsum[0] + tg_wsum[1] + tg_wsum[2] + tg_wsum[3];
    if (tid < 128u && my_cnt > 0u) {
        uint dst = pfx;
        for (uint w = 0u; w < simd; ++w) dst += tg_wsum[w];
        for (uint u = 0u; u < my_nu; ++u) {
            uint rem = my_cnt - u * 4u;
            uint c = (rem > 4u) ? 4u : rem;
            PLAN[1u + dst + u] = tid | (c << 8u);
            PLAN[129u + dst + u] = p[u];
        }
    }
"""

_UPS_NVFP4_GRP_SOURCE = r"""
    const uint u = thread_position_in_grid.z;
    if (u >= PLAN[0]) return;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = thread_position_in_grid.y;
    const uint ec = PLAN[1u + u];
    const uint eid = ec & 0xFFu;
    const uint cnt = ec >> 8u;
    const uint packed = PLAN[129u + u];
    constexpr uint WPR = K / 8u;
    constexpr uint SPR = K / 16u;
    const size_t base = size_t(eid) * (2u * I) + 2u * row0;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* srow = WS + base * SPR;
    const float sc = SC[eid] * PR[4];

    if (cnt == 1u) {
        const uint s0 = packed & 0xFFu;
        const device T* x0 = X + size_t(s0 / KSEL) * K;
        float accg = 0.0f, accu = 0.0f;
        for (uint k = lane * 16u; k < K; k += 512u) {
            float xr[16];
            for (uint i = 0u; i < 16u; ++i) xr[i] = float(x0[k + i]);
            accg += _ink2_nvfp4_chunk(wrow, srow, k >> 4, xr);
            accu += _ink2_nvfp4_chunk(wrow + WPR, srow + SPR, k >> 4, xr);
        }
        const float g = float(T(simd_sum(accg))) * sc;
        const float uv = float(T(simd_sum(accu))) * sc;
        const float capped = (metal::precise::tanh(g * PR[0]) * PR[1]) * (metal::precise::tanh(uv * PR[2]) * PR[3]);
        if (lane == 0u) ACT[size_t(s0) * I + row0] = T(capped / (1.0f + metal::precise::exp(-g)));
    } else if (cnt == 2u) {
        const uint s0 = packed & 0xFFu, s1 = (packed >> 8u) & 0xFFu;
        const device T* x0 = X + size_t(s0 / KSEL) * K;
        const device T* x1 = X + size_t(s1 / KSEL) * K;
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
        for (uint k = lane * 16u; k < K; k += 512u) {
            const uint c = k >> 4;
            const uint2 gw = ((const device uint2*)wrow)[c];
            const uint2 uw = ((const device uint2*)(wrow + WPR))[c];
            const float bs_g = _ink2_fp4_scale(uint(srow[c])) * 4194304.0f;
            const float bs_u = _ink2_fp4_scale(uint(srow[SPR + c])) * 4194304.0f;
            float dg0 = 0.0f, du0 = 0.0f, dg1 = 0.0f, du1 = 0.0f;
            for (int j = 0; j < 4; ++j) {
                const float2 pg = _ink2_e2m1x2(gw.x, j), qg = _ink2_e2m1x2(gw.y, j);
                const float2 pu = _ink2_e2m1x2(uw.x, j), qu = _ink2_e2m1x2(uw.y, j);
                const float a0 = float(x0[k + j]), b0 = float(x0[k + j + 4]), c0 = float(x0[k + j + 8]), d0 = float(x0[k + j + 12]);
                const float a1 = float(x1[k + j]), b1 = float(x1[k + j + 4]), c1 = float(x1[k + j + 8]), d1 = float(x1[k + j + 12]);
                dg0 += a0 * pg.x + b0 * pg.y + c0 * qg.x + d0 * qg.y;
                du0 += a0 * pu.x + b0 * pu.y + c0 * qu.x + d0 * qu.y;
                dg1 += a1 * pg.x + b1 * pg.y + c1 * qg.x + d1 * qg.y;
                du1 += a1 * pu.x + b1 * pu.y + c1 * qu.x + d1 * qu.y;
            }
            ag0 += dg0 * bs_g; au0 += du0 * bs_u; ag1 += dg1 * bs_g; au1 += du1 * bs_u;
        }
        { const float g = float(T(simd_sum(ag0))) * sc, uv = float(T(simd_sum(au0))) * sc;
          const float cp = (metal::precise::tanh(g * PR[0]) * PR[1]) * (metal::precise::tanh(uv * PR[2]) * PR[3]);
          if (lane == 0u) ACT[size_t(s0) * I + row0] = T(cp / (1.0f + metal::precise::exp(-g))); }
        { const float g = float(T(simd_sum(ag1))) * sc, uv = float(T(simd_sum(au1))) * sc;
          const float cp = (metal::precise::tanh(g * PR[0]) * PR[1]) * (metal::precise::tanh(uv * PR[2]) * PR[3]);
          if (lane == 0u) ACT[size_t(s1) * I + row0] = T(cp / (1.0f + metal::precise::exp(-g))); }
    } else {
        const uint s0 = packed & 0xFFu, s1 = (packed >> 8u) & 0xFFu, s2 = (packed >> 16u) & 0xFFu, s3 = (packed >> 24u) & 0xFFu;
        const device T* x0 = X + size_t(s0 / KSEL) * K;
        const device T* x1 = X + size_t(s1 / KSEL) * K;
        const device T* x2 = X + size_t(s2 / KSEL) * K;
        const device T* x3 = (cnt >= 4u) ? (X + size_t(s3 / KSEL) * K) : x2;
        float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f, ag2 = 0.0f, au2 = 0.0f, ag3 = 0.0f, au3 = 0.0f;
        for (uint k = lane * 16u; k < K; k += 512u) {
            const uint c = k >> 4;
            const uint2 gw = ((const device uint2*)wrow)[c];
            const uint2 uw = ((const device uint2*)(wrow + WPR))[c];
            const float bs_g = _ink2_fp4_scale(uint(srow[c])) * 4194304.0f;
            const float bs_u = _ink2_fp4_scale(uint(srow[SPR + c])) * 4194304.0f;
            float dg0=0.0f,du0=0.0f,dg1=0.0f,du1=0.0f,dg2=0.0f,du2=0.0f,dg3=0.0f,du3=0.0f;
            for (int j = 0; j < 4; ++j) {
                const float2 pg = _ink2_e2m1x2(gw.x, j), qg = _ink2_e2m1x2(gw.y, j);
                const float2 pu = _ink2_e2m1x2(uw.x, j), qu = _ink2_e2m1x2(uw.y, j);
                const float a0=float(x0[k+j]),b0=float(x0[k+j+4]),c0=float(x0[k+j+8]),d0=float(x0[k+j+12]);
                const float a1=float(x1[k+j]),b1=float(x1[k+j+4]),c1=float(x1[k+j+8]),d1=float(x1[k+j+12]);
                const float a2=float(x2[k+j]),b2=float(x2[k+j+4]),c2=float(x2[k+j+8]),d2=float(x2[k+j+12]);
                const float a3=float(x3[k+j]),b3=float(x3[k+j+4]),c3=float(x3[k+j+8]),d3=float(x3[k+j+12]);
                dg0+=a0*pg.x+b0*pg.y+c0*qg.x+d0*qg.y; du0+=a0*pu.x+b0*pu.y+c0*qu.x+d0*qu.y;
                dg1+=a1*pg.x+b1*pg.y+c1*qg.x+d1*qg.y; du1+=a1*pu.x+b1*pu.y+c1*qu.x+d1*qu.y;
                dg2+=a2*pg.x+b2*pg.y+c2*qg.x+d2*qg.y; du2+=a2*pu.x+b2*pu.y+c2*qu.x+d2*qu.y;
                dg3+=a3*pg.x+b3*pg.y+c3*qg.x+d3*qg.y; du3+=a3*pu.x+b3*pu.y+c3*qu.x+d3*qu.y;
            }
            ag0+=dg0*bs_g; au0+=du0*bs_u; ag1+=dg1*bs_g; au1+=du1*bs_u; ag2+=dg2*bs_g; au2+=du2*bs_u; ag3+=dg3*bs_g; au3+=du3*bs_u;
        }
        { const float g=float(T(simd_sum(ag0)))*sc, uv=float(T(simd_sum(au0)))*sc; const float cp=(metal::precise::tanh(g*PR[0])*PR[1])*(metal::precise::tanh(uv*PR[2])*PR[3]); if (lane==0u) ACT[size_t(s0)*I+row0]=T(cp/(1.0f+metal::precise::exp(-g))); }
        { const float g=float(T(simd_sum(ag1)))*sc, uv=float(T(simd_sum(au1)))*sc; const float cp=(metal::precise::tanh(g*PR[0])*PR[1])*(metal::precise::tanh(uv*PR[2])*PR[3]); if (lane==0u) ACT[size_t(s1)*I+row0]=T(cp/(1.0f+metal::precise::exp(-g))); }
        { const float g=float(T(simd_sum(ag2)))*sc, uv=float(T(simd_sum(au2)))*sc; const float cp=(metal::precise::tanh(g*PR[0])*PR[1])*(metal::precise::tanh(uv*PR[2])*PR[3]); if (lane==0u) ACT[size_t(s2)*I+row0]=T(cp/(1.0f+metal::precise::exp(-g))); }
        if (cnt >= 4u) { const float g=float(T(simd_sum(ag3)))*sc, uv=float(T(simd_sum(au3)))*sc; const float cp=(metal::precise::tanh(g*PR[0])*PR[1])*(metal::precise::tanh(uv*PR[2])*PR[3]); if (lane==0u) ACT[size_t(s3)*I+row0]=T(cp/(1.0f+metal::precise::exp(-g))); }
    }
"""

_DOWNS_NVFP4_GRP_SOURCE = r"""
    const uint u = thread_position_in_grid.z;
    if (u >= PLAN[0]) return;
    const uint lane = thread_index_in_simdgroup;
    const uint row0 = thread_position_in_grid.y;
    const uint ec = PLAN[1u + u];
    const uint eid = ec & 0xFFu;
    const uint cnt = ec >> 8u;
    const uint packed = PLAN[129u + u];
    constexpr uint WPR = I / 8u;
    constexpr uint SPR = I / 16u;
    const size_t base = size_t(eid) * D + row0;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* srow = WS + base * SPR;
    const float sc = SC[eid] * PR[0];

    if (cnt == 1u) {
        const uint s0 = packed & 0xFFu;
        const device T* a0 = ACT + size_t(s0) * I;
        float acc = 0.0f;
        for (uint k = lane * 16u; k < I; k += 512u) {
            float xr[16];
            for (uint i = 0u; i < 16u; ++i) xr[i] = float(a0[k + i]);
            acc += _ink2_nvfp4_chunk(wrow, srow, k >> 4, xr);
        }
        const float v = float(T(simd_sum(acc))) * sc * WTS[s0];
        if (lane == 0u) YSLOT[size_t(s0) * D + row0] = v;
    } else if (cnt == 2u) {
        const uint s0 = packed & 0xFFu, s1 = (packed >> 8u) & 0xFFu;
        const device T* a0 = ACT + size_t(s0) * I;
        const device T* a1 = ACT + size_t(s1) * I;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint k = lane * 16u; k < I; k += 512u) {
            const uint c = k >> 4;
            const uint2 w = ((const device uint2*)wrow)[c];
            const float bs = _ink2_fp4_scale(uint(srow[c])) * 4194304.0f;
            float d0 = 0.0f, d1 = 0.0f;
            for (int j = 0; j < 4; ++j) {
                const float2 p = _ink2_e2m1x2(w.x, j), q = _ink2_e2m1x2(w.y, j);
                d0 += float(a0[k+j])*p.x + float(a0[k+j+4])*p.y + float(a0[k+j+8])*q.x + float(a0[k+j+12])*q.y;
                d1 += float(a1[k+j])*p.x + float(a1[k+j+4])*p.y + float(a1[k+j+8])*q.x + float(a1[k+j+12])*q.y;
            }
            acc0 += d0 * bs; acc1 += d1 * bs;
        }
        const float v0 = float(T(simd_sum(acc0))) * sc * WTS[s0];
        const float v1 = float(T(simd_sum(acc1))) * sc * WTS[s1];
        if (lane == 0u) { YSLOT[size_t(s0) * D + row0] = v0; YSLOT[size_t(s1) * D + row0] = v1; }
    } else {
        const uint s0 = packed & 0xFFu, s1 = (packed >> 8u) & 0xFFu, s2 = (packed >> 16u) & 0xFFu, s3 = (packed >> 24u) & 0xFFu;
        const device T* a0 = ACT + size_t(s0) * I;
        const device T* a1 = ACT + size_t(s1) * I;
        const device T* a2 = ACT + size_t(s2) * I;
        const device T* a3 = (cnt >= 4u) ? (ACT + size_t(s3) * I) : a2;
        float acc0=0.0f, acc1=0.0f, acc2=0.0f, acc3=0.0f;
        for (uint k = lane * 16u; k < I; k += 512u) {
            const uint c = k >> 4;
            const uint2 w = ((const device uint2*)wrow)[c];
            const float bs = _ink2_fp4_scale(uint(srow[c])) * 4194304.0f;
            float d0=0.0f,d1=0.0f,d2=0.0f,d3=0.0f;
            for (int j = 0; j < 4; ++j) {
                const float2 p = _ink2_e2m1x2(w.x, j), q = _ink2_e2m1x2(w.y, j);
                d0 += float(a0[k+j])*p.x + float(a0[k+j+4])*p.y + float(a0[k+j+8])*q.x + float(a0[k+j+12])*q.y;
                d1 += float(a1[k+j])*p.x + float(a1[k+j+4])*p.y + float(a1[k+j+8])*q.x + float(a1[k+j+12])*q.y;
                d2 += float(a2[k+j])*p.x + float(a2[k+j+4])*p.y + float(a2[k+j+8])*q.x + float(a2[k+j+12])*q.y;
                d3 += float(a3[k+j])*p.x + float(a3[k+j+4])*p.y + float(a3[k+j+8])*q.x + float(a3[k+j+12])*q.y;
            }
            acc0+=d0*bs; acc1+=d1*bs; acc2+=d2*bs; acc3+=d3*bs;
        }
        const float v0=float(T(simd_sum(acc0)))*sc*WTS[s0], v1=float(T(simd_sum(acc1)))*sc*WTS[s1], v2=float(T(simd_sum(acc2)))*sc*WTS[s2], v3=float(T(simd_sum(acc3)))*sc*WTS[s3];
        if (lane == 0u) { YSLOT[size_t(s0)*D+row0]=v0; YSLOT[size_t(s1)*D+row0]=v1; YSLOT[size_t(s2)*D+row0]=v2; if (cnt>=4u) YSLOT[size_t(s3)*D+row0]=v3; }
    }
"""

_MOE_COMB_SOURCE = r"""
    const uint col = thread_position_in_grid.x;
    const uint tok = thread_position_in_grid.y;
    if (col >= D) return;
    const size_t base = size_t(tok) * KSEL * D + col;
    float s = 0.0f;
    for (uint k = 0u; k < KSEL; ++k) s += YSLOT[base + size_t(k) * D];
    OUT[size_t(tok) * D + col] = T(s);
"""

_fast_router_kernel = None
_ups_nvfp4_kernel = None
_downs_nvfp4_kernel = None
_ups_aff4_kernel = None
_downs_aff4_kernel = None
_moe_plan16_kernel = None
_ups_nvfp4_grp_kernel = None
_downs_nvfp4_grp_kernel = None
_moe_comb_kernel = None


def fast_router_logits(x: mx.array, router_weight: mx.array) -> mx.array:
    """Exact FP32 router dot product for small batches (supports FP32 or bit-exact BF16 weights)."""
    global _fast_router_kernel
    if _fast_router_kernel is None:
        _fast_router_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_fast_router",
            input_names=["X", "RT"],
            output_names=["SC"],
            source=_FAST_ROUTER_SOURCE,
        )
    rows, hidden = x.shape
    experts = router_weight.shape[0]
    return _fast_router_kernel(
        inputs=[x, router_weight],
        template=[
            ("T", x.dtype),
            ("RT_T", router_weight.dtype),
            ("D", hidden),
            ("E", experts),
        ],
        grid=(32, experts, rows),
        threadgroup=(32, 4, 1),
        output_shapes=[(rows, experts)],
        output_dtypes=[mx.float32],
    )[0]


def moe_fused_packed(
    x: mx.array,
    w13,
    w2,
    inds: mx.array,
    weights: mx.array,
    up_params: mx.array,
    dn_params: mx.array,
) -> mx.array:
    """Fused multi-row (1..16) packed NVFP4 (grouped for 3..16) or Affine-4 MoE."""
    global _ups_nvfp4_kernel, _downs_nvfp4_kernel, _ups_aff4_kernel, _downs_aff4_kernel
    global _moe_plan16_kernel, _ups_nvfp4_grp_kernel, _downs_nvfp4_grp_kernel, _moe_comb_kernel
    rows, hidden = x.shape
    top_k = inds.shape[1]
    inter = w13.weight.shape[1] // 2
    idx_f = inds.reshape(-1).astype(mx.uint32)
    wts_f = weights.reshape(-1).astype(mx.float32)
    w13_w, w13_s, w13_b = _moe_packed_views(w13)
    w2_w, w2_s, w2_b = _moe_packed_views(w2)
    if w13.mode == "nvfp4":
        if rows >= 2 and w13.weight.shape[0] <= 128:
            if _moe_plan16_kernel is None:
                _moe_plan16_kernel = mx.fast.metal_kernel(
                    name="omlx_inkling2_moe_plan16",
                    input_names=["IDX"],
                    output_names=["PLAN"],
                    source=_MOE_PLAN16_SOURCE,
                )
                _ups_nvfp4_grp_kernel = mx.fast.metal_kernel(
                    name="omlx_inkling2_ups_nvfp4_grp",
                    input_names=["X", "WQ", "WS", "SC", "PR", "PLAN"],
                    output_names=["ACT"],
                    source=_UPS_NVFP4_GRP_SOURCE,
                    header=_PACKED_MOE_HEADER,
                )
                _downs_nvfp4_grp_kernel = mx.fast.metal_kernel(
                    name="omlx_inkling2_downs_nvfp4_grp",
                    input_names=["ACT", "WQ", "WS", "SC", "WTS", "PR", "PLAN"],
                    output_names=["YSLOT"],
                    source=_DOWNS_NVFP4_GRP_SOURCE,
                    header=_PACKED_MOE_HEADER,
                )
                _moe_comb_kernel = mx.fast.metal_kernel(
                    name="omlx_inkling2_moe_comb",
                    input_names=["YSLOT"],
                    output_names=["OUT"],
                    source=_MOE_COMB_SOURCE,
                )
            plan = _moe_plan16_kernel(
                inputs=[idx_f],
                template=[("T_TOK", rows), ("KSEL", top_k)],
                grid=(256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(257,)],
                output_dtypes=[mx.uint32],
            )[0]
            act = _ups_nvfp4_grp_kernel(
                inputs=[x, w13_w, w13_s, w13.global_scale, up_params, plan],
                template=[("T", x.dtype), ("K", hidden), ("I", inter), ("KSEL", top_k)],
                grid=(32, inter, rows * top_k),
                threadgroup=(32, 2, 1),
                output_shapes=[(rows * top_k, inter)],
                output_dtypes=[x.dtype],
            )[0]
            yslot = _downs_nvfp4_grp_kernel(
                inputs=[act, w2_w, w2_s, w2.global_scale, wts_f, dn_params, plan],
                template=[("T", x.dtype), ("I", inter), ("D", hidden)],
                grid=(32, hidden, rows * top_k),
                threadgroup=(32, 2, 1),
                output_shapes=[(rows * top_k, hidden)],
                output_dtypes=[mx.float32],
            )[0]
            return _moe_comb_kernel(
                inputs=[yslot],
                template=[("T", x.dtype), ("D", hidden), ("KSEL", top_k)],
                grid=(hidden, rows, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(rows, hidden)],
                output_dtypes=[x.dtype],
            )[0]
        if _ups_nvfp4_kernel is None:
            _ups_nvfp4_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_ups_nvfp4",
                input_names=["X", "WQ", "WS", "IDX", "SC", "PR"],
                output_names=["ACT"],
                source=_UPS_NVFP4_SOURCE,
                header=_PACKED_MOE_HEADER,
            )
            _downs_nvfp4_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_downs_nvfp4",
                input_names=["ACT", "WQ", "WS", "IDX", "SC", "WTS", "PR"],
                output_names=["OUT"],
                source=_DOWNS_NVFP4_SOURCE,
                header=_PACKED_MOE_HEADER,
            )
        act = _ups_nvfp4_kernel(
            inputs=[x, w13_w, w13_s, idx_f, w13.global_scale, up_params],
            template=[("T", x.dtype), ("K", hidden), ("I", inter), ("KSEL", top_k)],
            grid=(32, inter, rows * top_k),
            threadgroup=(32, 2, 1),
            output_shapes=[(rows * top_k, inter)],
            output_dtypes=[x.dtype],
        )[0]
        return _downs_nvfp4_kernel(
            inputs=[act, w2_w, w2_s, idx_f, w2.global_scale, wts_f, dn_params],
            template=[("T", x.dtype), ("D", hidden), ("I", inter), ("KSEL", top_k)],
            grid=(32, top_k, rows * hidden),
            threadgroup=(32, top_k, 1),
            output_shapes=[(rows, hidden)],
            output_dtypes=[x.dtype],
        )[0]
    if _ups_aff4_kernel is None:
        _ups_aff4_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_ups_aff4",
            input_names=["X", "WQ", "WS", "WB", "IDX", "PR"],
            output_names=["ACT"],
            source=_UPS_AFF4_SOURCE,
            header=_PACKED_MOE_HEADER,
        )
        _downs_aff4_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_downs_aff4",
            input_names=["ACT", "WQ", "WS", "WB", "IDX", "WTS"],
            output_names=["OUT"],
            source=_DOWNS_AFF4_SOURCE,
            header=_PACKED_MOE_HEADER,
        )
    act = _ups_aff4_kernel(
        inputs=[x, w13_w, w13_s, w13_b, idx_f, up_params],
        template=[
            ("T", x.dtype),
            ("S", w13.scales.dtype),
            ("GS", w13.group_size),
            ("K", hidden),
            ("I", inter),
            ("KSEL", top_k),
        ],
        grid=(32, inter, rows * top_k),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows * top_k, inter)],
        output_dtypes=[x.dtype],
    )[0]
    return _downs_aff4_kernel(
        inputs=[act, w2_w, w2_s, w2_b, idx_f, wts_f],
        template=[
            ("T", x.dtype),
            ("S", w2.scales.dtype),
            ("GS", w2.group_size),
            ("D", hidden),
            ("I", inter),
            ("KSEL", top_k),
        ],
        grid=(32, top_k, rows * hidden),
        threadgroup=(32, top_k, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[x.dtype],
    )[0]


# ---------------------------------------------------------------------------
# M5 Neural Accelerator (Metal 4 matmul2d) batched MoE prefill (NVFP4 & Aff4)
# ---------------------------------------------------------------------------

_NAX_MOE_HEADER = r"""
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant float _INK2_E2M1_F[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float _ink2_e4m3_scale_f(uchar b) {
    return float(as_type<half>(ushort(((uint(b) & 0x7Fu) << 7) | ((uint(b) & 0x80u) << 8)))) * 256.0f;
}

inline void _ink2_stage_nvfp4_16_bf(
    const device uint* wrow,
    const device uchar* srow,
    uint c,
    thread bfloat4* out
) {
    const uint2 pk = ((const device uint2*)wrow)[c];
    const float sc = _ink2_e4m3_scale_f(srow[c]);
    const uint w0 = pk.x, w1 = pk.y;
    out[0] = bfloat4(float4(_INK2_E2M1_F[w0 & 15u], _INK2_E2M1_F[(w0 >> 4) & 15u],
                            _INK2_E2M1_F[(w0 >> 8) & 15u], _INK2_E2M1_F[(w0 >> 12) & 15u]) * sc);
    out[1] = bfloat4(float4(_INK2_E2M1_F[(w0 >> 16) & 15u], _INK2_E2M1_F[(w0 >> 20) & 15u],
                            _INK2_E2M1_F[(w0 >> 24) & 15u], _INK2_E2M1_F[(w0 >> 28) & 15u]) * sc);
    out[2] = bfloat4(float4(_INK2_E2M1_F[w1 & 15u], _INK2_E2M1_F[(w1 >> 4) & 15u],
                            _INK2_E2M1_F[(w1 >> 8) & 15u], _INK2_E2M1_F[(w1 >> 12) & 15u]) * sc);
    out[3] = bfloat4(float4(_INK2_E2M1_F[(w1 >> 16) & 15u], _INK2_E2M1_F[(w1 >> 20) & 15u],
                            _INK2_E2M1_F[(w1 >> 24) & 15u], _INK2_E2M1_F[(w1 >> 28) & 15u]) * sc);
}

template <typename S, int GS>
inline void _ink2_stage_aff4_16_bf(
    const device uint* wrow,
    const device S* srow,
    const device S* brow,
    uint c,
    thread bfloat4* out
) {
    const uint2 pk = ((const device uint2*)wrow)[c];
    const uint g = (c * 16u) / GS;
    const float sc = float(srow[g]);
    const float bi = float(brow[g]);
    const uint w0 = pk.x, w1 = pk.y;
    out[0] = bfloat4(fma(float4(w0 & 15u, (w0 >> 4) & 15u, (w0 >> 8) & 15u, (w0 >> 12) & 15u), sc, bi));
    out[1] = bfloat4(fma(float4((w0 >> 16) & 15u, (w0 >> 20) & 15u, (w0 >> 24) & 15u, (w0 >> 28) & 15u), sc, bi));
    out[2] = bfloat4(fma(float4(w1 & 15u, (w1 >> 4) & 15u, (w1 >> 8) & 15u, (w1 >> 12) & 15u), sc, bi));
    out[3] = bfloat4(fma(float4((w1 >> 16) & 15u, (w1 >> 20) & 15u, (w1 >> 24) & 15u, (w1 >> 28) & 15u), sc, bi));
}
"""

_NAX_MOE_LAYOUT_PROBE_SOURCE = r"""
    threadgroup bfloat sa[64 * 32];
    auto tA = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(32, 64));
    auto tX = tensor((device bfloat*)X, dextents<int32_t, 2>(32, 16), array<int, 2>({1, 32}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 64, 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tA), float>();
    uint tid = thread_index_in_threadgroup;
    for (short i = 0; i < cT.get_capacity() && i < 63; ++i) {
        if (cT.is_valid_element(i)) {
            auto ix = cT.get_multidimensional_index(i);
            OUT[tid * 64 + i] = int((ix[0] << 16) | ix[1]);
        } else {
            OUT[tid * 64 + i] = -1;
        }
    }
    OUT[tid * 64 + 63] = int(cT.get_capacity());
"""

_NAX_MOE_ROUTE_SOURCE = r"""
    threadgroup atomic_uint cnt[NEXP];
    threadgroup uint off[NEXP], toff[NEXP];
    threadgroup atomic_uint fill[NEXP];
    const uint tid = thread_index_in_threadgroup;
    if (tid < NEXP) {
        atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&fill[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = T_TOK * TOPK;
    for (uint i = tid; i < n; i += 256u) {
        atomic_fetch_add_explicit(&cnt[INDS[i]], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint s = 0u, ts = 0u;
        for (uint e = 0u; e < NEXP; ++e) {
            uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            off[e] = s;
            toff[e] = ts;
            s += c;
            ts += (c + TILE_SZ - 1u) / TILE_SZ;
        }
        NTILES[0] = ts;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += 256u) {
        uint e = INDS[i];
        uint slot = off[e] + atomic_fetch_add_explicit(&fill[e], 1u, memory_order_relaxed);
        PERM[slot] = i / TOPK;
        INV[i] = slot;
        AW[slot] = WTS[i];
    }
    if (tid < NEXP) {
        uint c = atomic_load_explicit(&cnt[tid], memory_order_relaxed);
        uint nt = (c + TILE_SZ - 1u) / TILE_SZ;
        for (uint j = 0u; j < nt; ++j) {
            uint base = (toff[tid] + j) * 4u;
            TILES[base + 0u] = tid;
            TILES[base + 1u] = off[tid] + j * TILE_SZ;
            TILES[base + 2u] = min(uint(TILE_SZ), c - j * TILE_SZ);
            TILES[base + 3u] = 0u;
        }
    }
"""

_NAX_MOE_GATHER_SOURCE = r"""
    const uint slot = threadgroup_position_in_grid.x;
    const uint tok = PERM[slot];
    const uint tid = thread_index_in_threadgroup;
    const device bfloat4* src = (const device bfloat4*)(X + size_t(tok) * D);
    device bfloat4* dst = (device bfloat4*)(XG + size_t(slot) * D);
    for (uint i = tid; i < (D >> 2); i += 128u) {
        dst[i] = src[i];
    }
"""

_NAX_MOE_UP_NVFP4_SOURCE = r"""
    const uint tile_idx = threadgroup_position_in_grid.x;
    const uint cb = threadgroup_position_in_grid.y;
    if (tile_idx >= NTILES[0]) return;
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint e = TILES[tile_idx * 4u + 0u];
    const uint as0 = TILES[tile_idx * 4u + 1u];
    const uint cnt = TILES[tile_idx * 4u + 2u];
    const uint ts0 = sg * 16u;
    const bool on = ts0 < cnt;
    const int nt = on ? min(16, int(cnt - ts0)) : 0;

    threadgroup bfloat sa[2][64 * 32];
    auto tA0 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(32, 64));
    auto tA1 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(32, 64));
    auto tX = tensor((device bfloat*)(XG + size_t(as0 + (on ? ts0 : 0u)) * K_DIM), dextents<int32_t, 2>(K_DIM, nt), array<int, 2>({1, K_DIM}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 64, 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();

    const bool wact = tid < 128u;
    const uint rr = tid >> 1;
    const uint hp = tid & 1u;
    const uint wrow = (rr < 32u) ? (2u * (cb * 32u + rr)) : (2u * (cb * 32u + (rr - 32u)) + 1u);
    const device uint* wp = W + (size_t(e) * (2u * FF_DIM) + wrow) * (K_DIM >> 3);
    const device uchar* sp = S + (size_t(e) * (2u * FF_DIM) + wrow) * (K_DIM >> 4);

    bfloat4 wv[4];
    if (wact) {
        _ink2_stage_nvfp4_16_bf(wp, sp, hp, wv);
        threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[0] + rr * 32u + hp * 16u);
        dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint k0 = 0u; k0 < K_DIM; k0 += 32u) {
        const bool more = (k0 + 32u) < K_DIM;
        if (more && wact) _ink2_stage_nvfp4_16_bf(wp, sp, ((k0 + 32u) >> 4) + hp, wv);
        if (on) {
            auto sx = tX.slice(int(k0), 0);
            if (buf == 0u) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        }
        if (more && wact) {
            threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[buf ^ 1u] + rr * 32u + hp * 16u);
            dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1u;
    }
    if (!on) return;
    const float gsc = GS_ARR[e] * PR[4];
    const float inv_gcap = PR[0], gcap = PR[1], inv_ucap = PR[2], ucap = PR[3];
    for (short q = 0; q < 4; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = ts0 + ix[1];
        if (t < cnt) {
            const float4 gt = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]) * gsc;
            const float4 up = float4(cT[4 * q + 16], cT[4 * q + 17], cT[4 * q + 18], cT[4 * q + 19]) * gsc;
            const float4 sg_v = gcap * tanh(gt * inv_gcap) / (1.0f + exp(-gt));
            const float4 su_v = ucap * tanh(up * inv_ucap);
            *(device bfloat4*)(ACT + size_t(as0 + t) * FF_DIM + cb * 32u + r) = bfloat4(sg_v * su_v);
        }
    }
"""

_NAX_MOE_DN_NVFP4_SOURCE = r"""
    const uint tile_idx = threadgroup_position_in_grid.x;
    const uint cb = threadgroup_position_in_grid.y;
    if (tile_idx >= NTILES[0]) return;
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint e = TILES[tile_idx * 4u + 0u];
    const uint as0 = TILES[tile_idx * 4u + 1u];
    const uint cnt = TILES[tile_idx * 4u + 2u];
    const uint ts0 = sg * 16u;
    const bool on = ts0 < cnt;
    const int nt = on ? min(16, int(cnt - ts0)) : 0;

    threadgroup bfloat sa[2][64 * 32];
    auto tA0 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(32, 64));
    auto tA1 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(32, 64));
    auto tX = tensor((device bfloat*)(ACT + size_t(as0 + (on ? ts0 : 0u)) * K_DIM), dextents<int32_t, 2>(K_DIM, nt), array<int, 2>({1, K_DIM}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 64, 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();

    const bool wact = tid < 128u;
    const uint rr = tid >> 1;
    const uint hp = tid & 1u;
    const uint wrow = cb * 64u + rr;
    const device uint* wp = W + (size_t(e) * OUT_DIM + wrow) * (K_DIM >> 3);
    const device uchar* sp = S + (size_t(e) * OUT_DIM + wrow) * (K_DIM >> 4);

    bfloat4 wv[4];
    if (wact) {
        _ink2_stage_nvfp4_16_bf(wp, sp, hp, wv);
        threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[0] + rr * 32u + hp * 16u);
        dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint k0 = 0u; k0 < K_DIM; k0 += 32u) {
        const bool more = (k0 + 32u) < K_DIM;
        if (more && wact) _ink2_stage_nvfp4_16_bf(wp, sp, ((k0 + 32u) >> 4) + hp, wv);
        if (on) {
            auto sx = tX.slice(int(k0), 0);
            if (buf == 0u) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        }
        if (more && wact) {
            threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[buf ^ 1u] + rr * 32u + hp * 16u);
            dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1u;
    }
    if (!on) return;
    const float gsc = GS_ARR[e] * INV_SC[0];
    for (short q = 0; q < 8; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = ts0 + ix[1];
        if (t < cnt) {
            const float4 v = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]) * gsc;
            *(device bfloat4*)(YD + size_t(as0 + t) * OUT_DIM + cb * 64u + r) = bfloat4(v);
        }
    }
"""

_NAX_MOE_UP_AFF4_SOURCE = r"""
    const uint tile_idx = threadgroup_position_in_grid.x;
    const uint cb = threadgroup_position_in_grid.y;
    if (tile_idx >= NTILES[0]) return;
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint e = TILES[tile_idx * 4u + 0u];
    const uint as0 = TILES[tile_idx * 4u + 1u];
    const uint cnt = TILES[tile_idx * 4u + 2u];
    const uint ts0 = sg * 16u;
    const bool on = ts0 < cnt;
    const int nt = on ? min(16, int(cnt - ts0)) : 0;

    threadgroup bfloat sa[2][64 * 32];
    auto tA0 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(32, 64));
    auto tA1 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(32, 64));
    auto tX = tensor((device bfloat*)(XG + size_t(as0 + (on ? ts0 : 0u)) * K_DIM), dextents<int32_t, 2>(K_DIM, nt), array<int, 2>({1, K_DIM}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 64, 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();

    const bool wact = tid < 128u;
    const uint rr = tid >> 1;
    const uint hp = tid & 1u;
    const uint wrow = (rr < 32u) ? (2u * (cb * 32u + rr)) : (2u * (cb * 32u + (rr - 32u)) + 1u);
    const device uint* wp = W + (size_t(e) * (2u * FF_DIM) + wrow) * (K_DIM >> 3);
    const device S_T* sp = S + (size_t(e) * (2u * FF_DIM) + wrow) * (K_DIM / GS);
    const device S_T* bp = B + (size_t(e) * (2u * FF_DIM) + wrow) * (K_DIM / GS);

    bfloat4 wv[4];
    if (wact) {
        _ink2_stage_aff4_16_bf<S_T, GS>(wp, sp, bp, hp, wv);
        threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[0] + rr * 32u + hp * 16u);
        dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint k0 = 0u; k0 < K_DIM; k0 += 32u) {
        const bool more = (k0 + 32u) < K_DIM;
        if (more && wact) _ink2_stage_aff4_16_bf<S_T, GS>(wp, sp, bp, ((k0 + 32u) >> 4) + hp, wv);
        if (on) {
            auto sx = tX.slice(int(k0), 0);
            if (buf == 0u) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        }
        if (more && wact) {
            threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[buf ^ 1u] + rr * 32u + hp * 16u);
            dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1u;
    }
    if (!on) return;
    const float inv_gcap = PR[0], gcap = PR[1], inv_ucap = PR[2], ucap = PR[3];
    for (short q = 0; q < 4; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = ts0 + ix[1];
        if (t < cnt) {
            const float4 gt = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            const float4 up = float4(cT[4 * q + 16], cT[4 * q + 17], cT[4 * q + 18], cT[4 * q + 19]);
            const float4 sg_v = gcap * tanh(gt * inv_gcap) / (1.0f + exp(-gt));
            const float4 su_v = ucap * tanh(up * inv_ucap);
            *(device bfloat4*)(ACT + size_t(as0 + t) * FF_DIM + cb * 32u + r) = bfloat4(sg_v * su_v);
        }
    }
"""

_NAX_MOE_DN_AFF4_SOURCE = r"""
    const uint tile_idx = threadgroup_position_in_grid.x;
    const uint cb = threadgroup_position_in_grid.y;
    if (tile_idx >= NTILES[0]) return;
    const uint tid = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint e = TILES[tile_idx * 4u + 0u];
    const uint as0 = TILES[tile_idx * 4u + 1u];
    const uint cnt = TILES[tile_idx * 4u + 2u];
    const uint ts0 = sg * 16u;
    const bool on = ts0 < cnt;
    const int nt = on ? min(16, int(cnt - ts0)) : 0;

    threadgroup bfloat sa[2][64 * 32];
    auto tA0 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(32, 64));
    auto tA1 = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(32, 64));
    auto tX = tensor((device bfloat*)(ACT + size_t(as0 + (on ? ts0 : 0u)) * K_DIM), dextents<int32_t, 2>(K_DIM, nt), array<int, 2>({1, K_DIM}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 64, 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();

    const bool wact = tid < 128u;
    const uint rr = tid >> 1;
    const uint hp = tid & 1u;
    const uint wrow = cb * 64u + rr;
    const device uint* wp = W + (size_t(e) * OUT_DIM + wrow) * (K_DIM >> 3);
    const device S_T* sp = S + (size_t(e) * OUT_DIM + wrow) * (K_DIM / GS);
    const device S_T* bp = B + (size_t(e) * OUT_DIM + wrow) * (K_DIM / GS);

    bfloat4 wv[4];
    if (wact) {
        _ink2_stage_aff4_16_bf<S_T, GS>(wp, sp, bp, hp, wv);
        threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[0] + rr * 32u + hp * 16u);
        dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint k0 = 0u; k0 < K_DIM; k0 += 32u) {
        const bool more = (k0 + 32u) < K_DIM;
        if (more && wact) _ink2_stage_aff4_16_bf<S_T, GS>(wp, sp, bp, ((k0 + 32u) >> 4) + hp, wv);
        if (on) {
            auto sx = tX.slice(int(k0), 0);
            if (buf == 0u) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        }
        if (more && wact) {
            threadgroup bfloat4* dst = (threadgroup bfloat4*)(sa[buf ^ 1u] + rr * 32u + hp * 16u);
            dst[0] = wv[0]; dst[1] = wv[1]; dst[2] = wv[2]; dst[3] = wv[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1u;
    }
    if (!on) return;
    for (short q = 0; q < 8; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = ts0 + ix[1];
        if (t < cnt) {
            const float4 v = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            *(device bfloat4*)(YD + size_t(as0 + t) * OUT_DIM + cb * 64u + r) = bfloat4(v);
        }
    }
"""

_NAX_MOE_COMB_SOURCE = r"""
    const uint tok = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    uint slots[TOPK];
    float ws[TOPK];
    for (uint k = 0u; k < TOPK; ++k) {
        uint s = INV[tok * TOPK + k];
        slots[k] = s;
        ws[k] = AW[s];
    }
    device bfloat4* dst = (device bfloat4*)(OUT + size_t(tok) * D);
    for (uint i = tid; i < (D >> 2); i += 128u) {
        float4 acc = float4(0.0f);
        for (uint k = 0u; k < TOPK; ++k) {
            acc += ws[k] * float4(((const device bfloat4*)(YD + size_t(slots[k]) * D))[i]);
        }
        dst[i] = bfloat4(acc);
    }
"""

_nax_moe_supported: bool | None = None
_nax_moe_route_kernel = None
_nax_moe_gather_kernel = None
_nax_moe_up_nvfp4_kernel = None
_nax_moe_dn_nvfp4_kernel = None
_nax_moe_up_aff4_kernel = None
_nax_moe_dn_aff4_kernel = None
_nax_moe_comb_kernel = None


def can_use_nax_moe() -> bool:
    """Return True if Metal 4 matmul2d is available and matches the expected quad layout."""
    global _nax_moe_supported
    if _nax_moe_supported is not None:
        return _nax_moe_supported
    if not is_nax_available() or mx.default_device() != mx.gpu:
        _nax_moe_supported = False
        return False
    try:
        probe = mx.fast.metal_kernel(
            name="omlx_inkling2_nax_moe_probe",
            input_names=["X"],
            output_names=["OUT"],
            header=_NAX_MOE_HEADER,
            source=_NAX_MOE_LAYOUT_PROBE_SOURCE,
        )
        out = probe(
            inputs=[mx.zeros((16, 32), dtype=mx.bfloat16)],
            output_shapes=[(32 * 64,)],
            output_dtypes=[mx.int32],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
        )[0]
        mx.eval(out)
        arr = out.tolist()
        ok = arr[63] == 32
        if ok:
            for lane in range(32):
                base = lane * 64
                for q in range(4):
                    r0_base = arr[base + 4 * q] >> 16
                    t0_base = arr[base + 4 * q] & 0xFFFF
                    for j in range(4):
                        v0 = arr[base + 4 * q + j]
                        v1 = arr[base + 4 * q + 16 + j]
                        if (
                            (v0 >> 16) != r0_base + j
                            or (v0 & 0xFFFF) != t0_base
                            or (v1 >> 16) != r0_base + 32 + j
                            or (v1 & 0xFFFF) != t0_base
                        ):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
        _nax_moe_supported = ok
    except Exception:
        _nax_moe_supported = False
    return _nax_moe_supported


def moe_prefill_nax(
    x: mx.array,
    w13,
    w2,
    inds: mx.array,
    weights: mx.array,
    up_params: mx.array,
    dn_params: mx.array,
) -> mx.array:
    """M5 Metal 4 matmul2d batched MoE prefill for NVFP4 and Affine-4 experts."""
    global _nax_moe_route_kernel, _nax_moe_gather_kernel, _nax_moe_comb_kernel
    global _nax_moe_up_nvfp4_kernel, _nax_moe_dn_nvfp4_kernel
    global _nax_moe_up_aff4_kernel, _nax_moe_dn_aff4_kernel
    rows, hidden = x.shape
    top_k = inds.shape[1]
    experts = w13.weight.shape[0]
    inter = w13.weight.shape[1] // 2
    nsg = 4 if rows < 1024 else 8
    tile_sz = nsg * 16
    max_tiles = (
        min(experts, rows * top_k)
        if rows <= tile_sz
        else ((rows * top_k) // tile_sz + experts)
    )
    w13_w, w13_s, w13_b = _moe_packed_views(w13)
    w2_w, w2_s, w2_b = _moe_packed_views(w2)

    if _nax_moe_route_kernel is None:
        _nax_moe_route_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_nax_moe_route",
            input_names=["INDS", "WTS"],
            output_names=["PERM", "INV", "AW", "TILES", "NTILES"],
            source=_NAX_MOE_ROUTE_SOURCE,
        )
        _nax_moe_gather_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_nax_moe_gather",
            input_names=["X", "PERM"],
            output_names=["XG"],
            source=_NAX_MOE_GATHER_SOURCE,
        )
        _nax_moe_comb_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_nax_moe_comb",
            input_names=["YD", "INV", "AW"],
            output_names=["OUT"],
            source=_NAX_MOE_COMB_SOURCE,
        )

    perm, inv, aw, tiles, ntiles = _nax_moe_route_kernel(
        inputs=[inds.reshape(-1).astype(mx.uint32), weights.reshape(-1).astype(mx.float32)],
        template=[("NEXP", experts), ("TOPK", top_k), ("T_TOK", rows), ("TILE_SZ", tile_sz)],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (rows * top_k,),
            (rows * top_k,),
            (rows * top_k,),
            (max_tiles * 4,),
            (1,),
        ],
        output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.uint32, mx.uint32],
    )
    xg = _nax_moe_gather_kernel(
        inputs=[x, perm],
        template=[("D", hidden)],
        grid=(rows * top_k * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows * top_k, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]

    if w13.mode == "nvfp4":
        if _nax_moe_up_nvfp4_kernel is None:
            _nax_moe_up_nvfp4_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_nax_moe_up_nvfp4",
                input_names=["XG", "W", "S", "GS_ARR", "TILES", "NTILES", "PR"],
                output_names=["ACT"],
                header=_NAX_MOE_HEADER,
                source=_NAX_MOE_UP_NVFP4_SOURCE,
            )
            _nax_moe_dn_nvfp4_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_nax_moe_dn_nvfp4",
                input_names=["ACT", "W", "S", "GS_ARR", "TILES", "NTILES", "INV_SC"],
                output_names=["YD"],
                header=_NAX_MOE_HEADER,
                source=_NAX_MOE_DN_NVFP4_SOURCE,
            )
        act = _nax_moe_up_nvfp4_kernel(
            inputs=[xg, w13_w, w13_s, w13.global_scale, tiles, ntiles, up_params],
            template=[("K_DIM", hidden), ("FF_DIM", inter)],
            grid=(max_tiles * nsg * 32, inter // 32, 1),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[(rows * top_k, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        yd = _nax_moe_dn_nvfp4_kernel(
            inputs=[act, w2_w, w2_s, w2.global_scale, tiles, ntiles, dn_params],
            template=[("K_DIM", inter), ("OUT_DIM", hidden)],
            grid=(max_tiles * nsg * 32, hidden // 64, 1),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[(rows * top_k, hidden)],
            output_dtypes=[mx.bfloat16],
        )[0]
    else:
        if _nax_moe_up_aff4_kernel is None:
            _nax_moe_up_aff4_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_nax_moe_up_aff4",
                input_names=["XG", "W", "S", "B", "TILES", "NTILES", "PR"],
                output_names=["ACT"],
                header=_NAX_MOE_HEADER,
                source=_NAX_MOE_UP_AFF4_SOURCE,
            )
            _nax_moe_dn_aff4_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_nax_moe_dn_aff4",
                input_names=["ACT", "W", "S", "B", "TILES", "NTILES"],
                output_names=["YD"],
                header=_NAX_MOE_HEADER,
                source=_NAX_MOE_DN_AFF4_SOURCE,
            )
        act = _nax_moe_up_aff4_kernel(
            inputs=[xg, w13_w, w13_s, w13_b, tiles, ntiles, up_params],
            template=[
                ("S_T", w13.scales.dtype),
                ("GS", w13.group_size),
                ("K_DIM", hidden),
                ("FF_DIM", inter),
            ],
            grid=(max_tiles * nsg * 32, inter // 32, 1),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[(rows * top_k, inter)],
            output_dtypes=[mx.bfloat16],
        )[0]
        yd = _nax_moe_dn_aff4_kernel(
            inputs=[act, w2_w, w2_s, w2_b, tiles, ntiles],
            template=[
                ("S_T", w2.scales.dtype),
                ("GS", w2.group_size),
                ("K_DIM", inter),
                ("OUT_DIM", hidden),
            ],
            grid=(max_tiles * nsg * 32, hidden // 64, 1),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[(rows * top_k, hidden)],
            output_dtypes=[mx.bfloat16],
        )[0]

    return _nax_moe_comb_kernel(
        inputs=[yd, inv, aw],
        template=[("D", hidden), ("TOPK", top_k)],
        grid=(rows * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ---------------------------------------------------------------------------
# M5 Neural Accelerator (Metal 4 matmul2d) FlashAttention with rel-bias + sink
# ---------------------------------------------------------------------------

_NAX_ATTN_PROBE_SOURCE = r"""
    threadgroup bfloat sp[32 * 64];
    auto tp = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(64, 32));
    auto tv0 = tensor((device bfloat*)V, dextents<int32_t, 2>(128, 64), array<int, 2>({1, 128}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(32, 128, 64, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<8>> mm_pv;
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    uint tid = thread_index_in_threadgroup;
    for (short i = 0; i < co.get_capacity() && i < 31; ++i) {
        if (co.is_valid_element(i)) {
            auto ix = co.get_multidimensional_index(i);
            OUT[tid * 32 + i] = int((ix[0] << 16) | ix[1]);
        } else {
            OUT[tid * 32 + i] = -1;
        }
    }
    OUT[tid * 32 + 31] = int(co.get_capacity());
"""

_NAX_FLASH_BANDED_SOURCE = r"""
    constexpr uint AQ = 32u;
    constexpr uint AK = 64u;
    const uint tgp_q = SWAP_GRID ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.x;
    const uint h = SWAP_GRID ? threadgroup_position_in_grid.x : threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    const uint L = uint(DIMS[0]);
    const uint S_LEN = uint(DIMS[1]);
    const uint lp = uint(DIMS[2 + b]);
    const uint t0 = tgp_q * AQ;
    if (t0 >= L) return;
    const uint nq = min(AQ, L - t0);
    const uint hk = h / (HQ / HKV);
    const float sc = SCALE[0];

    threadgroup float ss[AQ * AK];
    threadgroup bfloat sp[AQ * AK];
    threadgroup float sr[AQ];

    const device bfloat* q_ptr = Q + ((size_t(b) * HQ + h) * L + t0) * D;
    const device bfloat* kbase = K + (size_t(b) * HKV + hk) * S_LEN * D;
    const device bfloat* vbase = V + (size_t(b) * HKV + hk) * S_LEN * D;
    const device bfloat* rel_b = REL + (size_t(b) * L + t0) * HQ * R + h * R;

    auto tq = tensor((device bfloat*)q_ptr, dextents<int32_t, 2>(D, int(nq)), array<int, 2>({1, D}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(AK, AQ));
    auto tp = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AK, AQ));
    auto tv0 = tensor((device bfloat*)vbase, dextents<int32_t, 2>(D, AK), array<int, 2>({1, D}));

    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AK, D, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<8>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, D, AK, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<8>> mm_pv;

    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (short i = 0; i < 16; ++i) co[i] = 0.0f;
    ushort cd[4], cr[4];
    for (short q = 0; q < 4; ++q) {
        const auto ix = co.get_multidimensional_index(4 * q);
        cd[q] = ix[0];
        cr[q] = ix[1];
    }

    float M[4], Sm[4];
    for (short jj = 0; jj < 4; ++jj) {
        const uint j = jj * 8u + sg;
        M[jj] = (j < nq) ? SINK[((size_t(b) * L + (t0 + j)) * HQ) + h] : 0.0f;
        Sm[jj] = 1.0f;
    }

    const uint pfirst = S_LEN - L + t0;
    const uint plast = pfirst + nq - 1u;
    uint kstart = lp;
    if (WINDOW > 0 && pfirst + 1u > uint(WINDOW)) {
        kstart = max(kstart, pfirst + 1u - uint(WINDOW));
    }

    if (kstart <= plast) {
        for (uint kc = kstart; kc <= plast; kc += AK) {
            const int nk = int(min(AK, S_LEN - kc));
            {
                auto tk = tensor((device bfloat*)(kbase + size_t(kc) * D), dextents<int32_t, 2>(D, nk), array<int, 2>({1, D}));
                auto cs = mm_qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>();
                mm_qk.run(tq, tk, cs);
                cs.store(ts);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const int D0 = int(pfirst) - int(kc);
            const bool fast = (nq == AQ) && (kc >= lp) && (D0 >= int(AK - 1u))
                && (WINDOW == 0 || D0 + int(AQ - 1u) < WINDOW)
                && (D0 - int(AK - 1u) >= R || D0 + int(AQ - 1u) < R);
            const bool has_rel = (D0 - int(AK - 1u)) < R;
            for (short jj = 0; jj < 4; ++jj) {
                const uint j = jj * 8u + sg;
                float s0 = ss[j * AK + lane] * sc;
                float s1 = ss[j * AK + 32u + lane] * sc;
                if (fast) {
                    if (has_rel) {
                        const uint d0 = uint(D0 + int(j) - int(lane));
                        const device bfloat* rj = rel_b + size_t(j) * HQ * R;
                        s0 += float(rj[d0]);
                        s1 += float(rj[d0 - 32u]);
                    }
                } else {
                    const int d0 = D0 + int(j) - int(lane);
                    const int d1 = d0 - 32;
                    const uint k0 = kc + lane;
                    const uint k1 = kc + 32u + lane;
                    const bool v0 = (j < nq) && (k0 >= lp) && (d0 >= 0) && (WINDOW == 0 || d0 < WINDOW);
                    const bool v1 = (j < nq) && (k1 >= lp) && (d1 >= 0) && (WINDOW == 0 || d1 < WINDOW);
                    if (v0 && d0 < R) s0 += float(rel_b[size_t(j) * HQ * R + uint(d0)]);
                    if (v1 && d1 < R) s1 += float(rel_b[size_t(j) * HQ * R + uint(d1)]);
                    if (!v0) s0 = -INFINITY;
                    if (!v1) s1 = -INFINITY;
                }
                const float m = simd_max(max(M[jj], max(s0, s1)));
                const float ms = exp(M[jj] - m);
                M[jj] = m;
                const float p0 = exp(s0 - m);
                const float p1 = exp(s1 - m);
                sp[j * AK + lane] = bfloat(p0);
                sp[j * AK + 32u + lane] = bfloat(p1);
                Sm[jj] = Sm[jj] * ms + simd_sum(p0 + p1);
                if (lane == 0u) sr[j] = ms;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float r0 = sr[cr[0]], r1 = sr[cr[1]], r2 = sr[cr[2]], r3 = sr[cr[3]];
                co[0] *= r0; co[1] *= r0; co[2] *= r0; co[3] *= r0;
                co[4] *= r1; co[5] *= r1; co[6] *= r1; co[7] *= r1;
                co[8] *= r2; co[9] *= r2; co[10] *= r2; co[11] *= r2;
                co[12] *= r3; co[13] *= r3; co[14] *= r3; co[15] *= r3;
            }
            auto tv = tensor((device bfloat*)(vbase + size_t(kc) * D), dextents<int32_t, 2>(D, nk), array<int, 2>({1, D}));
            mm_pv.run(tp, tv, co);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short jj = 0; jj < 4; ++jj) {
        if (lane == 0u) sr[jj * 8u + sg] = 1.0f / Sm[jj];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device bfloat* out_b = OUT + (size_t(b) * L + t0) * HQ * D + h * D;
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            const float4 v = float4(co[4 * q], co[4 * q + 1], co[4 * q + 2], co[4 * q + 3]) * sr[row];
            *(device bfloat4*)(out_b + size_t(row) * HQ * D + cd[q]) = bfloat4(v);
        }
    }
"""

_nax_attn_supported: bool | None = None
_nax_flash_banded_kernel = None
_nax_flash_slab_kernel = None
_nax_flash_tail_kernel = None

_NAX_FLASH_SLAB_SOURCE = r"""
    constexpr uint AQ = 32u;
    constexpr uint AK = 64u;
    const uint h = threadgroup_position_in_grid.x;
    const uint tgp_q = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    const uint L = uint(DIMS[0]);
    const uint S_LEN = uint(DIMS[1]);
    const uint k_lo = uint(DIMS[2]);
    const uint k_hi = uint(DIMS[3]);
    const uint t0 = tgp_q * AQ;
    if (t0 >= L) return;
    const uint nq = min(AQ, L - t0);
    const uint hk = h / (HQ / HKV);
    const float sc = SCALE[0];

    threadgroup float ss[AQ * AK];
    threadgroup bfloat sp[AQ * AK];
    threadgroup float sr[AQ];

    const device bfloat* q_ptr = Q + ((size_t(b) * HQ + h) * L + t0) * D;
    const device bfloat* kbase = K + (size_t(b) * HKV + hk) * S_LEN * D;
    const device bfloat* vbase = V + (size_t(b) * HKV + hk) * S_LEN * D;

    auto tq = tensor((device bfloat*)q_ptr, dextents<int32_t, 2>(D, int(nq)), array<int, 2>({1, D}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(AK, AQ));
    auto tp = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AK, AQ));
    auto tv0 = tensor((device bfloat*)vbase, dextents<int32_t, 2>(D, AK), array<int, 2>({1, D}));

    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AK, D, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<8>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, D, AK, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<8>> mm_pv;

    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    ushort cd[4], cr[4];
    for (short q = 0; q < 4; ++q) {
        const auto ix = co.get_multidimensional_index(4 * q);
        cd[q] = ix[0];
        cr[q] = ix[1];
    }

    const size_t ml_base = (size_t(b) * HQ + h) * L + t0;
    const size_t o_base = ((size_t(b) * HQ + h) * L + t0) * D;

    float M[4], Sm[4];
    if (IS_FIRST) {
        for (short i = 0; i < 16; ++i) co[i] = 0.0f;
        for (short jj = 0; jj < 4; ++jj) {
            const uint j = jj * 8u + sg;
            M[jj] = (j < nq) ? SINK[((size_t(b) * L + (t0 + j)) * HQ) + h] : 0.0f;
            Sm[jj] = 1.0f;
        }
    } else {
        for (short q = 0; q < 4; ++q) {
            const uint row = cr[q];
            if (row < nq) {
                const size_t oo = o_base + size_t(row) * D + cd[q];
                co[4 * q] = O_IN[oo];
                co[4 * q + 1] = O_IN[oo + 1];
                co[4 * q + 2] = O_IN[oo + 2];
                co[4 * q + 3] = O_IN[oo + 3];
            } else {
                co[4 * q] = 0.0f; co[4 * q + 1] = 0.0f; co[4 * q + 2] = 0.0f; co[4 * q + 3] = 0.0f;
            }
        }
        for (short jj = 0; jj < 4; ++jj) {
            const uint j = jj * 8u + sg;
            M[jj] = (j < nq) ? M_IN[ml_base + j] : 0.0f;
            Sm[jj] = (j < nq) ? L_IN[ml_base + j] : 1.0f;
        }
    }

    for (uint kc = k_lo; kc < k_hi; kc += AK) {
        const int nk = int(min(AK, k_hi - kc));
        {
            auto tk = tensor((device bfloat*)(kbase + size_t(kc) * D), dextents<int32_t, 2>(D, nk), array<int, 2>({1, D}));
            auto cs = mm_qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>();
            mm_qk.run(tq, tk, cs);
            cs.store(ts);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short jj = 0; jj < 4; ++jj) {
            const uint j = jj * 8u + sg;
            float s0 = (j < nq && int(lane) < nk) ? (ss[j * AK + lane] * sc) : -INFINITY;
            float s1 = (j < nq && int(32u + lane) < nk) ? (ss[j * AK + 32u + lane] * sc) : -INFINITY;
            const float m = simd_max(max(M[jj], max(s0, s1)));
            const float ms = exp(M[jj] - m);
            M[jj] = m;
            const float p0 = exp(s0 - m);
            const float p1 = exp(s1 - m);
            sp[j * AK + lane] = bfloat(p0);
            sp[j * AK + 32u + lane] = bfloat(p1);
            Sm[jj] = Sm[jj] * ms + simd_sum(p0 + p1);
            if (lane == 0u) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float r0 = sr[cr[0]], r1 = sr[cr[1]], r2 = sr[cr[2]], r3 = sr[cr[3]];
            co[0] *= r0; co[1] *= r0; co[2] *= r0; co[3] *= r0;
            co[4] *= r1; co[5] *= r1; co[6] *= r1; co[7] *= r1;
            co[8] *= r2; co[9] *= r2; co[10] *= r2; co[11] *= r2;
            co[12] *= r3; co[13] *= r3; co[14] *= r3; co[15] *= r3;
        }
        auto tv = tensor((device bfloat*)(vbase + size_t(kc) * D), dextents<int32_t, 2>(D, nk), array<int, 2>({1, D}));
        mm_pv.run(tp, tv, co);
    }

    for (short jj = 0; jj < 4; ++jj) {
        const uint j = jj * 8u + sg;
        if (lane == 0u && j < nq) {
            M_OUT[ml_base + j] = M[jj];
            L_OUT[ml_base + j] = Sm[jj];
        }
    }
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            *(device float4*)(O_OUT + o_base + size_t(row) * D + cd[q]) =
                float4(co[4 * q], co[4 * q + 1], co[4 * q + 2], co[4 * q + 3]);
        }
    }
"""

_NAX_FLASH_TAIL_SOURCE = r"""
    constexpr uint AQ = 32u;
    constexpr uint AK = 64u;
    const uint tgp_q = threadgroup_position_in_grid.x;
    const uint h = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    const uint L = uint(DIMS[0]);
    const uint S_LEN = uint(DIMS[1]);
    const uint tail_lo = uint(DIMS[2]);
    const uint t0 = tgp_q * AQ;
    if (t0 >= L) return;
    const uint nq = min(AQ, L - t0);
    const uint hk = h / (HQ / HKV);
    const float sc = SCALE[0];

    threadgroup float ss[AQ * AK];
    threadgroup bfloat sp[AQ * AK];
    threadgroup float sr[AQ];

    const device bfloat* q_ptr = Q + ((size_t(b) * HQ + h) * L + t0) * D;
    const device bfloat* kbase = K + (size_t(b) * HKV + hk) * S_LEN * D;
    const device bfloat* vbase = V + (size_t(b) * HKV + hk) * S_LEN * D;
    const device bfloat* rel_b = REL + (size_t(b) * L + t0) * HQ * R + h * R;

    auto tq = tensor((device bfloat*)q_ptr, dextents<int32_t, 2>(D, int(nq)), array<int, 2>({1, D}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(AK, AQ));
    auto tp = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AK, AQ));
    auto tv0 = tensor((device bfloat*)vbase, dextents<int32_t, 2>(D, AK), array<int, 2>({1, D}));

    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AK, D, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<8>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, D, AK, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<8>> mm_pv;

    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    ushort cd[4], cr[4];
    for (short q = 0; q < 4; ++q) {
        const auto ix = co.get_multidimensional_index(4 * q);
        cd[q] = ix[0];
        cr[q] = ix[1];
    }

    const size_t ml_base = (size_t(b) * HQ + h) * L + t0;
    const size_t o_base = ((size_t(b) * HQ + h) * L + t0) * D;
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            const size_t oo = o_base + size_t(row) * D + cd[q];
            co[4 * q] = O_IN[oo];
            co[4 * q + 1] = O_IN[oo + 1];
            co[4 * q + 2] = O_IN[oo + 2];
            co[4 * q + 3] = O_IN[oo + 3];
        } else {
            co[4 * q] = 0.0f; co[4 * q + 1] = 0.0f; co[4 * q + 2] = 0.0f; co[4 * q + 3] = 0.0f;
        }
    }
    float M[4], Sm[4];
    for (short jj = 0; jj < 4; ++jj) {
        const uint j = jj * 8u + sg;
        M[jj] = (j < nq) ? M_IN[ml_base + j] : 0.0f;
        Sm[jj] = (j < nq) ? L_IN[ml_base + j] : 1.0f;
    }

    const uint pfirst = S_LEN - L + t0;
    const uint plast = pfirst + nq - 1u;
    for (uint kc = tail_lo; kc <= plast; kc += AK) {
        const int nk = int(min(AK, S_LEN - kc));
        {
            auto tk = tensor((device bfloat*)(kbase + size_t(kc) * D), dextents<int32_t, 2>(D, nk), array<int, 2>({1, D}));
            auto cs = mm_qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>();
            mm_qk.run(tq, tk, cs);
            cs.store(ts);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int D0 = int(pfirst) - int(kc);
        const bool fast = (nq == AQ) && (D0 >= int(AK - 1u))
            && (D0 - int(AK - 1u) >= R || D0 + int(AQ - 1u) < R);
        const bool has_rel = (D0 - int(AK - 1u)) < R;
        for (short jj = 0; jj < 4; ++jj) {
            const uint j = jj * 8u + sg;
            float s0 = ss[j * AK + lane] * sc;
            float s1 = ss[j * AK + 32u + lane] * sc;
            if (fast) {
                if (has_rel) {
                    const uint d0 = uint(D0 + int(j) - int(lane));
                    const device bfloat* rj = rel_b + size_t(j) * HQ * R;
                    s0 += float(rj[d0]);
                    s1 += float(rj[d0 - 32u]);
                }
            } else {
                const int d0 = D0 + int(j) - int(lane);
                const int d1 = d0 - 32;
                const bool v0 = (j < nq) && (d0 >= 0);
                const bool v1 = (j < nq) && (d1 >= 0);
                if (v0 && d0 < R) s0 += float(rel_b[size_t(j) * HQ * R + uint(d0)]);
                if (v1 && d1 < R) s1 += float(rel_b[size_t(j) * HQ * R + uint(d1)]);
                if (!v0) s0 = -INFINITY;
                if (!v1) s1 = -INFINITY;
            }
            const float m = simd_max(max(M[jj], max(s0, s1)));
            const float ms = exp(M[jj] - m);
            M[jj] = m;
            const float p0 = exp(s0 - m);
            const float p1 = exp(s1 - m);
            sp[j * AK + lane] = bfloat(p0);
            sp[j * AK + 32u + lane] = bfloat(p1);
            Sm[jj] = Sm[jj] * ms + simd_sum(p0 + p1);
            if (lane == 0u) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float r0 = sr[cr[0]], r1 = sr[cr[1]], r2 = sr[cr[2]], r3 = sr[cr[3]];
            co[0] *= r0; co[1] *= r0; co[2] *= r0; co[3] *= r0;
            co[4] *= r1; co[5] *= r1; co[6] *= r1; co[7] *= r1;
            co[8] *= r2; co[9] *= r2; co[10] *= r2; co[11] *= r2;
            co[12] *= r3; co[13] *= r3; co[14] *= r3; co[15] *= r3;
        }
        auto tv = tensor((device bfloat*)(vbase + size_t(kc) * D), dextents<int32_t, 2>(D, nk), array<int, 2>({1, D}));
        mm_pv.run(tp, tv, co);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short jj = 0; jj < 4; ++jj) {
        if (lane == 0u) sr[jj * 8u + sg] = 1.0f / Sm[jj];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device bfloat* out_b = OUT + (size_t(b) * L + t0) * HQ * D + h * D;
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            const float4 v = float4(co[4 * q], co[4 * q + 1], co[4 * q + 2], co[4 * q + 3]) * sr[row];
            *(device bfloat4*)(out_b + size_t(row) * HQ * D + cd[q]) = bfloat4(v);
        }
    }
"""


def can_use_nax_attn() -> bool:
    """Return True if Metal 4 matmul2d PV accumulator matches 4 quads of 4 consecutive dims."""
    global _nax_attn_supported
    if _nax_attn_supported is not None:
        return _nax_attn_supported
    if not is_nax_available() or mx.default_device() != mx.gpu:
        _nax_attn_supported = False
        return False
    try:
        probe = mx.fast.metal_kernel(
            name="omlx_inkling2_nax_attn_probe",
            input_names=["V"],
            output_names=["OUT"],
            header=_NAX_MOE_HEADER,
            source=_NAX_ATTN_PROBE_SOURCE,
        )
        out = probe(
            inputs=[mx.zeros((64, 128), dtype=mx.bfloat16)],
            output_shapes=[(256 * 32,)],
            output_dtypes=[mx.int32],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]
        mx.eval(out)
        arr = out.tolist()
        ok = arr[31] == 16
        if ok:
            for tid in range(256):
                base = tid * 32
                for q in range(4):
                    d0 = arr[base + 4 * q] >> 16
                    r0 = arr[base + 4 * q] & 0xFFFF
                    if d0 % 4 != 0:
                        ok = False
                        break
                    for j in range(1, 4):
                        vj = arr[base + 4 * q + j]
                        if (vj >> 16) != d0 + j or (vj & 0xFFFF) != r0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
        _nax_attn_supported = ok
    except Exception:
        _nax_attn_supported = False
    return _nax_attn_supported


def flash_banded_attention_nax(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sink: mx.array,
    *,
    scale: float,
    window: int,
    left_padding=None,
) -> mx.array:
    """M5 Metal 4 matmul2d banded-sink FlashAttention (BF16 or FP8 E4M3 K/V)."""
    global _nax_flash_banded_kernel, _nax_flash_slab_kernel, _nax_flash_tail_kernel
    if _nax_flash_banded_kernel is None:
        _nax_flash_banded_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_banded_nax",
            input_names=["Q", "K", "V", "REL", "SINK", "DIMS", "SCALE"],
            output_names=["OUT"],
            header=_NAX_MOE_HEADER,
            source=_NAX_FLASH_BANDED_SOURCE,
        )
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    r_ext = rel.shape[-1]
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    if window and s > length + window - 1:
        lo = s - length - window + 1
        k = k[:, :, lo:]
        v = v[:, :, lo:]
        s = k.shape[2]
        lp = mx.maximum(lp - lo, 0)
    if k.dtype == mx.uint8:
        k = fp8_e4m3_dequantize(k, mx.bfloat16)
        v = fp8_e4m3_dequantize(v, mx.bfloat16)

    q_c = mx.contiguous(q)
    k_c = mx.contiguous(k)
    v_c = mx.contiguous(v)
    rel_c = mx.contiguous(rel)
    sink_c = mx.contiguous(sink.astype(mx.float32))
    sc_arr = mx.array([scale], dtype=mx.float32)
    q_tiles = (length + 31) // 32

    # At very long full-attention contexts (>32K keys), one BF16 KV head pair
    # exceeds M5 Pro's SLC. Sweep old keys in 2K slabs (1 MB/KV head) across all
    # query tiles, then finish the relative-bias + causal tail in one pass.
    tail_lo = ((s - length - r_ext) // 64) * 64
    if not window and left_padding is None and b == 1 and length >= 512 and s > 32768 and tail_lo >= 4096:
        if _nax_flash_slab_kernel is None:
            _nax_flash_slab_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_flash_slab_nax",
                input_names=["Q", "K", "V", "SINK", "O_IN", "M_IN", "L_IN", "DIMS", "SCALE"],
                output_names=["O_OUT", "M_OUT", "L_OUT"],
                header=_NAX_MOE_HEADER,
                source=_NAX_FLASH_SLAB_SOURCE,
            )
            _nax_flash_tail_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_flash_tail_nax",
                input_names=["Q", "K", "V", "REL", "O_IN", "M_IN", "L_IN", "DIMS", "SCALE"],
                output_names=["OUT"],
                header=_NAX_MOE_HEADER,
                source=_NAX_FLASH_TAIL_SOURCE,
            )
        slab = 2048
        o_acc = m_acc = l_acc = _dummy_f32()
        first = 1
        for k_lo in range(0, tail_lo, slab):
            k_hi = min(tail_lo, k_lo + slab)
            sdims = mx.array([length, s, k_lo, k_hi], dtype=mx.int32)
            o_acc, m_acc, l_acc = _nax_flash_slab_kernel(
                inputs=[q_c, k_c, v_c, sink_c, o_acc, m_acc, l_acc, sdims, sc_arr],
                template=[("D", d), ("HQ", hq), ("HKV", hkv), ("IS_FIRST", first)],
                grid=(hq * 256, q_tiles, b),
                threadgroup=(256, 1, 1),
                output_shapes=[(b, hq, length, d), (b, hq, length), (b, hq, length)],
                output_dtypes=[mx.float32, mx.float32, mx.float32],
            )
            first = 0
        tdims = mx.array([length, s, tail_lo], dtype=mx.int32)
        out = _nax_flash_tail_kernel(
            inputs=[q_c, k_c, v_c, rel_c, o_acc, m_acc, l_acc, tdims, sc_arr],
            template=[("D", d), ("HQ", hq), ("HKV", hkv), ("R", r_ext)],
            grid=(q_tiles * 256, hq, b),
            threadgroup=(256, 1, 1),
            output_shapes=[(b, length, hq, d)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return out.transpose(0, 2, 1, 3)

    dims = mx.concatenate([mx.array([length, s], dtype=mx.int32), lp])
    swap_grid = 1 if (not window and s > 4096) else 0
    grid = (hq * 256, q_tiles, b) if swap_grid else (q_tiles * 256, hq, b)
    out = _nax_flash_banded_kernel(
        inputs=[q_c, k_c, v_c, rel_c, sink_c, dims, sc_arr],
        template=[
            ("D", d),
            ("HQ", hq),
            ("HKV", hkv),
            ("R", r_ext),
            ("WINDOW", int(window)),
            ("SWAP_GRID", swap_grid),
        ],
        grid=grid,
        threadgroup=(256, 1, 1),
        output_shapes=[(b, length, hq, d)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out.transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Fused attention head prep (L2-clip + V RMS norm + tau + head-major transpose)
# and fused attention output gating (out * sigmoid(gate))
# ---------------------------------------------------------------------------

_FUSED_HEAD_PREP_SOURCE = r"""
    constexpr uint PER = HD / 32u;
    constexpr uint SLOTS = 2u * NH + 2u * NKV;
    const uint lane = thread_index_in_simdgroup;
    const uint sid = threadgroup_position_in_grid.y;
    const uint row = sid / SLOTS;
    const uint slot = sid % SLOTS;
    const uint b = row / L;
    const uint l = row % L;
    const float tv = HAS_TAU ? TAU[b * TAU_SB + l] : 1.0f;

    if (slot < NH) {
        const size_t soff = size_t(slot) * HD + lane * PER;
        const size_t rbase = size_t(b) * QRAW_strides[0] + size_t(l) * QRAW_strides[1];
        const size_t dst_base = ((size_t(b) * NH + slot) * L + l) * HD + lane * PER;
        float xs[PER];
        float acc = 0.0f;
        for (uint j = 0u; j < PER; ++j) {
            xs[j] = float(QRAW[rbase + (soff + j) * QRAW_strides[2]]) * float(QSCALE[soff + j]);
            acc += xs[j] * xs[j];
        }
        acc = simd_sum(acc);
        const float f = min(PR[0] / (metal::precise::sqrt(acc) + 1e-6f), 1.0f);
        for (uint j = 0u; j < PER; ++j) {
            T v = T(xs[j] * f);
            if (HAS_TAU) v = T(float(v) * tv);
            QO[dst_base + j] = v;
        }
    } else if (slot < NH + NKV) {
        const uint h = slot - NH;
        const size_t coff = size_t(h) * HD + lane * PER;
        const size_t rbase = size_t(b) * KRAW_strides[0] + size_t(l) * KRAW_strides[1];
        const size_t dst_base = ((size_t(b) * NKV + h) * L + l) * HD + lane * PER;
        float xs[PER];
        float acc = 0.0f;
        for (uint j = 0u; j < PER; ++j) {
            xs[j] = float(KRAW[rbase + (coff + j) * KRAW_strides[2]]);
            acc += xs[j] * xs[j];
        }
        acc = simd_sum(acc);
        const float f = min(PR[0] / (metal::precise::sqrt(acc) + 1e-6f), 1.0f);
        for (uint j = 0u; j < PER; ++j) {
            T v = T(xs[j] * f);
            if (FP8_KV) {
                KO[dst_base + j] = KVT(float_to_fp8_e4m3(float(v)));
            } else {
                KO[dst_base + j] = KVT(v);
            }
        }
    } else if (slot < NH + 2u * NKV) {
        const uint h = slot - NH - NKV;
        const size_t coff = size_t(h) * HD + lane * PER;
        const size_t rbase = size_t(b) * VRAW_strides[0] + size_t(l) * VRAW_strides[1];
        const size_t dst_base = ((size_t(b) * NKV + h) * L + l) * HD + lane * PER;
        float xs[PER];
        float acc = 0.0f;
        for (uint j = 0u; j < PER; ++j) {
            xs[j] = float(VRAW[rbase + (coff + j) * VRAW_strides[2]]);
            acc += xs[j] * xs[j];
        }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / float(HD) + 1e-6f);
        for (uint j = 0u; j < PER; ++j) {
            T v = T(xs[j] * inv);
            if (FP8_KV) {
                VO[dst_base + j] = KVT(float_to_fp8_e4m3(float(v)));
            } else {
                VO[dst_base + j] = KVT(v);
            }
        }
    } else {
        const uint h = slot - NH - 2u * NKV;
        const size_t rbase = size_t(b) * RRAW_strides[0] + size_t(l) * RRAW_strides[1];
        const size_t dst_hbase = size_t(row) * (NH * DREL) + size_t(h) * DREL;
        const size_t soff = size_t(h) * DREL;
        float xj = (lane < DREL)
            ? (float(RRAW[rbase + (soff + lane) * RRAW_strides[2]]) * float(RSCALE[soff + lane]))
            : 0.0f;
        float acc = simd_sum(xj * xj);
        if (lane < DREL) {
            const float f = min(PR[1] / (metal::precise::sqrt(acc) + 1e-6f), 1.0f);
            T v = T(xj * f);
            if (HAS_TAU) v = T(float(v) * tv);
            RO[dst_hbase + lane] = v;
        }
    }
"""

_FUSED_GATED_OUT_SOURCE = r"""
    const uint idx = thread_position_in_grid.x;
    const uint total = B * L * HQ * D;
    if (idx >= total) return;
    const uint d = idx % D;
    const uint t1 = idx / D;
    const uint h = t1 % HQ;
    const uint t2 = t1 / HQ;
    const uint l = t2 % L;
    const uint b = t2 / L;
    const size_t o_off = size_t(b) * OUT_strides[0] + size_t(h) * OUT_strides[1]
                       + size_t(l) * OUT_strides[2] + size_t(d) * OUT_strides[3];
    const size_t g_off = size_t(b) * GATE_strides[0] + size_t(l) * GATE_strides[1]
                       + size_t(h * D + d) * GATE_strides[2];
    const float ov = float(OUT[o_off]);
    const float gv = float(GATE[g_off]);
    const float sig = 1.0f / (1.0f + metal::exp(-gv));
    Y[idx] = T(ov * sig);
"""

_fused_head_prep_kernel = None
_fused_gated_out_kernel = None


def fused_head_prep(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r: mx.array,
    q_scale: mx.array,
    r_scale: mx.array,
    params: mx.array,
    tau: mx.array | None,
    *,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    d_rel: int,
    fp8_kv: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused L2-clip (q, k, r), unit-RMS (v), optional tau scale, FP8 KV encode and head-major transpose."""
    global _fused_head_prep_kernel
    if _fused_head_prep_kernel is None:
        _fused_head_prep_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_fused_head_prep",
            input_names=["QRAW", "KRAW", "VRAW", "RRAW", "QSCALE", "RSCALE", "PR", "TAU"],
            output_names=["QO", "KO", "VO", "RO"],
            source=_FUSED_HEAD_PREP_SOURCE,
            header=_FP8_E4M3_HEADER,
            ensure_row_contiguous=False,
        )
    b, length, _ = q.shape if n_heads > 0 else k.shape
    has_tau = 1 if tau is not None else 0
    if has_tau:
        tau_sb = 0 if tau.shape[0] == 1 else length
        tau_arr = mx.contiguous(tau.astype(mx.float32).reshape(-1))
    else:
        tau_sb = 0
        tau_arr = _dummy_f32()
    kv_dtype = mx.uint8 if fp8_kv else k.dtype
    slots = 2 * n_heads + 2 * n_kv
    return _fused_head_prep_kernel(
        inputs=[
            q,
            k,
            v,
            r,
            mx.contiguous(q_scale),
            mx.contiguous(r_scale),
            params,
            tau_arr,
        ],
        template=[
            ("T", q.dtype if n_heads > 0 else k.dtype),
            ("KVT", kv_dtype),
            ("FP8_KV", 1 if fp8_kv else 0),
            ("L", length),
            ("NH", n_heads),
            ("NKV", n_kv),
            ("HD", head_dim),
            ("DREL", d_rel),
            ("HAS_TAU", has_tau),
            ("TAU_SB", tau_sb),
        ],
        grid=(32, b * length * slots, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (b, n_heads, length, head_dim) if n_heads > 0 else (1,),
            (b, n_kv, length, head_dim) if n_kv > 0 else (1,),
            (b, n_kv, length, head_dim) if n_kv > 0 else (1,),
            (b, length, n_heads, d_rel) if n_heads > 0 else (1,),
        ],
        output_dtypes=[
            q.dtype if n_heads > 0 else k.dtype,
            kv_dtype,
            kv_dtype,
            r.dtype if n_heads > 0 else k.dtype,
        ],
    )


def fused_gated_output(out: mx.array, gate: mx.array) -> mx.array:
    """Fused head-major-to-token-major transpose + out * sigmoid(gate)."""
    global _fused_gated_out_kernel
    if _fused_gated_out_kernel is None:
        _fused_gated_out_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_fused_gated_out",
            input_names=["OUT", "GATE"],
            output_names=["Y"],
            source=_FUSED_GATED_OUT_SOURCE,
            ensure_row_contiguous=False,
        )
    b, hq, length, d = out.shape
    total = b * length * hq * d
    return _fused_gated_out_kernel(
        inputs=[out, gate],
        template=[("T", gate.dtype), ("B", b), ("L", length), ("HQ", hq), ("D", d)],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, length, hq * d)],
        output_dtypes=[gate.dtype],
    )[0]
