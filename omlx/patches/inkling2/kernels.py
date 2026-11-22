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
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        if (WINDOW == 0 && k0 >= lp && k0 + BK <= k_end && k0 + BK <= a_first - R + 1) {
            for (int kb = 0; kb < BK / 8; ++kb) {
                float s0 = row_ok ? Sf[kb].thread_elements()[0] * qscale : -INFINITY;
                float s1 = row_ok ? Sf[kb].thread_elements()[1] * qscale : -INFINITY;
                Sf[kb].thread_elements()[0] = s0;
                Sf[kb].thread_elements()[1] = s1;
                rmax = max(rmax, max(s0, s1));
            }
        } else {
            for (int kb = 0; kb < BK / 8; ++kb) {
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
        for (int kb = 0; kb < BK / 8; ++kb) {
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
        if (SINGLE) {
            const float inv = 1.0f / l;
            size_t base_o = (((size_t)b * HQ + h) * L + q_pos) * D;
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
    const device float* st = state + (size_t)b * 3 * C;
    const device T* xb = x + (size_t)b * L * C;
    const device WT* wp = w + (size_t)c * 4;
    float x0 = (l >= 3) ? float(xb[(size_t)(l - 3) * C + c]) : st[(size_t)l * C + c];
    float x1 = (l >= 2) ? float(xb[(size_t)(l - 2) * C + c]) : st[(size_t)(l + 1) * C + c];
    float x2 = (l >= 1) ? float(xb[(size_t)(l - 1) * C + c]) : st[(size_t)(l + 2) * C + c];
    float x3 = float(xb[(size_t)l * C + c]);
    float conv = float(wp[0]) * x0 + float(wp[1]) * x1
        + float(wp[2]) * x2 + float(wp[3]) * x3;
    float y = (float(WT(conv)) + x3) * scale[0];
    out[idx] = T(y);
    if (L >= 3) {
        if (l >= L - 3) {
            next_state[((size_t)b * 3 + (l - (L - 3))) * C + c] = x3;
        }
    } else if (l == L - 1) {
        device float* nst = next_state + (size_t)b * 3 * C;
        if (L == 1) {
            nst[0 * C + c] = st[1 * C + c];
            nst[1 * C + c] = st[2 * C + c];
            nst[2 * C + c] = x3;
        } else {
            nst[0 * C + c] = st[2 * C + c];
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
) -> tuple[mx.array, mx.array]:
    """Fused K=4 depthwise causal conv, residual add, scale, and next state."""
    global _static_conv4_kernel
    if _static_conv4_kernel is None:
        _static_conv4_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_static_conv4",
            input_names=["x", "w", "state", "scale"],
            output_names=["out", "next_state"],
            source=_STATIC_CONV4_SOURCE,
        )
    b, length, channels = x.shape
    out, next_state = _static_conv4_kernel(
        inputs=[x, w, state.astype(mx.float32), scale],
        template=[
            ("T", x.dtype),
            ("WT", w.dtype),
            ("B", b),
            ("L", length),
            ("C", channels),
        ],
        grid=(b * length * channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, length, channels), (b, 3, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out, next_state


# ---------------------------------------------------------------------------
# Fast exact FP32 router and fused multi-row packed MoE kernels (NVFP4 / Affine-4)
# ---------------------------------------------------------------------------

_FAST_ROUTER_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint e = thread_position_in_grid.y;
    const uint s = thread_position_in_grid.z;
    const device T* x = X + size_t(s) * D;
    const device float* w = RT + size_t(e) * D;
    float acc = 0.0f;
    for (uint i = lane * 4u; i < D; i += 128u) {
        float4 w4 = *(const device float4*)(w + i);
        acc += float(x[i]) * w4.x + float(x[i + 1u]) * w4.y
             + float(x[i + 2u]) * w4.z + float(x[i + 3u]) * w4.w;
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
    """Exact FP32 router dot product for small batches (avoids M5 NAX slow TF32 path)."""
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
        template=[("T", x.dtype), ("D", hidden), ("E", experts)],
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
    if w13.mode == "nvfp4":
        if rows >= 3 and w13.weight.shape[0] <= 128:
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
                inputs=[x, w13.weight, w13.scales, w13.global_scale, up_params, plan],
                template=[("T", x.dtype), ("K", hidden), ("I", inter), ("KSEL", top_k)],
                grid=(32, inter, rows * top_k),
                threadgroup=(32, 2, 1),
                output_shapes=[(rows * top_k, inter)],
                output_dtypes=[x.dtype],
            )[0]
            yslot = _downs_nvfp4_grp_kernel(
                inputs=[act, w2.weight, w2.scales, w2.global_scale, wts_f, dn_params, plan],
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
            inputs=[x, w13.weight, w13.scales, idx_f, w13.global_scale, up_params],
            template=[("T", x.dtype), ("K", hidden), ("I", inter), ("KSEL", top_k)],
            grid=(32, inter, rows * top_k),
            threadgroup=(32, 2, 1),
            output_shapes=[(rows * top_k, inter)],
            output_dtypes=[x.dtype],
        )[0]
        return _downs_nvfp4_kernel(
            inputs=[act, w2.weight, w2.scales, idx_f, w2.global_scale, wts_f, dn_params],
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
        inputs=[x, w13.weight, w13.scales, w13.biases, idx_f, up_params],
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
        inputs=[act, w2.weight, w2.scales, w2.biases, idx_f, wts_f],
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
