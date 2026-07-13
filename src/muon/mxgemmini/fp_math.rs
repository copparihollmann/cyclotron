//! MX (microscaling) floating-point numerics.
//!
//! Faithful Rust port of
//! `chipyard-mx/generators/gemmini/software/libgemmini/mx_fp_math.h`.
//! The spike libgemmini model is the numeric oracle for the MX-Gemmini
//! accelerator; this module reproduces it bit-for-bit so cyclotron's functional
//! MX co-model agrees with spike (and, transitively, the Python golden the spike
//! model was itself validated against). See the header for the rationale behind
//! each rounding/canonicalization choice — the comments are carried over verbatim
//! where they explain a non-obvious invariant.
//!
//! Kept deliberately close to the C++ (same function names, same branch
//! structure) so the two can be diffed. The only structural change is that C's
//! `ldexpf`/`frexpf` are provided as exact local helpers.

/// Exact `ldexpf(x, n)` == x * 2^n rounded once to f32. Computing in f64 is exact
/// because an f32 value times an exact power of two is representable in f64
/// (for |n| <= 1023), so the single f64->f32 narrowing reproduces ldexpf's RNE.
#[inline]
pub fn ldexpf(x: f32, n: i32) -> f32 {
    ((x as f64) * (2.0f64).powi(n)) as f32
}

/// `frexpf`: return (m, e) with x == m * 2^e and m in [0.5, 1) (or x itself for
/// 0/inf/nan). Matches C frexpf via bit extraction.
#[inline]
pub fn frexpf(x: f32) -> (f32, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    if exp == 0 {
        // subnormal: normalize by scaling up, then correct the exponent
        let (m, e) = frexpf(x * ldexpf(1.0, 25));
        return (m, e - 25);
    }
    // Force the biased exponent to 126 so the value lands in [0.5, 1).
    let mant = (bits & 0x807F_FFFF) | (126 << 23);
    (f32::from_bits(mant), exp - 126)
}

#[inline]
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    // Canonicalize -0 to +0 to match the Python golden, which never emits BF16
    // 0x8000 (Python's matmul output uses unsigned 0).
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    if ((bits >> 23) & 0xFF) == 0xFF {
        let mut bf = bits >> 16;
        if (bits & 0x7F_FFFF) != 0 && (bf & 0x40) == 0 {
            bf |= 0x40;
        }
        return (bf & 0xFFFF) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
    let out = ((rounded >> 16) & 0xFFFF) as u16;
    // Tiny inputs that round down to zero magnitude: also canonicalize.
    if (out & 0x7FFF) == 0 {
        return 0;
    }
    out
}

#[inline]
pub fn bf16_to_f32(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << 16)
}

#[inline]
pub fn bf16_round(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(x))
}

#[inline]
pub fn fp_emax(e_bits: i32, m_bits: i32) -> i32 {
    let bias = (1 << (e_bits - 1)) - 1;
    if e_bits == 4 && m_bits == 3 {
        return bias + 1;
    }
    if e_bits == 2 && m_bits == 1 {
        return (1 << e_bits) - 1 - bias;
    }
    bias
}

#[inline]
pub fn mx_product_saturate(x: f32, e_bits: i32, m_bits: i32) -> f32 {
    let bias = (1 << (e_bits - 1)) - 1;
    let is_mx_fp8 = e_bits == 4 && m_bits == 3;
    let emax = if is_mx_fp8 { bias + 1 } else { bias };
    let scale = (1 << m_bits) as f32;
    let max_mant = if is_mx_fp8 {
        (1 << m_bits) - 2
    } else {
        (1 << m_bits) - 1
    };
    let max_normal = ldexpf(1.0 + max_mant as f32 / scale, emax);
    let sat_man = (1 << m_bits) - 2;
    let sat_val = ldexpf(1.0 + sat_man as f32 / scale, bias + 1);
    let ax = x.abs();
    // Mirror Python's torch.sign(-0.0) == 0.0 — canonicalize signed zero to +0.
    if ax == 0.0 {
        return 0.0;
    }
    if ax > max_normal {
        return if x.is_sign_negative() { -sat_val } else { sat_val };
    }
    x
}

#[inline]
pub fn mx_product_quantize_trunc(x: f32, e_bits: i32, frac_bits: i32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return mx_product_saturate(x, e_bits, frac_bits);
    }
    let (m, e) = frexpf(x.abs());
    let big_e = e - 1;
    let scale = (1 << frac_bits) as f32;
    let frac = 2.0 * m - 1.0;
    let frac_q = (frac * scale).floor() / scale;
    let val = ldexpf(1.0 + frac_q, big_e);
    let out = if x.is_sign_negative() { -val } else { val };
    mx_product_saturate(out, e_bits, frac_bits)
}

#[inline]
pub fn fp_quantize_trunc(x: f32, e_bits: i32, m_bits: i32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return 0.0;
    }
    let neg = x.is_sign_negative();
    let ax = x.abs();
    let (_, mut big_e) = frexpf(ax);
    big_e -= 1;
    let bias = (1 << (e_bits - 1)) - 1;
    let emin = 1 - bias;
    let is_mx_fp8 = e_bits == 4 && m_bits == 3;
    let emax = fp_emax(e_bits, m_bits);
    let subnorm_delta = ldexpf(1.0, emin - m_bits);
    let scale = (1 << m_bits) as f32;
    let v;
    if big_e < emin {
        let mut k = (ax / subnorm_delta).floor() as i32;
        if k > (1 << m_bits) - 1 {
            k = (1 << m_bits) - 1;
        }
        v = k as f32 * subnorm_delta;
    } else if big_e > emax {
        let base_max = ldexpf(1.0, emax);
        let delta_max = base_max / scale;
        let max_mant = if is_mx_fp8 {
            (1 << m_bits) - 2
        } else {
            (1 << m_bits) - 1
        };
        v = base_max + max_mant as f32 * delta_max;
    } else {
        let base = ldexpf(1.0, big_e);
        let delta = base / scale;
        let t = (ax - base) / delta;
        let mut k = t.floor() as i32;
        let mut hi = (1 << m_bits) - 1;
        if is_mx_fp8 && big_e == emax {
            hi = (1 << m_bits) - 2;
        }
        if k > hi {
            k = hi;
        }
        if k < 0 {
            k = 0;
        }
        v = base + k as f32 * delta;
    }
    if neg {
        -v
    } else {
        v
    }
}

#[inline]
pub fn round_div_pow2_rne_u64(n: u64, shift: i32) -> u64 {
    if shift <= 0 {
        return n << (-shift);
    }
    let q = n >> shift;
    let rem = n & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if rem > half || (rem == half && (q & 1) != 0) {
        q + 1
    } else {
        q
    }
}

#[inline]
pub fn fp_quantize_rne_scalar(x: f32, e_bits: i32, m_bits: i32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return 0.0;
    }
    let neg = x.is_sign_negative();
    let ax = x.abs();

    let bits = ax.to_bits();
    let mant = bits & 0x7F_FFFF;
    let e_raw = ((bits >> 23) & 0xFF) as i32;
    let num: u64;
    let exp2: i32;
    if e_raw == 0 {
        if mant == 0 {
            return if neg { -0.0 } else { 0.0 };
        }
        num = mant as u64;
        exp2 = 1 - 127 - 23;
    } else {
        num = (1u64 << 23) | mant as u64;
        exp2 = e_raw - 127 - 23;
    }
    let p = 63 - num.leading_zeros() as i32;
    let big_e = exp2 + p;

    let bias = (1 << (e_bits - 1)) - 1;
    let emin = 1 - bias;
    let emax = bias;

    if big_e < emin {
        let shift = (emin - m_bits) - exp2;
        let sub_sig = round_div_pow2_rne_u64(num, shift);
        // Match Python _round_dyadic_to_scalar: underflow returns unsigned +0,
        // dropping the sign (so signed-zero never leaks into the accumulator and
        // ultimately into bf16 0x8000).
        if sub_sig == 0 {
            return 0.0;
        }
        if sub_sig >= (1u64 << m_bits) {
            let mut total_sig = sub_sig;
            let mut e_fin = emin;
            if total_sig >= (1u64 << (m_bits + 1)) {
                total_sig >>= 1;
                e_fin += 1;
            }
            if e_fin > emax {
                return if neg { f32::NEG_INFINITY } else { f32::INFINITY };
            }
            let r = ldexpf(total_sig as f32, e_fin - m_bits);
            return if neg { -r } else { r };
        }
        let r = ldexpf(sub_sig as f32, emin - m_bits);
        return if neg { -r } else { r };
    }

    let mut total_sig = round_div_pow2_rne_u64(num, p - m_bits);
    let mut big_e = big_e;
    if total_sig >= (1u64 << (m_bits + 1)) {
        total_sig >>= 1;
        big_e += 1;
    }
    if big_e > emax {
        return if neg { f32::NEG_INFINITY } else { f32::INFINITY };
    }
    let r = ldexpf(total_sig as f32, big_e - m_bits);
    if neg {
        -r
    } else {
        r
    }
}

#[inline]
pub fn fp_quantize_rne(x: f32, e_bits: i32, m_bits: i32) -> f32 {
    if e_bits == 8 && m_bits == 7 {
        return bf16_round(x);
    }
    fp_quantize_rne_scalar(x, e_bits, m_bits)
}

#[inline]
pub fn fp_add_exact(x: f32, y: f32, e_bits: i32, m_bits: i32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() || y.is_infinite() {
        if x.is_infinite() && y.is_infinite() && (x.is_sign_negative() != y.is_sign_negative()) {
            return f32::NAN;
        }
        return if x.is_infinite() { x } else { y };
    }
    if x == 0.0 {
        return y;
    }
    if y == 0.0 {
        return x;
    }
    let s = x + y;
    fp_quantize_rne_scalar(s, e_bits, m_bits)
}

#[inline]
pub fn bf16_accum_add(x: f32, y: f32) -> f32 {
    fp_add_exact(bf16_round(x), bf16_round(y), 8, 7)
}

/// Banker's rounding (round-half-to-even) matching Python 3's int(round(x)).
#[inline]
pub fn round_half_to_even(x: f32) -> i32 {
    let fl = x.floor();
    let frac = x - fl;
    let fli = fl as i32;
    if frac < 0.5 {
        return fli;
    }
    if frac > 0.5 {
        return fli + 1;
    }
    if (fli & 1) != 0 {
        fli + 1
    } else {
        fli
    }
}

#[inline]
pub fn fp8_e4m3_to_code(v: f32) -> u8 {
    if v == 0.0 || !v.is_finite() {
        return 0;
    }
    let s = if v.is_sign_negative() { 1 } else { 0 };
    let av = v.abs();
    let big_e = av.log2().floor() as i32;
    let bias = 7;
    let emin = -6;
    let emax = 8;
    if big_e < emin {
        return 0;
    }
    let mut e_used = big_e;
    let mant;
    if big_e > emax {
        e_used = emax;
        mant = 6;
    } else {
        let base = ldexpf(1.0, e_used);
        let delta = base / 8.0;
        let mut k = round_half_to_even((av - base) / delta);
        if k >= 8 {
            e_used += 1;
            k = 0;
            if e_used > emax {
                e_used = emax;
                k = 6;
            }
        } else {
            let hi = if e_used == emax { 6 } else { 7 };
            if k > hi {
                k = hi;
            }
            if k < 0 {
                k = 0;
            }
        }
        mant = k;
    }
    ((s << 7) | (((e_used + bias) & 0xF) << 3) | (mant & 0x7)) as u8
}

#[inline]
pub fn fp8_e4m3_decode(code: u8) -> f32 {
    let s = (code >> 7) & 1;
    let e = ((code >> 3) & 0xF) as i32;
    let m = (code & 0x7) as i32;
    let bias = 7;
    let val = if e == 0 {
        (m as f32 / 8.0) * ldexpf(1.0, 1 - bias)
    } else {
        (1.0 + m as f32 / 8.0) * ldexpf(1.0, e - bias)
    };
    if s != 0 {
        -val
    } else {
        val
    }
}

#[inline]
pub fn fp4_e2m1_decode(code: u8) -> f32 {
    let s = (code >> 3) & 1;
    let e = ((code >> 1) & 0x3) as i32;
    let m = (code & 0x1) as i32;
    let val = if e == 0 {
        m as f32 / 2.0
    } else {
        (1.0 + m as f32 / 2.0) * ldexpf(1.0, e - 1)
    };
    if s != 0 {
        -val
    } else {
        val
    }
}

#[inline]
pub fn fpe8m0_decode(code: u8) -> f32 {
    if code == 0xFF {
        return f32::NAN;
    }
    ldexpf(1.0, code as i32 - 127)
}

/// Decode a 6-bit FP6 E3M2 code (1 sign | 3 exp | 2 mant, bias=3) to float.
#[inline]
pub fn fp6_e3m2_decode(code: u8) -> f32 {
    let s = (code >> 5) & 1;
    let e = ((code >> 2) & 0x7) as i32;
    let m = (code & 0x3) as i32;
    let val = if e == 0 {
        m as f32 * 0.0625
    } else {
        (1.0 + m as f32 * 0.25) * ldexpf(1.0, e - 3)
    };
    if s != 0 {
        -val
    } else {
        val
    }
}

// --- Output-side encoders, used by the requant post-passes ---

/// BF16 bits -> E4M2 float (RNE). Mirrors `lut_mapping_demo._bf16_to_e4m2_rne`.
pub fn bf16_bits_to_e4m2_rne(bits: u16) -> f32 {
    let sign_bit = (bits >> 15) & 1;
    let big_e = ((bits >> 7) & 0xFF) as i32;
    let m = (bits & 0x7F) as i32;
    if big_e == 0 {
        return 0.0;
    }
    let sign_f = if sign_bit != 0 { -1.0f32 } else { 1.0f32 };
    if big_e == 255 {
        return sign_f * f32::INFINITY;
    }
    let e = big_e - 127;
    if (-6..=7).contains(&e) {
        let q = (m >> 5) & 3;
        let r = (m >> 4) & 1;
        let sticky = (m & 0xF != 0) as i32;
        let lsb = (m >> 5) & 1;
        let round_up = r & (sticky | lsb);
        let sig_rounded = q + round_up;
        let carry = (sig_rounded >= 4) as i32;
        let mant_out = if carry != 0 { 0 } else { sig_rounded };
        let exp_out = if carry != 0 { e + 1 } else { e };
        if exp_out > 7 {
            return sign_f * f32::INFINITY;
        }
        return sign_f * (1.0 + mant_out as f32 * 0.25) * ldexpf(1.0, exp_out);
    }
    let k = match e {
        -7 => {
            if m <= 32 {
                2
            } else if m <= 95 {
                3
            } else {
                4
            }
        }
        -8 => {
            if m < 64 {
                1
            } else {
                2
            }
        }
        -9 => {
            if m == 0 {
                0
            } else {
                1
            }
        }
        _ => return 0.0,
    };
    sign_f * k as f32 * (1.0 / 256.0)
}

/// Deterministic E4M2 float -> FP6 grid float (`_e4m2_to_fp6`).
pub fn e4m2_to_fp6_float(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let ax = x.abs();
    if !ax.is_finite() || ax >= 32.0 {
        return sign * 28.0;
    }
    if ax <= 0.0546875 {
        return 0.0;
    }
    if (0.0625..=0.21875).contains(&ax) {
        let sub = if ax <= 0.078125 {
            0.0625
        } else if ax <= 0.15625 {
            0.125
        } else {
            0.1875
        };
        return sign * sub;
    }
    x
}

/// FP6 grid float -> 6-bit FP6 E3M2 code (`_fp6_value_to_code`).
pub fn fp6_value_to_code(v: f32) -> u8 {
    if v == 0.0 {
        return 0;
    }
    let s = (v < 0.0) as i32;
    let av = v.abs();
    const E_BITS: i32 = 3;
    const M_BITS: i32 = 2;
    const BIAS: i32 = 3;
    if av < 0.25 {
        // 2^(1-bias)
        let quantum = 0.0625f32;
        let mut mant = (av / quantum).round() as i32;
        mant = mant.clamp(0, (1 << M_BITS) - 1);
        return ((s << (E_BITS + M_BITS)) | mant) as u8;
    }
    let mut big_e = av.log2().floor() as i32;
    let base = ldexpf(1.0, big_e);
    let mut mant = ((av - base) / (base / 4.0)).round() as i32;
    if mant >= 4 {
        mant = 0;
        big_e += 1;
    }
    let biased = (big_e + BIAS).min((1 << E_BITS) - 1);
    let mant = mant.min((1 << M_BITS) - 1);
    ((s << (E_BITS + M_BITS)) | (biased << M_BITS) | mant) as u8
}

/// BF16 bits -> 6-bit FP6 E3M2 code (full HW pipeline).
pub fn bf16_bits_to_fp6_e3m2_code(bits: u16) -> u8 {
    fp6_value_to_code(e4m2_to_fp6_float(bf16_bits_to_e4m2_rne(bits)))
}

/// FP6 6-bit code -> 9-bit signed fixed point (mirrors FP6E3M2NearestFinder.scala).
pub fn fp6_to_fixed_point(val: u8) -> i32 {
    let val = val & 0x3F;
    let sign = (val >> 5) & 1;
    let exp = ((val >> 2) & 0x7) as i32;
    let mant = (val & 0x3) as i32;
    let is_zero = exp == 0 && mant == 0;
    let s_exp = if exp == 0 { -2 } else { exp - 3 };
    let implicit = if exp == 0 { 0 } else { 1 };
    let sig = (implicit << 2) | mant;
    let shift_amt = (s_exp + 2) & 0x7;
    let shifted = (sig << shift_amt) & 0x1FF;
    let magnitude = shifted & 0xFF;
    let signed_val = if sign != 0 { -magnitude } else { magnitude };
    if is_zero {
        0
    } else {
        signed_val
    }
}

/// Nearest LUT-entry finder; ties go to the lower index.
pub fn fp6e3m2_nearest_finder(in_code: u8, lut: &[u8]) -> usize {
    let fixed_in = fp6_to_fixed_point(in_code);
    let mut best = 0usize;
    let mut best_d = 0i32;
    for i in 0..16 {
        let f = fp6_to_fixed_point(lut.get(i).copied().unwrap_or(0));
        let d = (fixed_in - f).abs() & 0x1FF;
        if i == 0 || d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// BF16 -> E3M1 (RNE) -> E2M1 -> 4-bit FP4 code (`fp4_matmul_model.py::hw_bf16_to_e2m1`).
pub fn bf16_bits_to_fp4_e2m1_code(bf16: u16) -> u8 {
    let sign_bit = ((bf16 >> 15) & 1) as u8;
    let big_e = ((bf16 >> 7) & 0xFF) as i32;
    let m = (bf16 & 0x7F) as i32;
    if big_e == 0 {
        return 0;
    }
    if big_e == 255 {
        return (sign_bit << 3) | 0x7;
    }
    let e = big_e - 127;

    let e3m1_mag: f32 = if (-2..=3).contains(&e) {
        let s = 128 + m;
        let q = (s >> 6) & 1;
        let r = (s >> 5) & 1;
        let sticky = (s & 0x1F != 0) as i32;
        let round_up = r & (sticky | q);
        let sig2 = q + round_up;
        let mant_out = if sig2 >= 2 { 0 } else { sig2 };
        let exp_out = if sig2 >= 2 { e + 1 } else { e };
        if exp_out > 3 {
            return (sign_bit << 3) | 0x7;
        }
        (1.0 + mant_out as f32 * 0.5) * ldexpf(1.0, exp_out)
    } else if e == -3 {
        if m >= 64 {
            0.25
        } else {
            0.125
        }
    } else if e == -4 {
        if m > 0 {
            0.125
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mag_code: u8 = if e3m1_mag == 0.0 || e3m1_mag == 0.125 || e3m1_mag == 0.25 {
        return 0;
    } else if e3m1_mag == 0.375 || e3m1_mag == 0.5 {
        0x1
    } else if e3m1_mag == 0.75 || e3m1_mag == 1.0 {
        0x2
    } else if e3m1_mag == 1.5 {
        0x3
    } else if e3m1_mag == 2.0 {
        0x4
    } else if e3m1_mag == 3.0 {
        0x5
    } else if e3m1_mag == 4.0 {
        0x6
    } else {
        0x7 // 6.0 or larger
    };
    (sign_bit << 3) | mag_code
}

/// Unpack 16 x 6-bit FP6 E3M2 codes from 3 little-endian u32 words (96 bits).
/// This is the on-the-wire LUT format the MMIO path expects.
#[inline]
pub fn unpack_lut_96bit(dwords: [u32; 3], codes: &mut [u8; 16]) {
    let lo = dwords[0] as u64 | ((dwords[1] as u64) << 32);
    let hi = dwords[2] as u64;
    for (i, code) in codes.iter_mut().enumerate() {
        let bit = i * 6;
        *code = if bit + 6 <= 64 {
            ((lo >> bit) & 0x3F) as u8
        } else if bit >= 64 {
            ((hi >> (bit - 64)) & 0x3F) as u8
        } else {
            let merged = (lo >> bit) | (hi << (64 - bit));
            (merged & 0x3F) as u8
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldexp_matches_mul() {
        assert_eq!(ldexpf(1.0, 3), 8.0);
        assert_eq!(ldexpf(1.5, -2), 0.375);
        assert_eq!(ldexpf(1.0, -6), 1.0 / 64.0);
    }

    #[test]
    fn frexp_roundtrip() {
        for &x in &[1.0f32, 0.375, 12.0, 1.0 / 64.0, -3.5] {
            let (m, e) = frexpf(x);
            assert!(m.abs() >= 0.5 && m.abs() < 1.0, "m={m} out of range");
            assert!((ldexpf(m, e) - x).abs() < 1e-6, "x={x} m={m} e={e}");
        }
    }

    #[test]
    fn bf16_roundtrip_simple() {
        assert_eq!(bf16_to_f32(f32_to_bf16_rne(1.0)), 1.0);
        assert_eq!(bf16_to_f32(f32_to_bf16_rne(2.0)), 2.0);
        assert_eq!(f32_to_bf16_rne(0.0), 0);
        assert_eq!(f32_to_bf16_rne(-0.0), 0);
    }

    #[test]
    fn fp8_e4m3_decode_known() {
        // code for 1.0: sign 0, exp = bias(7) => 0b0111 << 3, mant 0 => 0x38
        assert_eq!(fp8_e4m3_decode(0x38), 1.0);
        // 2.0: exp 8 => 0b1000<<3 = 0x40
        assert_eq!(fp8_e4m3_decode(0x40), 2.0);
        // -1.0
        assert_eq!(fp8_e4m3_decode(0xB8), -1.0);
    }

    #[test]
    fn fp8_encode_decode_roundtrip() {
        for &v in &[0.0f32, 1.0, 2.0, 0.5, -1.5, 3.0, 4.0] {
            let c = fp8_e4m3_to_code(v);
            assert_eq!(fp8_e4m3_decode(c), v, "v={v} code={c:#x}");
        }
    }

    #[test]
    fn e8m0_scale_identity() {
        // code 127 => 2^0 == 1.0
        assert_eq!(fpe8m0_decode(127), 1.0);
        assert_eq!(fpe8m0_decode(128), 2.0);
        assert_eq!(fpe8m0_decode(126), 0.5);
    }

    // ---- Bit-exact cross-check against spike libgemmini (the numeric oracle) ----
    // Vectors in `super::super::spike_vectors` were emitted by compiling spike's
    // mx_fp_math.h natively. Every function below must reproduce them bit-for-bit.

    use crate::muon::mxgemmini::spike_vectors as sv;

    /// Compare a computed f32 to spike's expected bits; NaN compares by class
    /// (spike's nanf() and Rust's NAN may use different payloads).
    fn bits_eq(got: f32, exp_bits: u32) -> bool {
        let exp = f32::from_bits(exp_bits);
        if exp.is_nan() {
            got.is_nan()
        } else {
            got.to_bits() == exp_bits
        }
    }

    #[test]
    fn xcheck_fp8_decode_exhaustive() {
        for c in 0..256u32 {
            assert!(
                bits_eq(fp8_e4m3_decode(c as u8), sv::FP8_DECODE[c as usize]),
                "fp8 decode code={c}"
            );
        }
    }

    #[test]
    fn xcheck_e8m0_decode_exhaustive() {
        for c in 0..256u32 {
            assert!(
                bits_eq(fpe8m0_decode(c as u8), sv::E8M0_DECODE[c as usize]),
                "e8m0 decode code={c}"
            );
        }
    }

    #[test]
    fn xcheck_fp4_fp6_decode_exhaustive() {
        for c in 0..16u32 {
            assert!(bits_eq(fp4_e2m1_decode(c as u8), sv::FP4_DECODE[c as usize]), "fp4 {c}");
        }
        for c in 0..64u32 {
            assert!(bits_eq(fp6_e3m2_decode(c as u8), sv::FP6_DECODE[c as usize]), "fp6 {c}");
        }
    }

    #[test]
    fn xcheck_bf16_rne() {
        for &(xb, ob) in sv::BF16_RNE {
            let x = f32::from_bits(xb);
            assert!(bits_eq(bf16_round(x), ob), "bf16_round x={x}");
        }
        for &(xb, ob) in sv::BF16_BITS {
            assert_eq!(f32_to_bf16_rne(f32::from_bits(xb)), ob, "f32_to_bf16 x={}", f32::from_bits(xb));
        }
    }

    #[test]
    fn xcheck_mx_product_quantize_trunc() {
        for &(xb, ob) in sv::PROD_QUANT_43 {
            let x = f32::from_bits(xb);
            assert!(bits_eq(mx_product_quantize_trunc(x, 4, 3), ob), "prod_quant x={x}");
        }
    }

    #[test]
    fn xcheck_fp_quantize_rne_scalar() {
        for &(xb, ob) in sv::QUANT_RNE_44 {
            let x = f32::from_bits(xb);
            assert!(bits_eq(fp_quantize_rne_scalar(x, 4, 4), ob), "quant(4,4) x={x}");
        }
        for &(xb, ob) in sv::QUANT_RNE_46 {
            let x = f32::from_bits(xb);
            assert!(bits_eq(fp_quantize_rne_scalar(x, 4, 6), ob), "quant(4,6) x={x}");
        }
        for &(xb, ob) in sv::QUANT_RNE_87 {
            let x = f32::from_bits(xb);
            assert!(bits_eq(fp_quantize_rne_scalar(x, 8, 7), ob), "quant(8,7) x={x}");
        }
    }

    #[test]
    fn xcheck_bf16_accum_add() {
        for &(xb, yb, ob) in sv::ACCUM {
            let (x, y) = (f32::from_bits(xb), f32::from_bits(yb));
            assert!(bits_eq(bf16_accum_add(x, y), ob), "accum x={x} y={y}");
        }
    }

    #[test]
    fn xcheck_fp6_to_fixed_exhaustive() {
        for c in 0..64u32 {
            assert_eq!(
                fp6_to_fixed_point(c as u8),
                sv::FP6_TO_FIXED[c as usize],
                "fp6_to_fixed code={c}"
            );
        }
    }

    #[test]
    fn xcheck_fp6_nearest_finder_exhaustive() {
        for c in 0..64u32 {
            assert_eq!(
                fp6e3m2_nearest_finder(c as u8, &sv::NEAREST_LUT) as u8,
                sv::NEAREST_IDX[c as usize],
                "nearest_finder code={c}"
            );
        }
    }

    #[test]
    fn xcheck_e4m2_from_bf16() {
        for &(b, ob) in sv::E4M2_FROM_BF16 {
            assert!(bits_eq(bf16_bits_to_e4m2_rne(b), ob), "e4m2 bf16bits={b:#06x}");
        }
    }

    #[test]
    fn xcheck_fp6_code_from_bf16() {
        for &(b, code) in sv::FP6_CODE_FROM_BF16 {
            assert_eq!(bf16_bits_to_fp6_e3m2_code(b), code, "fp6 code bf16bits={b:#06x}");
        }
    }

    #[test]
    fn xcheck_fp4_code_from_bf16() {
        for &(b, code) in sv::FP4_CODE_FROM_BF16 {
            assert_eq!(bf16_bits_to_fp4_e2m1_code(b), code, "fp4 code bf16bits={b:#06x}");
        }
    }

    #[test]
    fn xcheck_fp8_code_from_f32() {
        for &(xb, code) in sv::FP8_CODE_FROM_F32 {
            let x = f32::from_bits(xb);
            assert_eq!(fp8_e4m3_to_code(x), code, "fp8 code x={x}");
        }
    }

    #[test]
    fn simple_fp8_product_accumulate() {
        // 1.0 * 1.0 accumulated 4 times with bf16 accum == 4.0
        let mut acc = 0.0f32;
        for _ in 0..4 {
            let p = mx_product_quantize_trunc(1.0 * 1.0, 4, 3);
            acc = bf16_accum_add(acc, p);
        }
        assert_eq!(acc, 4.0);
    }
}
