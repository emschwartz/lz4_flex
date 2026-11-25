//! The block decompression algorithm (unsafe fast variant, used without `safe-decode`).
//!
//! # Safety design
//!
//! All `unsafe` code lives in the [`prim`] module: a handful of fixed-size or bounded copies,
//! each with a written precondition stated in terms of slice indices. The decoder itself is safe
//! code over `(slice, index)` pairs. It never forms an out-of-bounds pointer. Before each call
//! into [`prim`], it establishes the precondition with an integer comparison of the form
//! `n <= len - pos`, which cannot overflow because `pos <= len` is an invariant of [`Decoder`].
//!
//! The fast path in [`Decoder::try_fast_sequence`] checks a single input and output margin up
//! front. That one check covers every copy the sequence makes, so the hot loop needs no
//! per-access checks.
//!
//! The output is `&mut [MaybeUninit<u8>]`, so it can be a `Vec`'s spare capacity. The decoder
//! keeps `out[..op]` initialized: it only writes at `op` and beyond, and a byte becomes part of
//! `out[..op]` only when a copy whose source was initialized writes it.
//!
//! The Kani proofs in `verification` check every primitive against its precondition, check that
//! each decoder step preserves the invariants, and run the whole decoder on all small inputs.
use crate::block::{DecompressError, MINMATCH};
use crate::sink::SliceSink;
#[cfg(feature = "alloc")]
#[allow(unused_imports)]
use alloc::vec::Vec;
use core::mem::MaybeUninit;

/// Input bytes the fast path may touch after the token: a 16-byte wild literal copy, the
/// 2-byte match offset and the next token.
const FAST_INPUT_MARGIN: usize = 16 + 2 + 1;
/// Output bytes the fast path may touch: a 16-byte wild literal copy and an 18-byte wild match
/// copy.
const FAST_OUTPUT_MARGIN: usize = 16 + 18;
/// Extra output margin in dictionary mode, where a dictionary copy of up to 17 bytes can come
/// before the 18-byte wild match copy.
const FAST_DICT_MARGIN: usize = 17;

/// The only `unsafe` code in the decoder. Every function states its precondition in terms of
/// slice lengths, and the Kani proofs in `verification` check that each precondition is
/// sufficient.
mod prim {
    use core::mem::MaybeUninit;
    use core::ptr;

    /// Copies `N` bytes from `src[s..s + N]` to `dst[d..d + N]`.
    ///
    /// # Safety
    /// `s + N <= src.len()` and `d + N <= dst.len()`.
    #[inline(always)]
    pub(super) unsafe fn copy_fixed<const N: usize>(
        src: &[u8],
        s: usize,
        dst: &mut [MaybeUninit<u8>],
        d: usize,
    ) {
        debug_assert!(s <= src.len() && N <= src.len() - s);
        debug_assert!(d <= dst.len() && N <= dst.len() - d);
        // SAFETY: both ranges are in bounds by the precondition. `src` is a shared borrow and
        // `dst` a unique one, so they cannot overlap.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr().add(s), dst.as_mut_ptr().add(d).cast::<u8>(), N)
        }
    }

    /// Copies `N` bytes from `out[from..from + N]` to `out[to..to + N]`. The ranges may overlap.
    ///
    /// # Safety
    /// `from + N <= out.len()` and `to + N <= out.len()`.
    #[inline(always)]
    pub(super) unsafe fn copy_within_fixed<const N: usize>(
        out: &mut [MaybeUninit<u8>],
        from: usize,
        to: usize,
    ) {
        debug_assert!(from <= out.len() && N <= out.len() - from);
        debug_assert!(to <= out.len() && N <= out.len() - to);
        let base = out.as_mut_ptr().cast::<u8>();
        // SAFETY: both ranges are in bounds by the precondition. `ptr::copy` allows overlap.
        unsafe { ptr::copy(base.add(from), base.add(to), N) }
    }

    /// Reads a little-endian `u16` from `src[s..s + 2]`.
    ///
    /// # Safety
    /// `s + 2 <= src.len()`.
    #[inline(always)]
    pub(super) unsafe fn read_u16_le(src: &[u8], s: usize) -> u16 {
        debug_assert!(s <= src.len() && 2 <= src.len() - s);
        // SAFETY: in bounds by the precondition. `read_unaligned` has no alignment requirement.
        u16::from_le_bytes(unsafe { ptr::read_unaligned(src.as_ptr().add(s).cast::<[u8; 2]>()) })
    }

    /// Copies `len` bytes from `src[s..s + len]` to `dst[d..d + len]`, optimized for short copies.
    ///
    /// # Safety
    /// `len >= 1`, `s + len <= src.len()` and `d + len <= dst.len()`.
    #[inline(always)]
    pub(super) unsafe fn copy_var(
        src: &[u8],
        s: usize,
        dst: &mut [MaybeUninit<u8>],
        d: usize,
        len: usize,
    ) {
        debug_assert!(len >= 1);
        debug_assert!(s <= src.len() && len <= src.len() - s);
        debug_assert!(d <= dst.len() && len <= dst.len() - d);
        // SAFETY: `slice_copy` reads and writes exactly `[0, len)` from its pointers and needs
        // `len >= 1`. Both ranges are in bounds and cannot overlap (shared vs unique borrow).
        unsafe {
            crate::fastcpy_unsafe::slice_copy(
                src.as_ptr().add(s),
                dst.as_mut_ptr().add(d).cast::<u8>(),
                len,
            )
        }
    }

    /// Copies `len` bytes from `out[from..]` to `out[to..]` one byte at a time, front to back, so
    /// a match may read bytes that the same copy just wrote (`to - from < len`).
    ///
    /// # Safety
    /// `from < to` and `to + len <= out.len()`.
    #[inline]
    #[cfg_attr(feature = "nightly", optimize(size))] // to avoid loop unrolling
    pub(super) unsafe fn copy_overlapping(
        out: &mut [MaybeUninit<u8>],
        from: usize,
        to: usize,
        len: usize,
    ) {
        debug_assert!(from < to);
        debug_assert!(to <= out.len() && len <= out.len() - to);
        let base = out.as_mut_ptr().cast::<u8>();
        let mut i = 0;
        // Manually unrolled by two: left alone, the compiler unrolls and vectorizes this with a
        // lot of branches, which does not pay off since long overlapping copies are rare.
        while i + 1 < len {
            // SAFETY: `from + i + 1 < to + i + 1 < to + len <= out.len()`.
            unsafe {
                ptr::copy(base.add(from + i), base.add(to + i), 1);
                ptr::copy(base.add(from + i + 1), base.add(to + i + 1), 1);
            }
            i += 2;
        }
        if i < len {
            // SAFETY: `from + i < to + i < to + len <= out.len()`.
            unsafe { ptr::copy(base.add(from + i), base.add(to + i), 1) }
        }
    }

    /// Copies `len` bytes from `out[from..]` to `out[to..]` in 16-byte chunks. Writes up to 15
    /// bytes past `to + len`, and reads up to 15 bytes past `from + len`.
    ///
    /// # Safety
    /// `len >= 1`, `from + len + 15 <= to` and `to + len + 15 <= out.len()`.
    #[inline]
    pub(super) unsafe fn wild_copy_within_16(
        out: &mut [MaybeUninit<u8>],
        from: usize,
        to: usize,
        len: usize,
    ) {
        debug_assert!(len >= 1);
        debug_assert!(from <= to && len + 15 <= to - from);
        debug_assert!(to <= out.len() && len + 15 <= out.len() - to);
        let base = out.as_mut_ptr().cast::<u8>();
        // Note: if the compiler auto-vectorizes this it'll hurt performance!
        // It's not the case for 16 bytes stepsize, but for 8 bytes.
        let mut i = 0;
        loop {
            // SAFETY: `i < len`, so `to + i + 16 <= to + len + 15 <= out.len()`, and
            // `from + i + 16 <= to + i`, so the two 16-byte ranges do not overlap.
            unsafe { ptr::copy_nonoverlapping(base.add(from + i), base.add(to + i), 16) }
            i += 16;
            if i >= len {
                break;
            }
        }
    }

    /// Copies `src` into `dst[d..d + src.len()]`. Safe: panics if out of bounds.
    #[inline]
    pub(super) fn write_slice(dst: &mut [MaybeUninit<u8>], d: usize, src: &[u8]) {
        for (o, i) in dst[d..d + src.len()].iter_mut().zip(src) {
            *o = MaybeUninit::new(*i);
        }
    }
}

/// Read an integer.
///
/// In LZ4, we encode small integers in a way that we can have an arbitrary number of bytes. In
/// particular, we add the bytes repeatedly until we hit a non-0xFF byte. When we do, we add
/// this byte to our sum and terminate the loop.
///
/// # Example
///
/// ```notest
///     255, 255, 255, 4, 2, 3, 4, 6, 7
/// ```
///
/// is encoded to _255 + 255 + 255 + 4 = 769_. The bytes after the first 4 is ignored, because
/// 4 is the first non-0xFF byte.
#[inline]
pub(super) fn read_integer(input: &[u8], input_pos: &mut usize) -> Result<usize, DecompressError> {
    let mut n: usize = 0;
    loop {
        let extra = *input
            .get(*input_pos)
            .ok_or(DecompressError::ExpectedAnotherByte)?;
        *input_pos += 1;
        // A wrapped sum is harmless: every length is checked against the remaining buffer
        // before use.
        n = n.wrapping_add(extra as usize);
        if extra != 0xFF {
            return Ok(n);
        }
    }
}

const FIT_TOKEN_MASK_LITERAL: u8 = 0b00001111;
const FIT_TOKEN_MASK_MATCH: u8 = 0b11110000;

#[test]
fn check_token() {
    assert!(!does_token_fit(15));
    assert!(does_token_fit(14));
    assert!(does_token_fit(114));
    assert!(!does_token_fit(0b11110000));
    assert!(does_token_fit(0b10110000));
}

/// The token consists of two parts, the literal length (upper 4 bits) and match_length (lower 4
/// bits) if the literal length and match_length are both below 15, we don't need to read additional
/// data, so the token does fit the metadata in a single u8.
#[inline]
fn does_token_fit(token: u8) -> bool {
    !((token & FIT_TOKEN_MASK_LITERAL) == FIT_TOKEN_MASK_LITERAL
        || (token & FIT_TOKEN_MASK_MATCH) == FIT_TOKEN_MASK_MATCH)
}

/// Decoder state.
///
/// # Invariants
/// - `ip <= input.len()`
/// - `op <= out.len()`
/// - `out[..op]` is initialized
struct Decoder<'a, const USE_DICT: bool> {
    input: &'a [u8],
    ip: usize,
    out: &'a mut [MaybeUninit<u8>],
    op: usize,
    ext_dict: &'a [u8],
}

impl<const USE_DICT: bool> Decoder<'_, USE_DICT> {
    #[inline(always)]
    fn read_u8(&mut self) -> Result<u8, DecompressError> {
        let b = *self
            .input
            .get(self.ip)
            .ok_or(DecompressError::ExpectedAnotherByte)?;
        self.ip += 1;
        Ok(b)
    }

    /// Copies the part of a match that lies in the external dictionary to the output. Requires
    /// `self.op < offset <= self.op + ext_dict.len()` and `match_length <= out.len() - op`.
    /// Returns the number of bytes copied.
    #[inline]
    fn copy_from_dict(&mut self, offset: usize, match_length: usize) -> usize {
        debug_assert!(self.op < offset && offset <= self.op + self.ext_dict.len());
        let dict_offset = self.ext_dict.len() + self.op - offset;
        // The match may continue past the end of the dictionary into the output.
        let dict_match_length = match_length.min(self.ext_dict.len() - dict_offset);
        let src = &self.ext_dict[dict_offset..dict_offset + dict_match_length];
        prim::write_slice(self.out, self.op, src);
        self.op += dict_match_length;
        dict_match_length
    }

    /// Decodes one whole sequence on the fast path, if `token` needs no extra length bytes and
    /// both buffers have room for wild copies. Returns `Ok(false)`, consuming nothing, when the
    /// fast path does not apply.
    #[inline(always)]
    fn try_fast_sequence(&mut self, token: u8) -> Result<bool, DecompressError> {
        let out_margin = FAST_OUTPUT_MARGIN + if USE_DICT { FAST_DICT_MARGIN } else { 0 };
        if !(does_token_fit(token)
            && self.input.len() - self.ip >= FAST_INPUT_MARGIN
            && self.out.len() - self.op > out_margin)
        {
            return Ok(false);
        }
        // From here on, `ip + 19 <= input.len()` and `op + out_margin < out.len()` for the
        // values of `ip` and `op` at entry.
        let literal_length = (token >> 4) as usize; // <= 14, since the token fits
        let mut match_length = MINMATCH + (token & 0xF) as usize; // <= 18

        // SAFETY: `ip + 16 <= input.len()` and `op + 16 <= out.len()` by the margins.
        unsafe { prim::copy_fixed::<16>(self.input, self.ip, self.out, self.op) };
        self.ip += literal_length;
        self.op += literal_length;

        // SAFETY: `ip` moved by at most 14 of the 19-byte margin, so `ip + 2 <= input.len()`.
        let offset = unsafe { prim::read_u16_le(self.input, self.ip) } as usize;
        self.ip += 2;
        // The margin still leaves at least one byte for the next token.
        debug_assert!(self.ip < self.input.len());

        if offset == 0 {
            return Err(DecompressError::OffsetZero);
        }
        if offset > self.op + self.ext_dict.len() {
            return Err(DecompressError::OffsetOutOfBounds);
        }
        if USE_DICT && offset > self.op {
            // `op + 14 + 18 <= out.len()` by the margin, so the dictionary copy fits.
            let copied = self.copy_from_dict(offset, match_length);
            if copied == match_length {
                return Ok(true);
            }
            // The match continues from the start of the output.
            match_length -= copied;
        }

        let start = self.op - offset;
        if offset >= match_length {
            // SAFETY: `start < op`, and `op` moved by at most 14 (literal) + 17 (dictionary) of the
            // `16 + 18 + 17` margin, so `start + 18 < op + 18 <= out.len()`.
            unsafe { prim::copy_within_fixed::<18>(self.out, start, self.op) };
        } else {
            // SAFETY: `start < op` since `offset >= 1`, and `op + match_length <= out.len()` by the
            // margin.
            unsafe { prim::copy_overlapping(self.out, start, self.op, match_length) };
        }
        self.op += match_length;
        Ok(true)
    }

    /// Copies a match of `match_length` bytes starting at `start` in the output. Requires
    /// `start < op` and `match_length <= out.len() - op`.
    #[inline]
    fn duplicate(&mut self, start: usize, match_length: usize) {
        debug_assert!(start < self.op);
        debug_assert!(match_length <= self.out.len() - self.op);
        // The wild copy can write up to 15 bytes past the match, and needs source and
        // destination 16 bytes apart. Use the byte-wise copy when either does not hold.
        if self.op - start < match_length + 15 || self.out.len() - self.op < match_length + 15 {
            // SAFETY: `start < op` and `op + match_length <= out.len()`.
            unsafe { prim::copy_overlapping(self.out, start, self.op, match_length) };
        } else {
            // SAFETY: `match_length >= 1` (every caller passes a positive length), and both
            // distances were checked just above.
            unsafe { prim::wild_copy_within_16(self.out, start, self.op, match_length) };
        }
        self.op += match_length;
    }

    /// Decodes one sequence. Returns `Ok(true)` if more sequences follow and `Ok(false)` at the
    /// end of the input.
    #[inline(always)]
    fn step(&mut self) -> Result<bool, DecompressError> {
        // The token holds two 4-bit fields: the literal length (high) and the match
        // length (low).
        let token = self.read_u8()?;

        // Most sequences fit in a single token and are far from both ends of the buffers.
        if self.try_fast_sequence(token)? {
            return Ok(true);
        }

        // Literals. A length of 15 means more length bytes follow.
        let mut literal_length = (token >> 4) as usize;
        if literal_length != 0 {
            if literal_length == 15 {
                literal_length =
                    literal_length.wrapping_add(read_integer(self.input, &mut self.ip)?);
            }
            if literal_length > self.input.len() - self.ip {
                return Err(DecompressError::LiteralOutOfBounds);
            }
            if literal_length > self.out.len() - self.op {
                return Err(DecompressError::OutputTooSmall {
                    expected: self.op.saturating_add(literal_length),
                    actual: self.out.len(),
                });
            }
            // SAFETY: `literal_length >= 1`, and both ranges were checked just above.
            unsafe { prim::copy_var(self.input, self.ip, self.out, self.op, literal_length) };
            self.ip += literal_length;
            self.op += literal_length;
        }

        // The last sequence holds only literals, so the input ends here.
        if self.ip >= self.input.len() {
            return Ok(false);
        }

        if self.input.len() - self.ip < 2 {
            return Err(DecompressError::ExpectedAnotherByte);
        }
        // SAFETY: `ip + 2 <= input.len()`, checked just above.
        let offset = unsafe { prim::read_u16_le(self.input, self.ip) } as usize;
        self.ip += 2;
        if offset == 0 {
            return Err(DecompressError::OffsetZero);
        }

        // A match is at least MINMATCH bytes. A low field of 15 means more length bytes
        // follow.
        let mut match_length = MINMATCH + (token & 0xF) as usize;
        if match_length == MINMATCH + 15 {
            match_length = match_length.wrapping_add(read_integer(self.input, &mut self.ip)?);
        }

        if offset > self.op + self.ext_dict.len() {
            return Err(DecompressError::OffsetOutOfBounds);
        }
        if match_length > self.out.len() - self.op {
            return Err(DecompressError::OutputTooSmall {
                expected: self.op.saturating_add(match_length),
                actual: self.out.len(),
            });
        }

        if USE_DICT && offset > self.op {
            let copied = self.copy_from_dict(offset, match_length);
            if copied == match_length {
                if self.ip >= self.input.len() {
                    return Err(DecompressError::ExpectedAnotherByte);
                }
                return Ok(true);
            }
            // The match continues from the start of the output.
            match_length -= copied;
        }

        // `offset <= op` here: either it was in bounds of the output, or the dictionary copy
        // advanced `op` to exactly `offset`. A match length below MINMATCH only happens after
        // a partial dictionary copy, which leaves at least one byte.
        let start = self.op - offset;
        self.duplicate(start, match_length);

        if self.ip >= self.input.len() {
            return Err(DecompressError::ExpectedAnotherByte);
        }
        Ok(true)
    }

    #[inline(always)]
    fn run(&mut self) -> Result<(), DecompressError> {
        while self.step()? {}
        Ok(())
    }
}

/// Decompresses `input` into `out[pos..]`, using `out[..pos]` and then `ext_dict` as the
/// lookback window. Returns the end position in `out`.
///
/// Writes to `out` only copies of bytes from `input`, `ext_dict` and `out` itself. So if `out`
/// was fully initialized on entry, every write is an initialized byte. On success,
/// `out[..returned position]` is initialized.
///
/// # Panics
/// Panics if `pos > out.len()`.
#[inline]
fn decompress_core<const USE_DICT: bool>(
    input: &[u8],
    out: &mut [MaybeUninit<u8>],
    pos: usize,
    ext_dict: &[u8],
) -> Result<usize, DecompressError> {
    assert!(pos <= out.len());
    // Prevent segfault for empty input
    if input.is_empty() {
        return Err(DecompressError::ExpectedAnotherByte);
    }
    let ext_dict = if USE_DICT {
        ext_dict
    } else {
        // ensure optimizer knows ext_dict length is 0 if !USE_DICT
        debug_assert!(ext_dict.is_empty());
        &[]
    };
    let mut decoder = Decoder::<USE_DICT> {
        input,
        ip: 0,
        out,
        op: pos,
        ext_dict,
    };
    decoder.run()?;
    Ok(decoder.op)
}

/// Decompress all bytes of `input` into `output`.
///
/// Returns the number of bytes written (decompressed) into `output`.
#[inline]
pub(crate) fn decompress_internal<const USE_DICT: bool>(
    input: &[u8],
    output: &mut SliceSink<'_>,
    ext_dict: &[u8],
) -> Result<usize, DecompressError> {
    let (buf, pos) = output.buf_and_pos_mut();
    let start = *pos;
    // SAFETY: `buf` is fully initialized, and `decompress_core` writes only copies of
    // initialized bytes (from `input`, `ext_dict` and `buf` itself), so `buf` stays a valid
    // `[u8]`. `MaybeUninit<u8>` has the same layout as `u8`.
    let out = unsafe { &mut *(buf as *mut [u8] as *mut [MaybeUninit<u8>]) };
    let end = decompress_core::<USE_DICT>(input, out, start, ext_dict)?;
    *pos = end;
    Ok(end - start)
}

/// Decompresses into the spare capacity of a fresh `Vec` without zero-filling it first.
#[cfg(feature = "alloc")]
#[inline]
fn decompress_to_vec<const USE_DICT: bool>(
    input: &[u8],
    min_uncompressed_size: usize,
    ext_dict: &[u8],
) -> Result<Vec<u8>, DecompressError> {
    let mut vec = Vec::with_capacity(min_uncompressed_size);
    let len = decompress_core::<USE_DICT>(input, vec.spare_capacity_mut(), 0, ext_dict)?;
    // SAFETY: `decompress_core` returns a position `len <= capacity` such that `[..len]` is
    // initialized.
    unsafe { vec.set_len(len) };
    Ok(vec)
}

/// Decompress all bytes of `input` into `output`.
/// `output` should be preallocated with a size of of the uncompressed data.
#[inline]
pub fn decompress_into(input: &[u8], output: &mut [u8]) -> Result<usize, DecompressError> {
    decompress_internal::<false>(input, &mut SliceSink::new(output, 0), b"")
}

/// Decompress all bytes of `input` into `output`.
///
/// Returns the number of bytes written (decompressed) into `output`.
#[inline]
pub fn decompress_into_with_dict(
    input: &[u8],
    output: &mut [u8],
    ext_dict: &[u8],
) -> Result<usize, DecompressError> {
    decompress_internal::<true>(input, &mut SliceSink::new(output, 0), ext_dict)
}

/// Decompress all bytes of `input` into a new vec.
/// The passed parameter `min_uncompressed_size` needs to be equal or larger than the uncompressed size.
///
/// # Panics
/// May panic if the parameter `min_uncompressed_size` is smaller than the
/// uncompressed data.

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[inline]
pub fn decompress_with_dict(
    input: &[u8],
    min_uncompressed_size: usize,
    ext_dict: &[u8],
) -> Result<Vec<u8>, DecompressError> {
    decompress_to_vec::<true>(input, min_uncompressed_size, ext_dict)
}

/// Decompress all bytes of `input` into a new vec. The first 4 bytes are the uncompressed size in
/// little endian. Can be used in conjunction with `compress_prepend_size`
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[inline]
pub fn decompress_size_prepended(input: &[u8]) -> Result<Vec<u8>, DecompressError> {
    let (uncompressed_size, input) = super::uncompressed_size(input)?;
    decompress(input, uncompressed_size)
}

/// Decompress all bytes of `input` into a new vec.
/// The passed parameter `min_uncompressed_size` needs to be equal or larger than the uncompressed size.
///
/// # Panics
/// May panic if the parameter `min_uncompressed_size` is smaller than the
/// uncompressed data.
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[inline]
pub fn decompress(input: &[u8], min_uncompressed_size: usize) -> Result<Vec<u8>, DecompressError> {
    decompress_to_vec::<false>(input, min_uncompressed_size, b"")
}

/// Decompress all bytes of `input` into a new vec. The first 4 bytes are the uncompressed size in
/// little endian. Can be used in conjunction with `compress_prepend_size_with_dict`
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[inline]
pub fn decompress_size_prepended_with_dict(
    input: &[u8],
    ext_dict: &[u8],
) -> Result<Vec<u8>, DecompressError> {
    let (uncompressed_size, input) = super::uncompressed_size(input)?;
    decompress_with_dict(input, uncompressed_size, ext_dict)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn all_literal() {
        assert_eq!(decompress(&[0x30, b'a', b'4', b'9'], 3).unwrap(), b"a49");
    }

    #[test]
    fn incomplete_input() {
        assert!(matches!(
            decompress(&[], 255),
            Err(DecompressError::ExpectedAnotherByte)
        ));
        assert!(matches!(
            // incomplete literal len
            decompress(&[0xF0], 255),
            Err(DecompressError::ExpectedAnotherByte)
        ));
        assert!(matches!(
            // incomplete match offset
            decompress(&[0x0F, 0], 255),
            Err(DecompressError::ExpectedAnotherByte)
        ));
        assert!(matches!(
            // incomplete match len
            decompress(&[0x0F, 1, 0], 255),
            Err(DecompressError::ExpectedAnotherByte)
        ));
    }

    // this error test is only valid in safe-decode.
    #[test]
    fn offset_oob() {
        // incomplete literal
        assert!(matches!(
            decompress(&[0x40, b'a', 1, 0], 4),
            Err(DecompressError::LiteralOutOfBounds)
        ));
        // literal too large for output
        assert!(matches!(
            decompress(&[0x20, b'a', b'a', 1, 0], 1),
            Err(DecompressError::OutputTooSmall {
                expected: 2,
                actual: 1
            })
        ));
        // match too large for output
        assert!(matches!(
            decompress(&[0x10, b'a', 1, 0], 4),
            Err(DecompressError::OutputTooSmall {
                expected: 5,
                actual: 4
            })
        ));

        // out-of-bounds hot-loop
        assert!(matches!(
            decompress(
                &[0x0E, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                256
            ),
            Err(DecompressError::OffsetOutOfBounds)
        ));
        // out-of-bounds for dict
        assert!(matches!(
            decompress_with_dict(
                &[0x0E, 255, 0, 0x70, 0, 0, 0, 0, 0, 0, 0],
                256,
                &[0_u8; 250]
            ),
            Err(DecompressError::OffsetOutOfBounds)
        ));
        // out-of-bounds non-hot-loop overlapping
        assert!(matches!(
            decompress(&[0x0F, 1, 0, 1, 0x70, 0, 0, 0, 0, 0, 0, 0], 256),
            Err(DecompressError::OffsetOutOfBounds)
        ));
        // out-of-bounds non-hot-loop non-overlapping
        assert!(matches!(
            decompress(&[0x40, 0, 0, 0, 0, 255, 0, 0x70, 0, 0, 0, 0, 0, 0, 0], 256),
            Err(DecompressError::OffsetOutOfBounds)
        ));
    }

    #[test]
    fn offset_0() {
        assert!(matches!(
            decompress(&[0x0E, 0, 0, 0x70, 0, 0, 0, 0, 0, 0, 0], 256),
            Err(DecompressError::OffsetZero)
        ));
    }
}

/// Kani proofs. Run with `cargo kani --no-default-features -Z uninit-checks`.
///
/// The step proofs are the inductive argument. Every state that satisfies the [`Decoder`]
/// invariants satisfies them again after one [`Decoder::step`], with no undefined behavior on
/// the way. `decompress_core` starts in such a state, so the loop is safe for inputs with any
/// number of sequences. Buffer sizes are bounded (`IN`, `OUT`, `DICT`), but large enough that
/// every margin check in the decoder can go both ways.
#[cfg(kani)]
mod verification {
    use super::*;

    const IN: usize = 24;
    const OUT: usize = 64;
    const DICT: usize = 20;

    fn any_prefix<const N: usize>(buf: &[u8; N]) -> &[u8] {
        let len: usize = kani::any();
        kani::assume(len <= N);
        &buf[..len]
    }

    /// An output buffer whose first `op` bytes are initialized and the rest uninitialized.
    fn any_out<const N: usize>(op: usize) -> [MaybeUninit<u8>; N] {
        let init: [u8; N] = kani::any();
        let mut out = [MaybeUninit::uninit(); N];
        for i in 0..op {
            out[i] = MaybeUninit::new(init[i]);
        }
        out
    }

    fn assert_init_prefix(out: &[MaybeUninit<u8>], len: usize) {
        for byte in &out[..len] {
            // With `-Z uninit-checks`, Kani fails this read if the byte is uninitialized.
            let _ = unsafe { byte.assume_init() };
        }
    }

    fn check_step<const USE_DICT: bool>() {
        let input_buf: [u8; IN] = kani::any();
        let input = any_prefix(&input_buf);
        let dict_buf: [u8; DICT] = kani::any();
        let ext_dict: &[u8] = if USE_DICT { any_prefix(&dict_buf) } else { &[] };
        let ip: usize = kani::any();
        kani::assume(ip <= input.len());
        let out_len: usize = kani::any();
        kani::assume(out_len <= OUT);
        let op: usize = kani::any();
        kani::assume(op <= out_len);
        let mut out_buf = any_out::<OUT>(op);

        let mut decoder = Decoder::<USE_DICT> {
            input,
            ip,
            out: &mut out_buf[..out_len],
            op,
            ext_dict,
        };
        let _ = decoder.step();

        assert!(decoder.ip <= decoder.input.len());
        assert!(decoder.op <= decoder.out.len());
        assert_init_prefix(decoder.out, decoder.op);
    }

    #[kani::proof]
    #[kani::unwind(66)]
    fn step_preserves_invariants() {
        check_step::<false>();
    }

    #[kani::proof]
    #[kani::unwind(66)]
    fn step_preserves_invariants_with_dict() {
        check_step::<true>();
    }

    /// End to end on all small inputs, starting from an entirely uninitialized output.
    #[kani::proof]
    #[kani::unwind(18)]
    fn decompress_core_small_inputs() {
        let input_buf: [u8; 8] = kani::any();
        let input = any_prefix(&input_buf);
        let dict_buf: [u8; 4] = kani::any();
        let ext_dict = any_prefix(&dict_buf);
        let mut out = [MaybeUninit::<u8>::uninit(); 16];
        if let Ok(end) = decompress_core::<true>(input, &mut out, 0, ext_dict) {
            assert!(end <= out.len());
            assert_init_prefix(&out, end);
        }
    }

    // The primitives, each checked against exactly its stated precondition.

    #[kani::proof]
    fn prim_copy_fixed() {
        let src_buf: [u8; 40] = kani::any();
        let src = any_prefix(&src_buf);
        let mut dst_buf = [MaybeUninit::<u8>::uninit(); 40];
        let dst_len: usize = kani::any();
        kani::assume(dst_len <= 40);
        let dst = &mut dst_buf[..dst_len];
        let (s, d): (usize, usize) = (kani::any(), kani::any());
        kani::assume(s <= src.len() && 16 <= src.len() - s);
        kani::assume(d <= dst.len() && 16 <= dst.len() - d);
        unsafe { prim::copy_fixed::<16>(src, s, dst, d) };
        let i: usize = kani::any();
        kani::assume(i < 16);
        assert!(unsafe { dst[d + i].assume_init() } == src[s + i]);
    }

    #[kani::proof]
    fn prim_copy_within_fixed() {
        let len: usize = kani::any();
        kani::assume(len <= 40);
        let mut buf = any_out::<40>(len);
        let out = &mut buf[..len];
        let (from, to): (usize, usize) = (kani::any(), kani::any());
        kani::assume(from <= out.len() && 18 <= out.len() - from);
        kani::assume(to <= out.len() && 18 <= out.len() - to);
        unsafe { prim::copy_within_fixed::<18>(out, from, to) };
    }

    #[kani::proof]
    fn prim_read_u16_le() {
        let src_buf: [u8; 8] = kani::any();
        let src = any_prefix(&src_buf);
        let s: usize = kani::any();
        kani::assume(s <= src.len() && 2 <= src.len() - s);
        let v = unsafe { prim::read_u16_le(src, s) };
        assert!(v == u16::from_le_bytes([src[s], src[s + 1]]));
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn prim_copy_var() {
        let src_buf: [u8; 40] = kani::any();
        let src = any_prefix(&src_buf);
        let mut dst_buf = [MaybeUninit::<u8>::uninit(); 40];
        let dst_len: usize = kani::any();
        kani::assume(dst_len <= 40);
        let dst = &mut dst_buf[..dst_len];
        let (s, d, len): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
        kani::assume(len >= 1);
        kani::assume(s <= src.len() && len <= src.len() - s);
        kani::assume(d <= dst.len() && len <= dst.len() - d);
        unsafe { prim::copy_var(src, s, dst, d, len) };
        let i: usize = kani::any();
        kani::assume(i < len);
        assert!(unsafe { dst[d + i].assume_init() } == src[s + i]);
    }

    #[kani::proof]
    #[kani::unwind(22)]
    fn prim_copy_overlapping() {
        let mut buf = any_out::<40>(40);
        let (from, to, len): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
        kani::assume(from < to && to <= buf.len() && len <= buf.len() - to);
        unsafe { prim::copy_overlapping(&mut buf, from, to, len) };
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn prim_wild_copy_within_16() {
        let mut buf = any_out::<64>(64);
        let (from, to, len): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
        kani::assume(len >= 1);
        kani::assume(from <= to && len + 15 <= to - from);
        kani::assume(to <= buf.len() && len + 15 <= buf.len() - to);
        unsafe { prim::wild_copy_within_16(&mut buf, from, to, len) };
    }
}
