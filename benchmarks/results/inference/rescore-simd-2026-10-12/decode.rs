/// The value a binary16 code stands for.
pub(crate) fn decode(code: u16) -> f32 {
    let sign = u32::from(code & 0x8000) << 16;
    let exponent = u32::from((code >> 10) & 0x1f);
    let mantissa = u32::from(code & 0x03ff);
    if exponent == 0 {
        let magnitude = mantissa as f32 / 16_777_216.0;
        return if sign == 0 { magnitude } else { -magnitude };
    }
    let bits = if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}
