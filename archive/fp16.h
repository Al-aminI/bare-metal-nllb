/*  fp16.h -- FP16 conversion utilities
 *
 *  IEEE 754 half-precision (FP16) conversion for KV cache compression.
 *  Reduces memory footprint by 50% with negligible quality impact.
 */

#ifndef FP16_H
#define FP16_H

#include <stdint.h>
#include <string.h>

/* ─── FP32 to FP16 conversion ───────────────────────────────────────────── */

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    
    /* Handle special cases */
    if (exp <= 0) {
        /* Underflow to zero or denormal */
        if (exp < -10) return (uint16_t)sign;  /* Too small, flush to zero */
        
        /* Denormal */
        mant |= 0x800000;  /* Add implicit 1 */
        int shift = 1 - exp;
        mant >>= shift;
        return (uint16_t)(sign | (mant >> 13));
    }
    
    if (exp >= 31) {
        /* Overflow to infinity */
        return (uint16_t)(sign | 0x7C00);
    }
    
    /* Normal number */
    mant += 0x00001000;  /* Round to nearest even */
    if (mant & 0x00800000) {
        /* Rounding caused mantissa overflow */
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);  /* Overflow */
    }
    
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* ─── FP16 to FP32 conversion ───────────────────────────────────────────── */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            uint32_t bits = sign;
            float result;
            memcpy(&result, &bits, sizeof(float));
            return result;
        }
        
        /* Denormal - normalize it */
        exp = 1;
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
    } else if (exp == 31) {
        /* Infinity or NaN */
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float result;
        memcpy(&result, &bits, sizeof(float));
        return result;
    }
    
    /* Normal number */
    exp = exp - 15 + 127;
    uint32_t bits = sign | ((uint32_t)exp << 23) | (mant << 13);
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

/* ─── Bulk conversion utilities ─────────────────────────────────────────── */

static inline void fp32_array_to_fp16(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

static inline void fp16_array_to_fp32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

#endif /* FP16_H */
