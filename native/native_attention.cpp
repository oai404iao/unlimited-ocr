// Build both baseline-safe libraries with: python build_native.py
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

__attribute__((target("avx2,fma")))
static float dot_avx2(const float* q, const float* k, int64_t d,
                 int64_t qs, int64_t ks) {
    int64_t i = 0;
    float sum = 0;
    if (qs == 1 && ks == 1) {
        __m256 acc = _mm256_setzero_ps();
        for (; i <= d - 8; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(q+i), _mm256_loadu_ps(k+i), acc);
        float lanes[8];
        _mm256_storeu_ps(lanes, acc);
        for (float lane : lanes) sum += lane;
    }
    for (; i < d; ++i) sum += q[i*qs] * k[i*ks];
    return sum;
}

__attribute__((target("avx512f,fma")))
static float dot_avx512(const float* q, const float* k, int64_t d,
                        int64_t qs, int64_t ks) {
    int64_t i = 0;
    float sum = 0;
    if (qs == 1 && ks == 1) {
        __m512 acc = _mm512_setzero_ps();
        for (; i <= d - 16; i += 16)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q+i), _mm512_loadu_ps(k+i), acc);
        float lanes[16];
        _mm512_storeu_ps(lanes, acc);
        for (float lane : lanes) sum += lane;
    }
    for (; i < d; ++i) sum += q[i*qs] * k[i*ks];
    return sum;
}

__attribute__((target("avx2,fma")))
static void accumulate_avx2(float* dst, const float* value, int64_t d,
                            int64_t stride, float w) {
    int64_t i=0;
    if (stride==1) {
        const __m256 ww=_mm256_set1_ps(w);
        for (; i<=d-8; i+=8)
            _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(ww,_mm256_loadu_ps(value+i),_mm256_loadu_ps(dst+i)));
    }
    for (; i<d; ++i) dst[i]+=w*value[i*stride];
}

__attribute__((target("avx512f,fma")))
static void accumulate_avx512(float* dst, const float* value, int64_t d,
                              int64_t stride, float w) {
    int64_t i=0;
    if (stride==1) {
        const __m512 ww=_mm512_set1_ps(w);
        for (; i<=d-16; i+=16)
            _mm512_storeu_ps(dst+i, _mm512_fmadd_ps(ww,_mm512_loadu_ps(value+i),_mm512_loadu_ps(dst+i)));
    }
    for (; i<d; ++i) dst[i]+=w*value[i*stride];
}

// Metadata: B,H,L,D,Hkv,S,Dv,K, then q/k/v element strides (four each).
// Routing arrays are contiguous; q/k/v can be arbitrary positive-stride views.
template <auto dot, auto accumulate>
static int decode_attention(
    const float* q, const float* k, const float* v, const int64_t* topi,
    const float* bias, const bool* valid, const int64_t* positions,
    float* out, const int64_t* m, int64_t block_size, int64_t window, float scale) {
    try {
        const int64_t B=m[0], H=m[1], L=m[2], D=m[3], HK=m[4], S=m[5], DV=m[6], K=m[7];
        const int64_t *qs=m+8, *ks=m+12, *vs=m+16;
        std::vector<int64_t> slots(K), tokens;
        std::vector<float> priors, weights;
        for (int64_t b=0; b<B; ++b) for (int64_t r=0; r<L; ++r) {
            const int64_t route=(b*L+r)*K, pos=positions[b*L+r];
            // Clamp to the actual cache before arithmetic can overflow at the
            // int64 limits. Positions outside the cache are legal causal queries.
            const int64_t local_end=pos>=S-1 ? S : std::max<int64_t>(0,pos+1);
            const int64_t local_start=window==0 ? local_end :
                (pos>=window-1 ? std::min(S,pos-(window-1)) : 0);
            tokens.clear(); priors.clear();
            for (int64_t j=0; j<K; ++j) slots[j]=j;
            std::sort(slots.begin(), slots.end(), [&](int64_t a, int64_t c) {
                return topi[route+a] == topi[route+c] ? a>c : topi[route+a]<topi[route+c];
            });
            for (int64_t j=0; j<K; ++j) {
                const int64_t slot=slots[j], block=topi[route+slot];
                // Sort duplicate slots newest first, including invalid slots.
                if (j && block == topi[route+slots[j-1]]) continue;
                if (!valid[route+slot] || !(bias[route+slot] > -5.e8f)) continue;
                // Out-of-cache blocks cannot contribute. Check before multiplication.
                if (block<0 || block>S/block_size) continue;
                int64_t start=block*block_size;
                int64_t end=std::min({start+std::min(block_size,S-start), local_end, local_start});
                for (int64_t t=start; t<end; ++t) {
                    tokens.push_back(t); priors.push_back(bias[route+slot]);
                }
            }
            for (int64_t t=local_start; t<local_end; ++t) {
                tokens.push_back(t); priors.push_back(0);
            }
            weights.resize(tokens.size());
            for (int64_t h=0; h<H; ++h) {
                float* dst=out+((b*H+h)*L+r)*DV;
                if (tokens.empty()) {
                    std::fill(dst,dst+DV,std::numeric_limits<float>::quiet_NaN());
                    continue;
                }
                const int64_t kh=h/(H/HK);
                const float* qr=q+b*qs[0]+h*qs[1]+r*qs[2];
                const float* kr=k+b*ks[0]+kh*ks[1];
                const float* vr=v+b*vs[0]+kh*vs[1];
                float maximum=-std::numeric_limits<float>::infinity();
                for (size_t t=0; t<tokens.size(); ++t) {
                    weights[t]=dot(qr,kr+tokens[t]*ks[2],D,qs[3],ks[3])*scale+priors[t];
                    maximum=std::max(maximum,weights[t]);
                }
                float sum=0;
                for (float& w : weights) { w=std::exp(w-maximum); sum+=w; }
                std::fill(dst,dst+DV,0.f);
                for (size_t t=0; t<tokens.size(); ++t) {
                    const float w=weights[t]/sum;
                    const float* value=vr+tokens[t]*vs[2];
                    accumulate(dst,value,DV,vs[3],w);
                }
            }
        }
        return 0;
    } catch (...) { return -1; }
}

// Callers select only variants supported by their CPU and OS.
extern "C" int decode_attention_avx2(
    const float* q, const float* k, const float* v, const int64_t* topi,
    const float* bias, const bool* valid, const int64_t* positions,
    float* out, const int64_t* m, int64_t block_size, int64_t window, float scale) {
    return decode_attention<dot_avx2, accumulate_avx2>(
        q,k,v,topi,bias,valid,positions,out,m,block_size,window,scale);
}

extern "C" int decode_attention_avx512(
    const float* q, const float* k, const float* v, const int64_t* topi,
    const float* bias, const bool* valid, const int64_t* positions,
    float* out, const int64_t* m, int64_t block_size, int64_t window, float scale) {
    return decode_attention<dot_avx512, accumulate_avx512>(
        q,k,v,topi,bias,valid,positions,out,m,block_size,window,scale);
}
