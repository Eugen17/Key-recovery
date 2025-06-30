/**********************************************************************
 * present_integral_attack.cu
 *
 * 8-round integral key-recovery attack against PRESENT
 * ----------------------------------------------------
 * – needs tables_PRESENT.inc (pBox8_* and sBox tables) in include path
 * – re-uses your high-performance S/P-layer tables
 *********************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <cuda.h>
#include "tables_PRESENT.inc"

#define NSAMPLES    (1u << 12)          /* 4096 plaintexts in one coset   */
#define THREADS     256                 /* generic block size             */
#define SBOX_INV_SZ 16

/* convenient aliases matching your typedefs ------------------------ */
typedef uint8_t  bit8;
typedef uint64_t bit64;

/* ------------------------------ constant memory ------------------- */
__constant__ uint8_t  d_sbox_inv[SBOX_INV_SZ];
__constant__ uint8_t  d_div_exp[4];          /* (4,1,2,1) */

/* ------------------------------------------------------------------ */
/*                      DEVICE-SIDE 8-round encryption                */
/* ------------------------------------------------------------------ */
__global__ void PRESENT_encrypt8(const bit64* __restrict__ pt,
                                 bit64* __restrict__ ct,
                                 const bit64* __restrict__ rk,
                                 const bit64* __restrict__ p0,
                                 const bit64* __restrict__ p1,
                                 const bit64* __restrict__ p2,
                                 const bit64* __restrict__ p3,
                                 const bit64* __restrict__ p4,
                                 const bit64* __restrict__ p5,
                                 const bit64* __restrict__ p6,
                                 const bit64* __restrict__ p7,
                                 const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bit64 s = pt[idx];

    /* 8 rounds = 7 full rounds + final AddRoundKey ----------------- */
#pragma unroll
    for (int r = 0; r < 7; ++r) {
        s ^= rk[r];
        s  = p0[s & 0xFF] |
             p1[(s >>  8) & 0xFF] |
             p2[(s >> 16) & 0xFF] |
             p3[(s >> 24) & 0xFF] |
             p4[(s >> 32) & 0xFF] |
             p5[(s >> 40) & 0xFF] |
             p6[(s >> 48) & 0xFF] |
             p7[(s >> 56)];
    }
    ct[idx] = s ^ rk[7];
}

/* ------------------------------------------------------------------ */
/*                    PHASE-1 : test 4-bit nibble                     */
/* ------------------------------------------------------------------ */
__device__ __forceinline__
uint8_t undo_half_round(bit64 ct, uint8_t k4)
{
    return d_sbox_inv[(ct & 0xF) ^ k4];
}

extern "C" __global__
void test_nibble_kernel(const bit64* __restrict__ ct,
                        const int n_ct,          // 4096
                        uint32_t* survivors)     // length ≥ 1
{
    const uint8_t k4 = threadIdx.x;              // 0 … 15
    uint32_t cnt[4] = {0,0,0,0};

    /* ---------- scan all ciphertexts ----------------------------- */
    for (int i = 0; i < n_ct; ++i) {
        uint8_t v = undo_half_round(ct[i], k4);  // 4-bit nibble @ 6th rnd
        cnt[0] +=  v       & 1u;
        cnt[1] += (v >> 1) & 1u;
        cnt[2] += (v >> 2) & 1u;
        cnt[3] +=  v >> 3;
    }

    /* ---------- 2-adic divisibility test ------------------------- */
    bool ok = true;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        uint32_t mask = (1u << d_div_exp[b]) - 1u;
        if (cnt[b] & mask) { ok = false; break; }
    }

    /* ---------- write survivors ---------------------------------- */
    if (ok) survivors[atomicAdd(&survivors[0],1)+1] = k4;
}

/* ------------------------------------------------------------------ */
/*                    PHASE-2 : test 20-bit word                      */
/* ------------------------------------------------------------------ */
extern "C" __global__
void test_word_kernel(const bit64* __restrict__ ct,
                      const int n_ct,            // 4096 here
                      const uint8_t  k4,         // nibble from phase-1
                      uint32_t* survivors)       // len ≥ 1
{
    const uint32_t kid = blockIdx.x * blockDim.x + threadIdx.x; // 0..65535
    if (kid >= 65536u) return;

    /* split 16-bit word into 4 nibbles entering S-boxes 8-4-0-12 ----- */
    uint8_t kA =  kid        & 0xF;
    uint8_t kB = (kid >> 4 ) & 0xF;
    uint8_t kC = (kid >> 8 ) & 0xF;
    uint8_t kD =  kid >> 12;

    uint32_t xor_sum = 0;

    /* ---------- iterate over every ciphertext once ---------------- */
    for (int i = 0; i < n_ct; ++i) {
        bit64 v = ct[i];

        /* Undo the two half-rounds relevant for the integral.
           Only the four right-most nibbles matter. */
        uint8_t s0 = d_sbox_inv[((v      ) & 0xF) ^ k4];
        uint8_t s1 = d_sbox_inv[((v>>  4) & 0xF) ^ kA];
        uint8_t s2 = d_sbox_inv[((v>> 16) & 0xF) ^ kB];
        uint8_t s3 = d_sbox_inv[((v>> 32) & 0xF) ^ kC];
        uint8_t s4 = d_sbox_inv[((v>> 48) & 0xF) ^ kD];

        uint8_t bit0 = (s4 ^ s3 ^ s2 ^ s1 ^ s0) & 1u;
        xor_sum ^= bit0;
    }

    if (xor_sum == 0)
        survivors[atomicAdd(&survivors[0],1)+1] =
            (kid << 4) | k4;              /* concatenate to 20-bit word */
}

/* ------------------------------------------------------------------ */
/*                    HOST-SIDE HELPER FUNCTIONS                      */
/* ------------------------------------------------------------------ */
static const uint8_t h_sbox_inv[SBOX_INV_SZ] =
/*  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F */
 { 5,14,15, 8,12, 1, 2,13,11, 4, 6, 3, 0, 7, 9,10 };

static const uint8_t h_div_exp[4] = {4,1,2,1};

void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {   fprintf(stderr,"CUDA error %s:%d: %s\n",
                file,line,cudaGetErrorString(code)); exit(code); }
}
#define cuCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*--------------------------------------------------------------------*/
int main()
{
    cuCHECK( cudaSetDevice(0) );

    /* --------------------------------------------------------------- *
     *            1.  build plaintext coset 0x…fff0 (4096)             *
     * --------------------------------------------------------------- */
    bit64 *h_pt  = (bit64*)malloc(NSAMPLES*sizeof(bit64));
    bit64 *h_ct  = (bit64*)malloc(NSAMPLES*sizeof(bit64));

    srand((unsigned)time(NULL));
    bit64 high52 = ((bit64)rand() << 20) & ((1ULL<<52)-1);
    high52 <<= 12;                         /* shift into top bits */

    for (uint32_t i = 0; i < NSAMPLES; ++i)
        h_pt[i] = high52 | ((bit64)i << 4);  /* …xxx yyy0 – low nibble 0 */

    /* --------------------------------------------------------------- *
     *            2.  copy constant data to the GPU                    *
     * --------------------------------------------------------------- */
    cuCHECK( cudaMemcpyToSymbol(d_sbox_inv, h_sbox_inv,
                                SBOX_INV_SZ, 0, cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpyToSymbol(d_div_exp,  h_div_exp,
                                4,           0, cudaMemcpyHostToDevice) );

    /* copy p-layer lookup tables supplied by tables_PRESENT.inc  */
    bit64 *d_p0,*d_p1,*d_p2,*d_p3,*d_p4,*d_p5,*d_p6,*d_p7;
    cuCHECK( cudaMalloc((void**)&d_p0, 256*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_p1, 256*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_p2, 256*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_p3, 256*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_p4, 256*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_p5, 256*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_p6, 256*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_p7, 256*sizeof(bit64)) );

    cuCHECK( cudaMemcpy(d_p0, pBox8_0, 256*sizeof(bit64), cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpy(d_p1, pBox8_1, 256*sizeof(bit64), cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpy(d_p2, pBox8_2, 256*sizeof(bit64), cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpy(d_p3, pBox8_3, 256*sizeof(bit64), cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpy(d_p4, pBox8_4, 256*sizeof(bit64), cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpy(d_p5, pBox8_5, 256*sizeof(bit64), cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpy(d_p6, pBox8_6, 256*sizeof(bit64), cudaMemcpyHostToDevice) );
    cuCHECK( cudaMemcpy(d_p7, pBox8_7, 256*sizeof(bit64), cudaMemcpyHostToDevice) );

    /* --------------------------------------------------------------- *
     *            3.  choose random master key, make 8-round keys      *
     * --------------------------------------------------------------- */
    bit64 master_key[2];
    master_key[0] = (((bit64)rand()<<32) | rand());   /* low 64 bits   */
    master_key[1] = 0;                                /* we ignore top */

    bit64 round_key[32];
    key_schedule(master_key);              /* uses your host-side func  */
    for(int i=0;i<32;++i) round_key[i]=::round_key[i];

    bit64 *d_rk;
    cuCHECK( cudaMalloc((void**)&d_rk, 8*sizeof(bit64)) );
    cuCHECK( cudaMemcpy(d_rk, round_key, 8*sizeof(bit64),
                        cudaMemcpyHostToDevice) );

    /* --------------------------------------------------------------- *
     *            4.  encrypt all plaintexts through 8 rounds          *
     * --------------------------------------------------------------- */
    bit64 *d_pt,*d_ct;
    cuCHECK( cudaMalloc((void**)&d_pt, NSAMPLES*sizeof(bit64)) );
    cuCHECK( cudaMalloc((void**)&d_ct, NSAMPLES*sizeof(bit64)) );
    cuCHECK( cudaMemcpy(d_pt, h_pt, NSAMPLES*sizeof(bit64),
                        cudaMemcpyHostToDevice) );

    dim3 blocks_enc( (NSAMPLES+THREADS-1)/THREADS );
    PRESENT_encrypt8<<<blocks_enc, THREADS>>>(d_pt,d_ct,d_rk,
                                              d_p0,d_p1,d_p2,d_p3,
                                              d_p4,d_p5,d_p6,d_p7,
                                              NSAMPLES);
    cuCHECK( cudaDeviceSynchronize() );
    cuCHECK( cudaGetLastError() );

    /* --------------------------------------------------------------- *
     *            5-A.  Phase 1 – find surviving k4 nibbles            *
     * --------------------------------------------------------------- */
    uint32_t *d_surv1;
    cuCHECK( cudaMalloc((void**)&d_surv1, (16+1)*sizeof(uint32_t)) );
    cuCHECK( cudaMemset(d_surv1, 0, (16+1)*sizeof(uint32_t)) );

    test_nibble_kernel<<<1,16>>>(d_ct,NSAMPLES,d_surv1);
    cuCHECK( cudaDeviceSynchronize() ); cuCHECK( cudaGetLastError() );

    uint32_t h_surv1[17];
    cuCHECK( cudaMemcpy(h_surv1, d_surv1, (16+1)*sizeof(uint32_t),
                        cudaMemcpyDeviceToHost) );
    uint32_t n_k4 = h_surv1[0];
    printf("[+] Phase-1 survivors (%u):", n_k4);
    for (uint32_t i=1;i<=n_k4;++i) printf(" %u", h_surv1[i]);
    puts("");

    /* --------------------------------------------------------------- *
     *            5-B.  Phase 2 – expand to 20-bit words               *
     * --------------------------------------------------------------- */
    uint32_t *d_surv2;
    cuCHECK( cudaMalloc((void**)&d_surv2, (65536+1)*sizeof(uint32_t)) );

    for (uint32_t s=1; s<=n_k4; ++s)
    {
        uint8_t k4 = (uint8_t)h_surv1[s];
        cuCHECK( cudaMemset(d_surv2, 0, (65536+1)*sizeof(uint32_t)) );

        dim3 blocks_word((65536+THREADS-1)/THREADS);
        test_word_kernel<<<blocks_word, THREADS>>>(d_ct,NSAMPLES,k4,d_surv2);
        cuCHECK( cudaDeviceSynchronize() ); cuCHECK( cudaGetLastError() );

        uint32_t h_surv2_cnt;
        cuCHECK( cudaMemcpy(&h_surv2_cnt, d_surv2, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost) );

        if (h_surv2_cnt==0) continue;

        uint32_t *h_surv2 =
            (uint32_t*)malloc((h_surv2_cnt+1)*sizeof(uint32_t));
        cuCHECK( cudaMemcpy(h_surv2, d_surv2,
                            (h_surv2_cnt+1)*sizeof(uint32_t),
                            cudaMemcpyDeviceToHost) );

        printf("    k4=%2u  →  %u surviving 20-bit keys\n",
               k4, h_surv2_cnt);
        for(uint32_t j=1;j<=h_surv2_cnt;++j)
            printf("        0x%05x\n", h_surv2[j]);

        free(h_surv2);
    }

    /* --------------------------------------------------------------- *
     *            6.  Verification (did we keep the true key?)         *
     * --------------------------------------------------------------- */
    uint32_t true_subkey = ((round_key[7] & 0xF) << 16) |
                           ((round_key[7] >> 60) & 0xF) |
                           ((round_key[7] >> 44) & 0xF) << 4 |
                           ((round_key[7] >> 28) & 0xF) << 8 |
                           ((round_key[7] >> 12) & 0xF) << 12;

    /* Clean-up & finish -------------------------------------------- */
    cuCHECK( cudaFree(d_pt)); cuCHECK( cudaFree(d_ct));
    cuCHECK( cudaFree(d_rk)); cuCHECK( cudaFree(d_surv1));
    cuCHECK( cudaFree(d_surv2));
    cuCHECK( cudaFree(d_p0)); cuCHECK( cudaFree(d_p1)); cuCHECK( cudaFree(d_p2));
    cuCHECK( cudaFree(d_p3)); cuCHECK( cudaFree(d_p4)); cuCHECK( cudaFree(d_p5));
    cuCHECK( cudaFree(d_p6)); cuCHECK( cudaFree(d_p7));

    free(h_pt); free(h_ct);

    return 0;
}
