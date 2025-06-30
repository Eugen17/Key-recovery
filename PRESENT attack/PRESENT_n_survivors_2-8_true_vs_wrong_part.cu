#ifdef _WIN32
#include <Windows.h>
#endif
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include "tables_PRESENT.inc"
#define Exhaustive 65536
#define THREAD 1024
#define BLOCK 512
bit64 round_key[32] = { 0 };
__constant__ uint8_t d_div_exp[4] = {4,1,2,1};
// ► key schedule for PRESENT-80 (80-bit master-key) -------------------------
void key_schedule(bit64 key[2]) {
	bit64 keylow, keyhigh, temp_0;
	bit8 sBox4[16] = { 0xc,0x5,0x6,0xb,0x9,0x0,0xa,0xd,0x3,0xe,0xf,0x8,0x4,0x7,0x1,0x2 };  // PRESENT's S-box
	keyhigh = key[0];
	keylow = key[1];
	round_key[0] = keyhigh;
	for (int i = 1; i < 32; i++) {
		temp_0 = keylow & 0xffff;
		keylow = (keyhigh >> 3) & 0xffff;
		keyhigh = (keyhigh << 61) ^ (keyhigh >> 19) ^ (temp_0 << 45);
		temp_0 = sBox4[keyhigh >> 60];
		keyhigh = (keyhigh & 0xfffffffffffffff) | (temp_0 << 60);
		keyhigh ^= (i >> 1);
		keylow ^= (i << 15);
		round_key[i] = keyhigh;
	}
	// for (int i = 0; i < 7; i++) {
	// 	printf("Round key[%2d] = 0x%016llx\n", i, (unsigned long long)round_key[i]);
	// }
}



// 8-round PRESENT encryption of a 2¹²-element coset 0x0..0FFF0
// -----------------------------------------------------------------------------
#define COS_SZ           4096          // 2¹² plaintexts
#define R8_THREAD       256            // threads / block
#define R8_BLOCKS    ((COS_SZ+R8_THREAD-1)/R8_THREAD)

// ► host helper ---------------------------------------------------------------
#include <random>

static std::mt19937_64  g_rng; 
inline uint64_t rand64() { return g_rng(); }
/* produce a random base that fulfils the pattern …0000?????0000      */

/* choose seed → call once at start of main() */
inline void init_rng(uint64_t seed) { g_rng.seed(seed); }


static inline uint64_t random_coset_base()
{
    /* keep nibbles 0 and 4 zero, nibbles 1-3 will be filled in the kernel */
    uint64_t rnd = rand64();
    rnd &= 0xFFFFFFFFFFF00000ULL;      // clear low 20 bits
    return rnd;                        // nibbles 15‥5 random, 4‥0 = 0
}


// ─────────────────────────────────────────────────────────────────────────────
//  four 4-bit nibbles  →  16-bit word per ciphertext   (8 kB for 4096 texts)
// ─────────────────────────────────────────────────────────────────────────────
typedef uint16_t CtWord;            // convenience
#define CT_BUF_SZ   (COS_SZ * sizeof(CtWord))


// ►  9 round-keys are needed (AddRoundKey after round 8) ---------------
__global__ void PRESENT8_coset_kernel(uint64_t       baseP,
                                      const uint64_t rk[32],  // first 9 used
                                      const uint64_t * __restrict__ p0,
                                      const uint64_t * __restrict__ p1,
                                      const uint64_t * __restrict__ p2,
                                      const uint64_t * __restrict__ p3,
                                      const uint64_t * __restrict__ p4,
                                      const uint64_t * __restrict__ p5,
                                      const uint64_t * __restrict__ p6,
                                      const uint64_t * __restrict__ p7,
                                       CtWord*        __restrict__ ct_out)
{
    // --- load tables & round-keys to shared memory ---------------------------
    __shared__ uint64_t P0[256],P1[256],P2[256],P3[256],
                        P4[256],P5[256],P6[256],P7[256],
                        RK[9];
    if (threadIdx.x < 256) {
        P0[threadIdx.x] = p0[threadIdx.x];
        P1[threadIdx.x] = p1[threadIdx.x];
        P2[threadIdx.x] = p2[threadIdx.x];
        P3[threadIdx.x] = p3[threadIdx.x];
        P4[threadIdx.x] = p4[threadIdx.x];
        P5[threadIdx.x] = p5[threadIdx.x];
        P6[threadIdx.x] = p6[threadIdx.x];
        P7[threadIdx.x] = p7[threadIdx.x];
    }
    if (threadIdx.x < 9) RK[threadIdx.x] = rk[threadIdx.x];
    __syncthreads();

    // --- index inside the coset ---------------------------------------------
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= COS_SZ) return;

    /* embed the varying 12 bits into nibbles 3‥1, leave nibble0 & 4 zero   */
    uint64_t state = baseP ^ ((uint64_t)gid << 4);


	// --- initial key-whitening (AddRoundKey) -------------------------------
	state ^= RK[0];  // K₀
    // --- 7 full rounds -------------------------------------------------------
    #pragma unroll
    for (int r = 1; r < 9; r++) {
		// if (threadIdx.x == 161) {
		// 	printf("r: %llx\n", (unsigned long long)r);
		// }
        
        state  = P0[state       & 0xFF]  |  P1[(state>> 8) & 0xFF] |
                 P2[(state>>16) & 0xFF]  |  P3[(state>>24) & 0xFF] |
                 P4[(state>>32) & 0xFF]  |  P5[(state>>40) & 0xFF] |
                 P6[(state>>48) & 0xFF]  |  P7[(state>>56)      ];
		state ^= RK[r];
    }

    auto g4 = [](uint64_t k, int p0,int p1,int p2,int p3) -> uint8_t {
        return  ( (k >> p0) & 1u)       | (((k >> p1) & 1u) << 1) |
               (((k >> p2) & 1u) << 2) | (((k >> p3) & 1u) << 3);
    };

	    /* 1, 5, 9, 13 ← nibble positions                     */
    uint8_t n1 = g4(state,  0, 16, 32, 48);  
    uint8_t n5 = g4(state, 4,20,36,52);      
    uint8_t n9 = g4(state, 8,24,40,56);      
    uint8_t n13= g4(state, 12,28,44,60);     

    /* pack into 16 bits:  n13|n9|n5|n1   (high … low) */
    ct_out[gid] = (n13 << 12) | (n9 << 8) | (n5 << 4) | n1;
	
}

// -----------------------------------------------------------------------------
// host wrapper – fills     h_ct_coset[4096]  with the 8-round ciphertexts
// -----------------------------------------------------------------------------
void encrypt_8round_coset(CtWord h_ct_coset[COS_SZ],
                          uint64_t key80_out[2], uint64_t* baseP_out)          // ← returns key
{
    /* -------- choose reproducible 80-bit master-key ----------------------- */
    uint64_t key80[2] = { rand64(), rand64() & 0xFFFF };  // 80 bits total, first 64 are high
    key80_out[0] = key80[0]; key80_out[1] = key80[1];

    key_schedule(key80);                                  // fills round_key[]

    /* -------- pick random coset base (still reproducible) ----------------- */
    uint64_t baseP = random_coset_base();
    printf("Coset base: 0x%016llx\n", (unsigned long long)baseP);
    *baseP_out = baseP;
    /* -------- device allocations ------------------------------------------ */
    CtWord *d_ct = nullptr;
    uint64_t *d_rk = nullptr;
    cudaMalloc(&d_ct, CT_BUF_SZ);  
    cudaMalloc(&d_rk, 32     * sizeof(uint64_t));
    cudaMemcpy(d_rk, round_key, 32*sizeof(uint64_t), cudaMemcpyHostToDevice);

    /* -------- copy pre-computed p-layer tables ---------------------------- */
    uint64_t *d_p0,*d_p1,*d_p2,*d_p3,*d_p4,*d_p5,*d_p6,*d_p7;
    auto copy_tbl = [](uint64_t*& dst, const uint64_t* src) {
        cudaMalloc(&dst, 256*sizeof(uint64_t));
        cudaMemcpy(dst, src, 256*sizeof(uint64_t), cudaMemcpyHostToDevice);
    };
    copy_tbl(d_p0,pBox8_0); copy_tbl(d_p1,pBox8_1); copy_tbl(d_p2,pBox8_2);
    copy_tbl(d_p3,pBox8_3); copy_tbl(d_p4,pBox8_4); copy_tbl(d_p5,pBox8_5);
    copy_tbl(d_p6,pBox8_6); copy_tbl(d_p7,pBox8_7);

    /* -------- launch kernel ------------------------------------------------ */
    PRESENT8_coset_kernel<<<R8_BLOCKS,R8_THREAD>>>(baseP, d_rk,
                                                   d_p0,d_p1,d_p2,d_p3,
                                                   d_p4,d_p5,d_p6,d_p7,
                                                   d_ct);
    cudaDeviceSynchronize();

    /* -------- fetch results + clean-up ------------------------------------ */
    cudaMemcpy(h_ct_coset, d_ct, COS_SZ*sizeof(uint16_t), cudaMemcpyDeviceToHost);

    cudaFree(d_ct);  cudaFree(d_rk);
    cudaFree(d_p0);  cudaFree(d_p1);  cudaFree(d_p2);  cudaFree(d_p3);
    cudaFree(d_p4);  cudaFree(d_p5);  cudaFree(d_p6);  cudaFree(d_p7);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6-round integral test  – 2²⁰ candidate partial keys
// ─────────────────────────────────────────────────────────────────────────────
__constant__ uint8_t d_invS[16];        // inverse S-box in constant mem


__global__ void integral_test_kernel(const CtWord* __restrict__ ct_compact,
                                     uint32_t*      __restrict__ survivors)
{
    /* key layout:
         bits 0‥15 → kA,kB,kC,kD   (cipher-nibbles 1,5,9,13)
         bits 16‥19→ kE            (second InvSbox layer)                */
    const uint32_t kid = blockIdx.x * blockDim.x + threadIdx.x;
    if (kid >= (1u << 20)) return;

    /* ----- split 20-bit candidate into five nibbles (registers) -------- */
    uint8_t kA =  kid        & 0xF;
    uint8_t kB = (kid >>  4) & 0xF;
    uint8_t kC = (kid >>  8) & 0xF;
    uint8_t kD = (kid >> 12) & 0xF;
    uint8_t kE =  kid >> 16;

    /* ----- counters for the 4 bits of ‘val’ ---------------------------- */
    uint32_t cnt[4] = {0,0,0,0};

    for (int i = 0; i < COS_SZ; i++)
    {
                 // 16-bit packed nibbles

        /* unpack ciphertext nibbles 1,5,9,13 */

        CtWord ct = ct_compact[i];
        uint8_t c1  =  ct        & 0xF;
        uint8_t c5  = (ct >>  4) & 0xF;
        uint8_t c9  = (ct >>  8) & 0xF;
        uint8_t c13 =  ct >> 12;

        /* first InvSbox layer */
        uint8_t s1  = d_invS[c1  ^ kA];
        uint8_t s5  = d_invS[c5  ^ kB];
        uint8_t s9  = d_invS[c9  ^ kC];
        uint8_t s13 = d_invS[c13 ^ kD];

        /* LSB concatenation → second InvSbox */
        uint8_t lsb_concat = ((s13 & 1u) << 3) |
                             ((s9  & 1u) << 2) |
                             ((s5  & 1u) << 1) |
                             ( s1  & 1u);
        uint8_t val = d_invS[lsb_concat ^ kE];   // 4-bit value
        // if (kid==0x531da && i < 10) {
        //     printf("kid: %d, i: %d, ct: 0x%04x, lsb_concat: 0x%x, val: 0x%x\n", kid, i, ct, lsb_concat, val);
        // }


        cnt[0] +=  val       & 1u;
        cnt[1] += (val >> 1) & 1u;
        cnt[2] += (val >> 2) & 1u;
        cnt[3] +=  val >> 3;
        
    }


    /* ----- divisibility test ------------------------------------------- */
    bool ok = true;
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        uint32_t mask = (1u << d_div_exp[b]) - 1u;  // 2^v  − 1
        if (cnt[b] & mask) { ok = false; break; }   // not divisible
    }

    if (ok)
        survivors[atomicAdd(&survivors[0], 1) + 1] = kid;
}


std::vector<uint32_t> recover_partial_keys(const CtWord  ct_compact[CT_BUF_SZ])
{
    /* ---- copy compact ciphertexts to device ----------------------------- */
    CtWord  *d_ct;   cudaMalloc(&d_ct, CT_BUF_SZ);
    cudaMemcpy(d_ct, ct_compact, CT_BUF_SZ, cudaMemcpyHostToDevice);

    /* ---- survivor counter + list  (max 2²⁰) ----------------------------- */
    uint32_t *d_surv; cudaMalloc(&d_surv, ((1<<20)+1)*sizeof(uint32_t));
    cudaMemset(d_surv, 0, sizeof(uint32_t));

    /* ---- upload inverse S-box once -------------------------------------- */
    static const uint8_t invS[16] =
        {0x5,0xe,0xf,0x8,0xc,0x1,0x2,0xd,0xb,0x4,0x6,0x3,0x0,0x7,0x9,0xa};
    cudaMemcpyToSymbol(d_invS, invS, 16);

    /* ---- launch: 1024 threads / block, enough blocks for 2²⁰ keys ------- */
    dim3 blk(1024), grd((1<<20)+1023>>10);
    integral_test_kernel<<<grd,blk>>>(d_ct, d_surv);
    cudaDeviceSynchronize();

    /* ---- copy survivors back -------------------------------------------- */
    uint32_t cnt;
    cudaMemcpy(&cnt, d_surv, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::vector<uint32_t> list(cnt);
    if (cnt)
        cudaMemcpy(list.data(), d_surv+1, cnt*sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

    cudaFree(d_ct);  cudaFree(d_surv);
    return list;          // 20-bit candidates that satisfy the integral
}




/* prints the 5 nibbles and returns the packed 20-bit word */
uint32_t true_partial_key()
{
    auto g4 = [](uint64_t k, int p0,int p1,int p2,int p3) -> uint8_t {
        return  ( (k >> p0) & 1u)       | (((k >> p1) & 1u) << 1) |
               (((k >> p2) & 1u) << 2) | (((k >> p3) & 1u) << 3);
    };

    uint64_t rk8 = round_key[8];
    uint64_t rk7 = round_key[7];

    uint8_t kA = g4(rk8,  0, 16, 32, 48);        // nibble 1 in state
    uint8_t kB = g4(rk8, 4,20,36,52);        // nibble 5
    uint8_t kC = g4(rk8, 8,24,40,56);        // nibble 9
    uint8_t kD = g4(rk8, 12,28,44,60);        // nibble 13
    uint8_t kE = g4(rk7, 0, 16, 32, 48);        // second-layer nibble


    uint32_t packed = (kE << 16) | (kD << 12)
                    | (kC <<  8) | (kB <<  4) | kA;

    return packed;
}




/* run one full attack, return (#survivors, #correct_last4, avg_survivors_per_wrong_last4, seed) */
struct RunResult {
    uint32_t survivors;
    uint32_t correct_last4;
    double avg_survivors_per_wrong_last4;
    uint64_t seed;
};

RunResult run_once(uint64_t seed, std::ofstream& log)
{
    init_rng(seed);
    /* ---------- generate coset & run integral exactly as before ---------- */
    CtWord  h_ct[COS_SZ];
    uint64_t key80[2], base_coset;

    encrypt_8round_coset(h_ct, key80, &base_coset);
    std::vector<uint32_t> subs = recover_partial_keys(h_ct);

    // Get the correct key's last 4 bytes (i.e., last 16 bits)
    uint32_t true_key = true_partial_key();
    uint16_t true_last4 = true_key & 0xFFFF;

    // Count survivors with correct last 4 bytes and build histogram for all last4
    uint32_t correct_last4 = 0;
    std::vector<uint32_t> last4_hist(1 << 16, 0);
    for (uint32_t kid : subs) {
        uint16_t last4 = kid & 0xFFFF;
        last4_hist[last4]++;
        if (last4 == true_last4) correct_last4++;
    }

    // Compute average survivors per wrong last4 (excluding the correct one)
    uint32_t sum_wrong = 0, count_wrong = 2<<16-1; // total possible last4 values minus 1 for the correct one
    for (uint32_t i = 0; i < last4_hist.size(); ++i) {
        if (i == true_last4) continue;
        sum_wrong += last4_hist[i];
    }
    double avg_survivors_per_wrong_last4 = (count_wrong > 0) ? (double)sum_wrong / count_wrong : 0.0;

    /* ---------- write one CSV line -------------------------------------- */
    log << seed << ',' << subs.size() << ',' << correct_last4-1 << ',' << avg_survivors_per_wrong_last4 << '\n';
    log.flush();

    return { static_cast<uint32_t>(subs.size()), correct_last4-1, avg_survivors_per_wrong_last4, seed };
}
void write_header(std::ofstream& log)
{
    log << "seed,survivors,correct_last4,avg_survivors_per_wrong_last4\n";
}

int main(int argc, char* argv[])
{
    /* ---- how many repetitions? ---------------------------------------- */
    const int N = (argc > 1) ? std::atoi(argv[1]) : 10000;   // default 10 runs

    std::ofstream log("integral_runs_true_vs_wrong_part.csv",
                      std::ios::out | std::ios::trunc);
    write_header(log);

    /* ---- run N independent experiments -------------------------------- */
    uint64_t start_seed = 0xDAADBEEFCAFEBABEull;          // or time(NULL)
    for (int i = 0; i < N; ++i) {
        uint64_t seed = start_seed + i;                   // simple variation
        RunResult r = run_once(seed, log);
        std::cout << "Run " << i
                  << "  seed=0x" << std::hex << r.seed
                  << "  survivors=" << std::dec << r.survivors
                  << "  correct_last4=" << r.correct_last4
                  << "  avg_survivors_per_wrong_last4=" << r.avg_survivors_per_wrong_last4
                  << '\n';
    }
    std::cout << "\nAll results written to integral_runs_true_vs_wrong_part.csv\n";
    return 0;
}