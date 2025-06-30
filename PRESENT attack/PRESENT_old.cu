#ifdef _WIN32
#include <Windows.h>
#endif
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "tables_PRESENT.inc"
#define Exhaustive 65536
#define THREAD 1024
#define BLOCK 512
bit64 ciphertext[BLOCK * THREAD] = { 0 };
bit64 round_key[32] = { 0 };
__global__ void PRESENT_exhaustive_multitable(bit64* plaintext_d, bit64 key0, bit64 key1, bit64 plaintext, bit64 ciphertext_real, bit8* sBox4_d, bit64* pBox8_0_d, bit64* pBox8_1_d, bit64* pBox8_2_d, bit64* pBox8_3_d, bit64* pBox8_4_d, bit64* pBox8_5_d, bit64* pBox8_6_d, bit64* pBox8_7_d) {
	__shared__ bit8 sBox4[16][32];
	__shared__ bit64 pBox8_0[256], pBox8_1[256], pBox8_2[256], pBox8_3[256], pBox8_4[256], pBox8_5[256], pBox8_6[256], pBox8_7[256];
	int wTI = threadIdx.x & 31;
	if (threadIdx.x < 256) {
		pBox8_0[threadIdx.x] = pBox8_0_d[threadIdx.x];
		pBox8_1[threadIdx.x] = pBox8_1_d[threadIdx.x];
		pBox8_2[threadIdx.x] = pBox8_2_d[threadIdx.x];
		pBox8_3[threadIdx.x] = pBox8_3_d[threadIdx.x];
		pBox8_4[threadIdx.x] = pBox8_4_d[threadIdx.x];
		pBox8_5[threadIdx.x] = pBox8_5_d[threadIdx.x];
		pBox8_6[threadIdx.x] = pBox8_6_d[threadIdx.x];
		pBox8_7[threadIdx.x] = pBox8_7_d[threadIdx.x];
	}
	if (threadIdx.x < 32) {	for (int i = 0; i < 16; i++) sBox4[i][threadIdx.x] = sBox4_d[i];}
	__syncthreads();
	int i, j;
	bit64 plaintextr = plaintext, key1r = key1, key0r = key0 + blockIdx.x * blockDim.x + threadIdx.x, ciphertext_real2 = ciphertext_real;
	//	bit64 keyhigh=key1r, keylow=key0r, state=plaintextr, temp_0;
	bit64 keyhigh = key0r, keylow = key1r, state = plaintextr, temp_0;
	for (j = 0; j < Exhaustive; j++) { //2^32
		for (i = 1; i < 32; i++) {
			state ^= keyhigh;
			state = pBox8_0[state & 0xFF] | pBox8_1[(state >> 8) & 0xFF] | pBox8_2[(state >> 16) & 0xFF] | pBox8_3[(state >> 24) & 0xFF] | pBox8_4[(state >> 32) & 0xFF] | pBox8_5[(state >> 40) & 0xFF] | pBox8_6[(state >> 48) & 0xFF] | pBox8_7[state >> 56];
			temp_0 = keylow & 0xffff;
			keylow = (keyhigh >> 3) & 0xffff;
			keyhigh = (keyhigh << 61) ^ (keyhigh >> 19) ^ (temp_0 << 45);
			temp_0 = sBox4[keyhigh >> 60][wTI];
			keyhigh = (keyhigh & 0xfffffffffffffff) | (temp_0 << 60);
			keyhigh ^= (i >> 1);
			keylow ^= (i << 15);
		}
		state ^= keyhigh;
		//		if (state == ciphertext_real2) { plaintext_d[0] = key0 + blockIdx.x * blockDim.x + threadIdx.x; plaintext_d[1] = key1; }
		//		if (state == ciphertext_real2) { plaintext_d[0] = key0 + blockIdx.x * blockDim.x + threadIdx.x; plaintext_d[1] = key1; printf("Found the key! %llx %llx\n",key0r, key1r); }
		if (state == ciphertext_real2) { plaintext_d[0] = key0r; plaintext_d[1] = key1r;  printf("Found the key! %llx %llx\n", key0r, key1r); }
		/*		key1r++;
				keyhigh=key1r;
				keylow=key0r;*/
		key1r++;
		keyhigh = key0r;
		keylow = key1r;
		state = plaintextr;
	}
}
__global__ void PRESENT_exhaustive(bit64 *plaintext_d, bit64 key0, bit64 key1, bit64 plaintext, bit64 ciphertext_real,bit8 *sBox4_d,bit64 *pBox8_0_d,bit64 *pBox8_1_d,bit64 *pBox8_2_d,bit64 *pBox8_3_d,bit64 *pBox8_4_d,bit64 *pBox8_5_d,bit64 *pBox8_6_d,bit64 *pBox8_7_d){
	__shared__ bit8 sBox4[16];
	__shared__ bit64 pBox8_0[256], pBox8_1[256], pBox8_2[256], pBox8_3[256], pBox8_4[256], pBox8_5[256], pBox8_6[256], pBox8_7[256];
	if (threadIdx.x < 256) {
		pBox8_0[threadIdx.x] = pBox8_0_d[threadIdx.x];
		pBox8_1[threadIdx.x] = pBox8_1_d[threadIdx.x];
		pBox8_2[threadIdx.x] = pBox8_2_d[threadIdx.x];
		pBox8_3[threadIdx.x] = pBox8_3_d[threadIdx.x];
		pBox8_4[threadIdx.x] = pBox8_4_d[threadIdx.x];
		pBox8_5[threadIdx.x] = pBox8_5_d[threadIdx.x];
		pBox8_6[threadIdx.x] = pBox8_6_d[threadIdx.x];
		pBox8_7[threadIdx.x] = pBox8_7_d[threadIdx.x];
	}
	if (threadIdx.x<16) sBox4[threadIdx.x]=sBox4_d[threadIdx.x];
	__syncthreads();
	int i,j;
	bit64 plaintextr=plaintext,key1r=key1,key0r=key0+blockIdx.x*blockDim.x+threadIdx.x,ciphertext_real2=ciphertext_real;
//	bit64 keyhigh=key1r, keylow=key0r, state=plaintextr, temp_0;
	bit64 keyhigh = key0r, keylow = key1r, state = plaintextr, temp_0;
	for (j = 0; j< Exhaustive; j++) { //2^32
		for (i=1;i<32;i++) {
			state ^= keyhigh;
			state= pBox8_0[state&0xFF]|pBox8_1[(state>>8)&0xFF]|pBox8_2[(state>>16)&0xFF]|pBox8_3[(state>>24)&0xFF]|pBox8_4[(state>>32)&0xFF]|pBox8_5[(state>>40)&0xFF]|pBox8_6[(state>>48)&0xFF]|pBox8_7[state>>56];
			temp_0 = keylow & 0xffff;
			keylow=(keyhigh>>3)&0xffff;
			keyhigh=(keyhigh<<61)^(keyhigh>>19)^(temp_0<<45);
			temp_0=sBox4[keyhigh>>60];
			keyhigh=(keyhigh & 0xfffffffffffffff)|(temp_0<<60);
			keyhigh ^= (i >> 1);
			keylow ^= (i << 15);
		}
		state ^= keyhigh;
//		if (state == ciphertext_real2) { plaintext_d[0] = key0 + blockIdx.x * blockDim.x + threadIdx.x; plaintext_d[1] = key1; }
//		if (state == ciphertext_real2) { plaintext_d[0] = key0 + blockIdx.x * blockDim.x + threadIdx.x; plaintext_d[1] = key1; printf("Found the key! %llx %llx\n",key0r, key1r); }
		if (state == ciphertext_real2) { plaintext_d[0] = key0r; plaintext_d[1] = key1r;  printf("Found the key! %llx %llx\n", key0r, key1r); }
/*		key1r++;
		keyhigh=key1r;
		keylow=key0r;*/
		key1r++;
		keyhigh = key0r;
		keylow = key1r;
		state=plaintextr;
	}
}
__global__ void PRESENT_CTR(bit64* key_d, bit64* ciphertext, bit64* pBox8_0_d, bit64* pBox8_1_d, bit64* pBox8_2_d, bit64* pBox8_3_d, bit64* pBox8_4_d, bit64* pBox8_5_d, bit64* pBox8_6_d, bit64* pBox8_7_d) {
	__shared__ bit64 pBox8_0[256], pBox8_1[256], pBox8_2[256], pBox8_3[256], pBox8_4[256], pBox8_5[256], pBox8_6[256], pBox8_7[256], keyhigh[32];
	bit64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x < 256) {
		pBox8_0[threadIdx.x] = pBox8_0_d[threadIdx.x];
		pBox8_1[threadIdx.x] = pBox8_1_d[threadIdx.x];
		pBox8_2[threadIdx.x] = pBox8_2_d[threadIdx.x];
		pBox8_3[threadIdx.x] = pBox8_3_d[threadIdx.x];
		pBox8_4[threadIdx.x] = pBox8_4_d[threadIdx.x];
		pBox8_5[threadIdx.x] = pBox8_5_d[threadIdx.x];
		pBox8_6[threadIdx.x] = pBox8_6_d[threadIdx.x];
		pBox8_7[threadIdx.x] = pBox8_7_d[threadIdx.x];
	}
	if (threadIdx.x < 32) keyhigh[threadIdx.x] = key_d[threadIdx.x];
	__syncthreads();
	int i, j;
	bit64 plaintextr = threadIndex;
	bit64 state = plaintextr;
	for (j = 0; j < Exhaustive; j++) { //2^32
		for (i = 0; i < 31; i++) {
			state ^= keyhigh[i];
			state = pBox8_0[state & 0xFF] | pBox8_1[(state >> 8) & 0xFF] | pBox8_2[(state >> 16) & 0xFF] | pBox8_3[(state >> 24) & 0xFF] | pBox8_4[(state >> 32) & 0xFF] | pBox8_5[(state >> 40) & 0xFF] | pBox8_6[(state >> 48) & 0xFF] | pBox8_7[state >> 56];
		}
		ciphertext[threadIndex] = state ^ keyhigh[31];
//		ciphertext[0] = state ^ keyhigh[31];
/*		state^= keyhigh[31];
		if (threadIndex == 1048575) {
			printf("threadIndex : %d\n", threadIndex);
			printf("Ciphertext  : %llx\n", state);
			printf("-------------------------------\n");
		}*/
		state = plaintextr + THREAD*BLOCK;

	}
}
/**
 * key_schedule
 * -------------
 * Computes the round keys for the PRESENT cipher given an 80-bit master key.
 * 
 * Parameters:
 *   key - An array of two 64-bit values representing the 80-bit master key (key[0]: low 64 bits, key[1]: high 16 bits).
 * 
 * Side effects:
 *   Fills the global array 'round_key' with the computed round keys.
 *   Prints the master key and all round keys to stdout.
 */
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
	// for (int i = 0; i < 32; i++) {
	// 	printf("Round key[%2d] = 0x%016llx\n", i, (unsigned long long)round_key[i]);
	// }
}
void CTR() {
	bit64 k[2] = { 0,0 };
	bit64* pBox8_0_d, * pBox8_1_d, * pBox8_2_d, * pBox8_3_d, * pBox8_4_d, * pBox8_5_d, * pBox8_6_d, * pBox8_7_d;
	bit64* round_keys_d, * ciphertext_d;
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);  // Has no effect after Compute Capability 3.5
	key_schedule(k);
	// Allocate array on device
	cudaMalloc((void**)&pBox8_0_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_0_d, pBox8_0, 256 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pBox8_1_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_1_d, pBox8_1, 256 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pBox8_2_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_2_d, pBox8_2, 256 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pBox8_3_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_3_d, pBox8_3, 256 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pBox8_4_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_4_d, pBox8_4, 256 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pBox8_5_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_5_d, pBox8_5, 256 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pBox8_6_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_6_d, pBox8_6, 256 * sizeof(bit64), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&pBox8_7_d, 256 * sizeof(bit64)); cudaMemcpy(pBox8_7_d, pBox8_7, 256 * sizeof(bit64), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&ciphertext_d, THREAD * BLOCK * sizeof(bit64)); 
	cudaMalloc((void**)&round_keys_d, 32 * sizeof(bit64)); cudaMemcpy(round_keys_d, round_key, 32 * sizeof(bit64), cudaMemcpyHostToDevice);
	
	// Kernel call
	StartCounter();
	PRESENT_CTR << <BLOCK, THREAD >> > (round_keys_d, ciphertext_d, pBox8_0_d, pBox8_1_d, pBox8_2_d, pBox8_3_d, pBox8_4_d, pBox8_5_d, pBox8_6_d, pBox8_7_d);
	//	PRESENT_exhaustive_multitable << <BLOCK, THREAD >> > (plaintext_d, k[0], k[1], plaintext, ciphertext, sBox4_d, pBox8_0_d, pBox8_1_d, pBox8_2_d, pBox8_3_d, pBox8_4_d, pBox8_5_d, pBox8_6_d, pBox8_7_d);
//	cudaMemcpy(ciphertext, ciphertext_d, THREAD* BLOCK * sizeof(bit64), cudaMemcpyDeviceToHost);
	cudaMemcpy(ciphertext, ciphertext_d, 2 * sizeof(bit64), cudaMemcpyDeviceToHost);
//	cudaDeviceSynchronize();
	printf("Time: %lf\n", GetCounter());
	printf("Ciphertext is: %I64x %I64x\n", ciphertext[0], ciphertext[1]);
//	printf("Time: %u seconds\n", clock() / CLOCKS_PER_SEC);
	// Cleanup
	cudaFree(ciphertext_d); cudaFree(round_keys_d); cudaFree(pBox8_0_d); cudaFree(pBox8_1_d); cudaFree(pBox8_2_d); cudaFree(pBox8_3_d); cudaFree(pBox8_4_d); cudaFree(pBox8_5_d); cudaFree(pBox8_6_d); cudaFree(pBox8_7_d);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

void exhaustive() {
	bit64 *plaintext_d;
	bit64 ciphertext = 0x26f9752bdde03b8c;
	bit64 plaintext	 = 0xbc7681ece3f79d1a;
	bit64 k[2]={0,0};
	bit64 *pBox8_0_d,*pBox8_1_d,*pBox8_2_d,*pBox8_3_d,*pBox8_4_d,*pBox8_5_d,*pBox8_6_d,*pBox8_7_d;
	bit8 S[16] = {0xc,0x5,0x6,0xb,0x9,0x0,0xa,0xd,0x3,0xe,0xf,0x8,0x4,0x7,0x1,0x2};  // PRESENT's S-box
	bit8 *sBox4_d;
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);  // Has no effect after Compute Capability 3.5
	// Allocate array on device
	cudaMalloc((void **)&plaintext_d, 2*sizeof(bit64));	cudaMemset(plaintext_d,0,2*sizeof(bit64)); 
	cudaMalloc((void **)&sBox4_d, 16*sizeof(bit8));		cudaMemcpy(sBox4_d,S,16*sizeof(bit8),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_0_d, 256*sizeof(bit64));	cudaMemcpy(pBox8_0_d,pBox8_0,256*sizeof(bit64),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_1_d, 256*sizeof(bit64)); cudaMemcpy(pBox8_1_d,pBox8_1,256*sizeof(bit64),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_2_d, 256*sizeof(bit64)); cudaMemcpy(pBox8_2_d,pBox8_2,256*sizeof(bit64),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_3_d, 256*sizeof(bit64)); cudaMemcpy(pBox8_3_d,pBox8_3,256*sizeof(bit64),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_4_d, 256*sizeof(bit64)); cudaMemcpy(pBox8_4_d,pBox8_4,256*sizeof(bit64),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_5_d, 256*sizeof(bit64)); cudaMemcpy(pBox8_5_d,pBox8_5,256*sizeof(bit64),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_6_d, 256*sizeof(bit64)); cudaMemcpy(pBox8_6_d,pBox8_6,256*sizeof(bit64),cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pBox8_7_d, 256*sizeof(bit64)); cudaMemcpy(pBox8_7_d,pBox8_7,256*sizeof(bit64),cudaMemcpyHostToDevice);
	// Kernel call
	StartCounter();
	PRESENT_exhaustive<<<BLOCK, THREAD >>>(plaintext_d,k[0],k[1],plaintext,ciphertext,sBox4_d,pBox8_0_d,pBox8_1_d,pBox8_2_d,pBox8_3_d,pBox8_4_d,pBox8_5_d,pBox8_6_d,pBox8_7_d);
//	PRESENT_exhaustive_multitable << <BLOCK, THREAD >> > (plaintext_d, k[0], k[1], plaintext, ciphertext, sBox4_d, pBox8_0_d, pBox8_1_d, pBox8_2_d, pBox8_3_d, pBox8_4_d, pBox8_5_d, pBox8_6_d, pBox8_7_d);
	cudaMemcpy(k,plaintext_d,2*sizeof(bit64),cudaMemcpyDeviceToHost);
	if (k[0] || k[1]) printf("Correct key is: %I64x %I64x\n",k[1], k[0]);
	printf("Time: %lf\n", GetCounter());
//	printf("Time: %u seconds\n", clock() / CLOCKS_PER_SEC);
	// Cleanup
	cudaFree(plaintext_d);cudaFree(sBox4_d);cudaFree(pBox8_0_d);cudaFree(pBox8_1_d);cudaFree(pBox8_2_d);cudaFree(pBox8_3_d);cudaFree(pBox8_4_d);cudaFree(pBox8_5_d);cudaFree(pBox8_6_d);cudaFree(pBox8_7_d);
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));
}
int main(int argc, char* argv[]) {
	// int choice = 0;
	// printf(
	// 	"(1) Exhaustive search\n"
	// 	"(2) Counter mode\n"
	// 	"Enter choice: "
	// );
	// scanf_s("%d", &choice);
	// if (choice == 1) exhaustive();
	// if (choice == 2) CTR();
	// return 1;
	// scanf_s("%d");
    }