
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	if((blockIdx.x * blockDim.x + threadIdx.x) < size) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		y[i] += scale * x[i];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	uint64_t vectorBytes = vectorSize * sizeof(float);

	float * a, * b, * c;

	a = (float *) malloc(vectorSize * sizeof(float));
	b = (float *) malloc(vectorSize * sizeof(float));
	c = (float *) malloc(vectorSize * sizeof(float));

	if (a == NULL || b == NULL || c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	//	C = B
	std::memcpy(c, b, vectorSize * sizeof(float));
	float scale = 2.1f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	float * gpu_a, * gpu_c;

	cudaMalloc(&gpu_a, vectorSize * sizeof(float));
	cudaMalloc(&gpu_c, vectorSize * sizeof(float));
	
	cudaMemcpy(gpu_a, a, vectorBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_c, c, vectorBytes, cudaMemcpyHostToDevice);

	cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
	saxpy_gpu<<< ceil(vectorSize/props.maxThreadsPerBlock), props.maxThreadsPerBlock >>>(gpu_a, gpu_c, scale, vectorSize);

	cudaMemcpy(c, gpu_c, vectorBytes, cudaMemcpyDeviceToHost);

	cudaFree(gpu_a);
	cudaFree(gpu_c);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	free(a);
	free(b);
	free(c);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	if(blockIdx.x * blockDim.x + threadIdx.x < pSumSize){
		int count = 0;
		curandState_t rng;
		curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &rng);
		for(int i=0; i<sampleSize; i++){
			// Rand
    	double x = curand_uniform(&rng);
			double y = curand_uniform(&rng);
			// Magnitude
			double magnitude = x * x + y * y;
			// Count or No Count
			if(magnitude <= 1) count++;
		}
		pSums[blockIdx.x * blockDim.x + threadIdx.x] = count;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t totalsSize, uint64_t reduceSize) {
	if(blockIdx.x * blockDim.x + threadIdx.x < totalsSize){
		uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
		uint64_t count = 0;
		uint64_t pIdx;
		for(int i=0; i<reduceSize; i++){
			pIdx = threadId * reduceSize + i;
			if(pIdx < pSumSize){
				count += pSums[pIdx];
			}
		}
		totals[threadId] = count;
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	// Insert code here
	uint64_t /** counts,*/ * partial_sums;
	//counts = (uint64_t *) malloc(generateThreadCount * sizeof(uint64_t));
	partial_sums = (uint64_t *) malloc(reduceThreadCount * sizeof(uint64_t));

	uint64_t * gpu_counts, * gpu_partial_sums;
	cudaMalloc(&gpu_counts, generateThreadCount * sizeof(uint64_t));
	cudaMalloc(&gpu_partial_sums, reduceThreadCount * sizeof(uint64_t));

	cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);

	generatePoints<<< (int)ceil((double)(generateThreadCount)/(double)(props.maxThreadsPerBlock)), props.maxThreadsPerBlock >>>(gpu_counts, generateThreadCount, sampleSize);
	
	reduceCounts<<< (int)ceil((double)(reduceThreadCount)/(double)(props.maxThreadsPerBlock)), props.maxThreadsPerBlock >>>(gpu_counts, gpu_partial_sums, generateThreadCount, reduceThreadCount, reduceSize);

	uint64_t countBytes = generateThreadCount * sizeof(uint64_t);
	uint64_t partialBytes = reduceThreadCount * sizeof(uint64_t);

//	cudaMemcpy(counts, gpu_counts, countBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(partial_sums, gpu_partial_sums, partialBytes, cudaMemcpyDeviceToHost);

	cudaFree(gpu_counts);
	cudaFree(gpu_partial_sums);

	uint64_t totalSum;

	totalSum = 0;
	for(int i=0; i<reduceThreadCount; i++){
		totalSum += partial_sums[i];
	}

	approxPi = 4.0 * (double)(totalSum) / (double)(generateThreadCount * sampleSize);

	//free(counts);
	free(partial_sums);

	return approxPi;
}
