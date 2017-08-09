#include "filters.h"

#include <iostream>
#include <assert.h>
#include <string>
#include <fstream>
#include <vector>
#include "stdlib.h"
#include <math.h>

using namespace std;
using namespace kernels::filters;

#define BLOCK_DIM_X    29
#define BLOCK_DIM_Y    29
#define THREADS_PER_BLOCK (1 << 10)

#define cucheck_dev(call)                                   \
{								\
    cudaError_t cucheck_err = (call);				\
    if(cucheck_err != cudaSuccess) {				\
        const char *err_str = cudaGetErrorString(cucheck_err);	\
	printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);	\
	assert(0);						\
    }								\
}

// Helpers
__device__ static int getGlobalIdx_1D_1D()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ static int getGlobalIdx_1D_2D()
{
    return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

// GRAY SCALING Time O(1), work O(n)
__global__ static void par_scale( pixel *pixels, int w, int h, float wr, float wg, float wb )
{
    int sz = w*h;
    int gid = getGlobalIdx_1D_1D();

    if( gid < sz )
    {
        pixels[gid].grayScale(wr, wg, wb);
    }

    __syncthreads();
}

void kernels::filters::scale( pixel *pixels, int w, int h, const std::vector<float> &args )
{
    int len = w * h;
    int numBlocks = (int) ceil( (1.0f * len)/THREADS_PER_BLOCK );
    cout<<"gray " << args[0] << " " << args[1] << " " << args[2] << endl;
    par_scale<<<numBlocks, THREADS_PER_BLOCK>>>( pixels, w, h, args[0], args[1], args[2] );
}

// GRAY SCALING Time O(1), work O(n)
__global__ static void par_modulate( pixel *pixels, int w, int h, float wr, float wg, float wb )
{
    int sz = w*h;
    int gid = getGlobalIdx_1D_1D();

    if( gid < sz )
    {
        pixels[gid].apply_weights(wr, wg, wb);
    }

    __syncthreads();
}

void kernels::filters::modulate( pixel *pixels, int w, int h, const std::vector<float> &args )
{
    int len = w * h;
    int numBlocks = (int) ceil( (1.0f * len)/THREADS_PER_BLOCK );
    cout<<"gray " << args[0] << " " << args[1] << " " << args[2] << endl;
    par_modulate<<<numBlocks, THREADS_PER_BLOCK>>>( pixels, w, h, args[0], args[1], args[2] );
}

// BLURRING OPS
// Computes the index where to write global memory to the block shared memory
__device__ static int prepare_shared_data( const pixel *pixels, pixel *sData,int w, \
					   int h, int n, int *start, int *xMinLim, \
					   int *xMaxLim, int *yMinLim, int *yMaxLim)
{
    int idx = -1;
    int tidX = threadIdx.x;
    int tidY = threadIdx.y;
    int sIdx = threadIdx.y * blockDim.x + threadIdx.x;
    
    __shared__ int bDimX;
    __shared__ int bDimY;
    __shared__ int max_blocks_per_width;
    __shared__ int max_blocks_per_height;

    if( tidX == 3 && tidY == 3 )
    {
	bDimX = blockDim.x - 2*n;
	bDimY = blockDim.y - 2*n;
	*start = sIdx - (n * blockDim.x + n);
        max_blocks_per_width   = (int) ceil( w / (1.f * bDimX) );
        max_blocks_per_height  = (int) ceil( h / (1.f * bDimY) );
	*xMinLim = n;
	*yMinLim = n;

	*xMaxLim = bDimX + n - 1;
	if( (blockIdx.x + 1) % max_blocks_per_width == 0 )
	{
	    *xMaxLim += (w % bDimX) - bDimX;
	}

	*yMaxLim = bDimY + n - 1;
	if( ((blockIdx.x / max_blocks_per_width) + 1) % max_blocks_per_height == 0 )
	{
	    *yMaxLim += (h % bDimY) - bDimY;
	}
    }
    __syncthreads();

    // Compute the start block if the 1D array is considered 2D row aligned
    int xB = blockIdx.x % max_blocks_per_width;
    int yB = blockIdx.x / max_blocks_per_width;

    // Compute position in row direction, then in col direction
    int x_w = xB * bDimX + tidX - n;
    int y_h = yB * bDimY + tidY - n;

    // bool valid = ( x_w >= 0 && x_w < w && y_h >= 0 && y_h < h );
    bool valid = ( x_w >= 0 && x_w < w && y_h >= 0 && y_h < h );
    if( valid )
    {
        idx = w * y_h + x_w;
        sData[sIdx] = pixels[idx];
	sData[sIdx].valid = true;
    }
    else
    {
      	sData[sIdx].valid = false;
    }

    __syncthreads();

    // if( blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 )
    // {
    // 	printf("\nBlock#%d\n", blockIdx.x);
    // 	for(int i = 0; i < blockDim.x*blockDim.x; i++)
    // 	{
    // 	    if( i % blockDim.x == 0 )
    // 		printf("\n> ");
    // 	    if( sData[i].valid == true)
    // 		printf("%003d ", sData[i].r);
    // 		// printf("%003d ", sData[i].valid);
    // 	    else
    // 		printf("--- ");
    // 	}
    // 	printf("\n");
    // }
    // __syncthreads();
    return idx;
}

__global__ void par_par_convolution( pixel *pixels, int n, const float *coeffs ) 
{
    int tid = threadIdx.x;
    extern __shared__ pixel shMem[];

    // copy from global to local
    shMem[tid] = pixels[tid];

    // Apply each neighbor their coefficient
    shMem[tid].attenuate( coeffs[tid] );

    // Do parallel reduce on array
    int h = (int) ceil( log2(n*1.0f) );

    if( tid < n )
    {
	pixel sum;
	for(int i = 1; i <= h; i++)
	{
	    int base = (int) ceil( n/pow(2.0f, i) );
	    if( tid < base )
	    {
		// compute the left and right indeces
		int L = 2 * tid;
		int R = 2 * tid + 1;

		// Assign temp max to left child
		sum = shMem[L];

		if( R < (int)ceil(n/pow(2.0f, i-1)) )
		    sum = sum + shMem[R];
	    }
	    __syncthreads();
	    shMem[tid] = sum;
	    __syncthreads();
	}
    }
}

__device__ static pixel seq_convolution( pixel *sData, int n, const float *coefs, int w, int h )
{
    accumulator res = {0,0,0,0,0};
    int sz = blockDim.x * blockDim.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int startX = threadIdx.x - n;
    int startY = threadIdx.y - n;

    int N = 2*n + 1;
    pixel tp = {23,56,89,41,5};

    for( int i = 0; i < N; i++ )
    {
        for( int j = 0; j < N; j++ )
	{
	    int idx = (startX + j) + ( (startY + j) * blockDim.x );

	    // if( startX + j < 0 || startX + j > blockDim.x ||
	    // 	startY + i < 0 || startY + i > blockDim.y ||
	    // 	idx < 0 || idx > sz || !sData[idx].valid )
  	   
	    if( idx < 0 || idx >= sz || !sData[idx].valid )
	    {
		tp = sData[tid];
	    }
	    else
	    {
		tp = sData[idx];
	    }

	    res.accumulate( tp, coefs[i*N + j] );
	}
    }

    tp = res;
    return tp;
}

__global__ static void par_convolution( pixel *pixels, const float *coeffs, int w, int h, int n )
{
    int gid = getGlobalIdx_1D_2D();

    __shared__ int xMin;
    __shared__ int xMax;
    __shared__ int yMin;
    __shared__ int yMax;
    __shared__ int start;
    extern __shared__ pixel sData[];

    int sIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int idx = prepare_shared_data( pixels, sData, w, h, n, &start, \
				   &xMin, &xMax, &yMin, &yMax );
    __syncthreads();

    if( sData[sIdx].valid )
    {  
	/*  
	    Dynamic parallelism

	    if(blockIdx.x < 0)
	    {
	        // Unused. Very slow
		int N = 2*n +1;
		int shSz = N * sizeof(pixel);
		pixel *P = (pixel*) malloc( N*N*sizeof(pixel) );
		for(int i = 0; i < N; i++)
		{
		    memcpy( P + (N*i), sData+(start+i*blockDim.x), N*sizeof(pixel));
		}

		par_par_convolution<<<1, N*N, shSz>>>(P, N, coeffs);
		cucheck_dev(cudaGetLastError());
	    
		free(P);
		p = P[0];
	    }
	*/

        pixel p = seq_convolution( sData, n, coeffs, w, h );
        pixels[idx] = p;
    }
    __syncthreads();
}

void kernels::filters::convolution( pixel *pixels, int w, int h, const std::vector<float> &args )
{
    int neighbors = int(sqrt(args.size()) / 2);
    int blockX = (int) ceil( (1.0f * w) / BLOCK_DIM_X );
    int blockY = (int) ceil( (1.0f * h) / BLOCK_DIM_Y );
    int numBlocks = blockX * blockY;

    float *d_coeffs;
    int sz = args.size() * sizeof(float); 
    cudaMalloc( (float**)&d_coeffs, sz );
    cudaMemcpy( d_coeffs, &args[0], sz, cudaMemcpyHostToDevice );

    // Create a shared array of width = w + 2*neighbors and length = h + 2* neighbors
    int W = BLOCK_DIM_X + 2*neighbors;
    int H = BLOCK_DIM_Y + 2*neighbors;
    int shSize = W * H * sizeof(pixel);

    dim3 blockSize( W, H );
    dim3 gridSize( numBlocks );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    par_convolution<<< gridSize, blockSize, shSize>>>( pixels, d_coeffs, \
						       w, h, neighbors );
    cudaEventRecord(stop);
    cudaFree(d_coeffs);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Convolution: Input Size " << w*h*sizeof(pixel) \
	      << " Time " << milliseconds << std::endl;
}



// MEMORY MANAGEMENT
pixel* kernels::init_mem(const pixel *h_in, int w, int h)
{
    int len = w * h;
    pixel *d_in;

    // Allocate the space on device
    cudaMalloc( (void**) &d_in,  len*sizeof(pixel) );
    // Copy to device allocated memory
    cudaMemcpy( d_in, h_in, len*sizeof(pixel), cudaMemcpyHostToDevice);

    return d_in;
}

void kernels::copy_mem( pixel *d_in, pixel *h_out, int w, int h )
{
    int len = w * h;
    cudaMemcpy( h_out, d_in, len*sizeof(pixel), cudaMemcpyDeviceToHost );
}

//clears data from device memory
void kernels::end_mem( pixel *d_in )
{
    cudaFree( d_in );
}

// DEVICE PROPERTIES

void kernels::getProperties( int &devices, int &devId, struct cudaDeviceProp &props )
{
    cudaGetDeviceCount(&devices);
    cudaGetDevice(&devId);

    if( cudaGetDeviceProperties( &props, devId ) )
    {
        cout << "error: Failed to get device properties." << endl;
        return;
    }
}
