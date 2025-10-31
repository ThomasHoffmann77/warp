#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void Bin1DKernel(tfloat* d_input, tfloat* d_output, size_t elements, int binsize);

#if __CUDACC_VER_MAJOR__ < 12
	static template <class T> __global__ void Bin2DKernel(T* d_output, int width);   //added static in order to allow parallel builds
#else
	template <class T> __global__ void Bin2DKernel(T* d_output, int width, cudaTextureObject_t texInput2d_obj);
#endif

	__global__ void Bin3DKernel(tfloat* d_input, tfloat* d_output, int width, int height, int binnedwidth, int binnedheight, int binsize);

	///////////
	//Globals//
	///////////

#if __CUDACC_VER_MAJOR__ < 12
	texture<tfloat, 2> texInput2d;
	#define TEX_INPUT2D(x, y) tex2D(texInput2d, x, y)
#else
	#define TEX_INPUT2D(x, y) tex2D<float>(texInput2d_obj, x, y)
#endif

	/////////////////////////////////////
	//Binning with linear interpolation//
	/////////////////////////////////////

	template <class T> void d_Bin2D(T* d_input, T* d_output, int2 dims, int bincount)
	{
		T* d_intermediate = NULL;

#if __CUDACC_VER_MAJOR__ < 12
		texInput2d.normalized = false;
		texInput2d.filterMode = cudaFilterModeLinear;
#else
		cudaTextureObject_t texInput2d_obj;
#endif

		size_t elements = dims.x * dims.y;
		size_t binnedelements = dims.x * dims.y / (1 << (bincount * 2));

		for (int i = 0; i < bincount; i++)
		{
			tfloat* d_result;
			if (i < bincount - 1)
				cudaMalloc((void**)&d_result, dims.x / (2 << i) * dims.y / (2 << i) * sizeof(tfloat));
			else
				d_result = d_output + binnedelements;

#if __CUDACC_VER_MAJOR__ < 12
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
			cudaBindTexture2D(NULL,
				texInput2d,
				i == 0 ? d_input : d_intermediate,
				desc,
				dims.x / (2 << i),
				dims.y / (2 << i),
				(dims.x / (2 << i)) * sizeof(tfloat));

			int TpB = min(256, dims.x / (2 << i));
			int totalblocks = min((dims.x / (2 << i) + TpB - 1) / TpB, 32768);
			dim3 grid = dim3((uint)totalblocks, dims.y / (2 << i));

			Bin2DKernel << <grid, (uint)TpB >> > (d_result, dims.x / (2 << i));

			cudaUnbindTexture(texInput2d);
#else
			//CUDA >= 12 texture object setup
			cudaResourceDesc resDesc{};
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr = i == 0 ? d_input : d_intermediate;
			resDesc.res.pitch2D.pitchInBytes = (dims.x / (2 << i)) * sizeof(tfloat);
			resDesc.res.pitch2D.width = dims.x / (2 << i);
			resDesc.res.pitch2D.height = dims.y / (2 << i);
			resDesc.res.pitch2D.desc = cudaCreateChannelDesc<tfloat>();

			cudaTextureDesc texDesc{};
			texDesc.addressMode[0] = cudaAddressModeClamp;
			texDesc.addressMode[1] = cudaAddressModeClamp;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 0;

			cudaCreateTextureObject(&texInput2d_obj, &resDesc, &texDesc, nullptr);

			int TpB = min(256, dims.x / (2 << i));
			int totalblocks = min((dims.x / (2 << i) + TpB - 1) / TpB, 32768);
			dim3 grid = dim3((uint)totalblocks, dims.y / (2 << i));

			Bin2DKernel << <grid, (uint)TpB >> > (d_result, dims.x / (2 << i), texInput2d_obj);

			cudaDestroyTextureObject(texInput2d_obj);
#endif

			if (d_result != d_output + binnedelements)
			{
				if (d_intermediate != NULL)
					cudaFree(d_intermediate);
				d_intermediate = d_result;
			}
		}
	}

	////////////////
	//CUDA kernels//
	////////////////

	__global__ void Bin1DKernel(tfloat* d_input, tfloat* d_output, size_t elements, int binsize)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
		{
			tfloat binsum = 0;
			for (int i = 0; i < binsize; i++)
				binsum += d_input[id * binsize + i];
			d_output[id] = binsum / (tfloat)binsize;
		}
	}

#if __CUDACC_VER_MAJOR__ < 12
	template <class T> __global__ void Bin2DKernel(T* d_output, int width)
#else
	template <class T> __global__ void Bin2DKernel(T* d_output, int width, cudaTextureObject_t texInput2d_obj)
#endif
	{
		for (int x = blockIdx.x * blockDim.x + threadIdx.x;
			x < width;
			x += blockDim.x * gridDim.x)
			d_output[blockIdx.y * width + x] = TEX_INPUT2D((float)(x * 2 + 1), (float)(blockIdx.y * 2 + 1));
	}

	__global__ void Bin3DKernel(tfloat* d_input, tfloat* d_output, int width, int height, int binnedwidth, int binnedheight, int binsize)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		if (x >= binnedwidth)
			return;

		for (int y = 0; y < binnedheight; y++)
		{
			tfloat binsum = 0;
			int binvolume = 0;
			for (int dz = 0; dz < binsize; dz++)
				for (int dy = 0; dy < binsize; dy++)
				{
					int xi = x * binsize + dz;
					int yi = y * binsize + dy;
					if (xi < width && yi < height)
					{
						binsum += d_input[yi * width + xi];
						binvolume++;
					}
				}
			d_output[(blockIdx.z * binnedheight + blockIdx.y) * binnedwidth + x] = binsum / (tfloat)binvolume;
		}
	}
}
