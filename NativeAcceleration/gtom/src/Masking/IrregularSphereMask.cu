#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

#if __CUDACC_VER_MAJOR__ >= 12
	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center, cudaTextureObject_t texObj);
#else
	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center);
#endif


	///////////
	//Globals//
	///////////

#if __CUDACC_VER_MAJOR__ < 12
	texture<tfloat, 2, cudaReadModeElementType> texIrregularSphereRadius2d;
#endif


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_IrregularSphereMask(T* d_input,
		T* d_output,
		int3 dims,
		tfloat* d_radiusmap,
		int2 anglesteps,
		tfloat sigma,
		tfloat3* center,
		int batch)
	{
#if __CUDACC_VER_MAJOR__ >= 12
		// Pitch allocation
		tfloat* d_pitched = NULL;
		int pitchedwidth = anglesteps.x * sizeof(tfloat);
		d_pitched = (tfloat*)CudaMallocAligned2D(anglesteps.x * sizeof(tfloat), anglesteps.y, &pitchedwidth);
		for (int y = 0; y < anglesteps.y; y++)
			cudaMemcpy((char*)d_pitched + y * pitchedwidth,
				d_radiusmap + y * anglesteps.x,
				anglesteps.x * sizeof(tfloat),
				cudaMemcpyDeviceToDevice);

		// Texture Object Setup
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<tfloat>();
		cudaArray_t cuArray;
		cudaMallocArray(&cuArray, &channelDesc, anglesteps.x, anglesteps.y);
		cudaMemcpy2DToArray(cuArray, 0, 0, d_pitched, pitchedwidth, anglesteps.x * sizeof(tfloat), anglesteps.y, cudaMemcpyDeviceToDevice);

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = true;

		cudaTextureObject_t texIrregularSphereRadius2dObj;
		cudaCreateTextureObject(&texIrregularSphereRadius2dObj, &resDesc, &texDesc, nullptr);
#else
		texIrregularSphereRadius2d.normalized = true;
		texIrregularSphereRadius2d.filterMode = cudaFilterModeLinear;
		texIrregularSphereRadius2d.addressMode[0] = cudaAddressModeMirror;
		texIrregularSphereRadius2d.addressMode[1] = cudaAddressModeMirror;
#endif

		tfloat3 _center = center != NULL ? *center : tfloat3(dims.x / 2, dims.y / 2, dims.z / 2);

		int TpB = min(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, batch);
#if __CUDACC_VER_MAJOR__ >= 12
		if (DimensionCount(dims) <= 2)
			IrregularSphereMaskKernel<T, 2> <<<grid, TpB>>> (d_input, d_output, dims, sigma, _center, texIrregularSphereRadius2dObj);
		else
			IrregularSphereMaskKernel<T, 3> <<<grid, TpB>>> (d_input, d_output, dims, sigma, _center, texIrregularSphereRadius2dObj);
#else
		if (DimensionCount(dims) <= 2)
			IrregularSphereMaskKernel<T, 2> <<<grid, TpB>>> (d_input, d_output, dims, sigma, _center);
		else
			IrregularSphereMaskKernel<T, 3> <<<grid, TpB>>> (d_input, d_output, dims, sigma, _center);
#endif

#if __CUDACC_VER_MAJOR__ >= 12
		cudaDestroyTextureObject(texIrregularSphereRadius2dObj);
		cudaFreeArray(cuArray);
		cudaFree(d_pitched);
#else
		cudaUnbindTexture(texIrregularSphereRadius2d);
		cudaFree(d_pitched);
#endif
	}
	template void d_IrregularSphereMask<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* d_radiusmap, int2 anglesteps, tfloat sigma, tfloat3* center, int batch);


	////////////////
	//CUDA kernels//
	////////////////

#if __CUDACC_VER_MAJOR__ >= 12
	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center, cudaTextureObject_t texObj)
#else
	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center)
#endif
	{
		if (threadIdx.x >= dims.x)
			return;

		//For batch mode
		int offset = blockIdx.z * Elements(dims) + blockIdx.y * dims.x * dims.y + blockIdx.x * dims.x;

		int x, y, z;
		float length;
		T maskvalue;

		//Squared y and z distance from center
		y = blockIdx.x - center.y;
		if (ndims > 2)
			z = blockIdx.y - center.z;
		else
			z = 0;

		for (int idx = threadIdx.x; idx < dims.x; idx += blockDim.x)
		{
			x = idx - center.x;

			length = sqrt((float)(x * x + y * y + z * z));

			glm::vec3 direction((float)x / length, (float)y / length, (float)z / length);
			float theta = acos((float)(-direction.x));
			float phi = atan2((float)direction.y / sin(theta), (float)direction.z / sin(theta));

			theta /= PI * 0.5f;
			phi /= PI2;

#if __CUDACC_VER_MAJOR__ >= 12
			tfloat radius = tex2D<tfloat>(texObj, phi, theta);
#else
			tfloat radius = tex2D(texIrregularSphereRadius2d, phi, theta);
#endif

			if (length < radius)
				maskvalue = 1;
			else
			{
				//Smooth border
				if (sigma > (tfloat)0)
				{
					maskvalue = exp(-((length - radius) * (length - radius) / (sigma * sigma)));
					if (maskvalue < (tfloat)0.1353)
						maskvalue = 0;
				}
				//Hard border
				else
					maskvalue = max((T)1 - (length - radius), (T)0);
			}

			//Write masked input to output
			d_output[offset + idx] = maskvalue * d_input[offset + idx];
			//d_output[offset + idx] = radius;
		}
	}
}
