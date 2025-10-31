#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	//Kernel for applying an irregular spherical mask to the input
	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center
#if __CUDACC_VER_MAJOR__ >= 12
		, cudaTextureObject_t texIrregularSphereRadius2d_obj
#endif
	);

	///////////
	//Globals//
	///////////

#if __CUDACC_VER_MAJOR__ < 12
	static texture<tfloat, 2, cudaReadModeElementType> texIrregularSphereRadius2d;  //added static in order to allow parallel builds
	#define TEX_RADIUS(phi, theta) tex2D(texIrregularSphereRadius2d, phi, theta)
#else
	#define TEX_RADIUS(phi, theta) tex2D<tfloat>(texIrregularSphereRadius2d_obj, phi, theta)
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
		tfloat* d_pitched = NULL;
		int pitchedwidth = anglesteps.x * sizeof(tfloat);
		d_pitched = (tfloat*)CudaMallocAligned2D(anglesteps.x * sizeof(tfloat), anglesteps.y, &pitchedwidth);
		for (int y = 0; y < anglesteps.y; y++)
			cudaMemcpy((char*)d_pitched + y * pitchedwidth,
				d_radiusmap + y * anglesteps.x,
				anglesteps.x * sizeof(tfloat),
				cudaMemcpyDeviceToDevice);

#if __CUDACC_VER_MAJOR__ < 12
		texIrregularSphereRadius2d.normalized = true;
		texIrregularSphereRadius2d.filterMode = cudaFilterModeLinear;
		texIrregularSphereRadius2d.addressMode[0] = cudaAddressModeMirror;
		texIrregularSphereRadius2d.addressMode[1] = cudaAddressModeMirror;

		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		cudaBindTexture2D(NULL,
			texIrregularSphereRadius2d,
			d_pitched,
			desc,
			anglesteps.x,
			anglesteps.y,
			pitchedwidth);
#else
		//CUDA >=12 texture object setup
		cudaResourceDesc resDesc{};
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = d_pitched;
		resDesc.res.pitch2D.pitchInBytes = pitchedwidth;
		resDesc.res.pitch2D.width = anglesteps.x;
		resDesc.res.pitch2D.height = anglesteps.y;
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc<tfloat>();

		cudaTextureDesc texDesc{};
		texDesc.addressMode[0] = cudaAddressModeMirror;
		texDesc.addressMode[1] = cudaAddressModeMirror;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		cudaTextureObject_t texIrregularSphereRadius2d_obj;
		cudaCreateTextureObject(&texIrregularSphereRadius2d_obj, &resDesc, &texDesc, nullptr);
#endif

		tfloat3 _center = center != NULL ? *center : tfloat3(dims.x / 2, dims.y / 2, dims.z / 2);

		int TpB = min(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, batch);
		if (DimensionCount(dims) <= 2)
			IrregularSphereMaskKernel<T, 2> << <grid, TpB >> > (d_input, d_output, dims, sigma, _center
#if __CUDACC_VER_MAJOR__ >= 12
				, texIrregularSphereRadius2d_obj
#endif
				);
		else
			IrregularSphereMaskKernel<T, 3> << <grid, TpB >> > (d_input, d_output, dims, sigma, _center
#if __CUDACC_VER_MAJOR__ >= 12
				, texIrregularSphereRadius2d_obj
#endif
				);

#if __CUDACC_VER_MAJOR__ < 12
		//Unbind texture
		cudaUnbindTexture(texIrregularSphereRadius2d);
#else
		//Destroy texture object
		cudaDestroyTextureObject(texIrregularSphereRadius2d_obj);
#endif
		cudaFree(d_pitched);
	}

	template void d_IrregularSphereMask<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* d_radiusmap, int2 anglesteps, tfloat sigma, tfloat3* center, int batch);

	////////////////
	//CUDA kernels//
	////////////////

	template <class T, int ndims> __global__ void IrregularSphereMaskKernel(T* d_input, T* d_output, int3 dims, tfloat sigma, tfloat3 center
#if __CUDACC_VER_MAJOR__ >= 12
		, cudaTextureObject_t texIrregularSphereRadius2d_obj
#endif
	)
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

			tfloat radius = TEX_RADIUS(phi, theta);

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
