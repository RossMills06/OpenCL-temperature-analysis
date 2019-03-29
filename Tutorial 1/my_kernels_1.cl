
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int lN = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < lN; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < lN))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_add(&B[0], scratch[lid]);
	}
}

kernel void reduce_max_4(global const int* A, global int* C, local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid + i] > scratch[lid])
			{
				scratch[lid] = scratch[lid + i];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if (!lid)
	{
		atomic_max(&C[0], scratch[lid]);
	}
}

kernel void reduce_min_4(global const int* A, global int* D, local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid + i] < scratch[lid])
			{
				scratch[lid] = scratch[lid + i];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if (!lid)
	{
		atomic_min(&D[0], scratch[lid]);
	}
}

kernel void reduce_standDev_4(global const int* A, global int* B, global int* avgTotal, local int* scratch)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	int lid = get_local_id(0);
	int lN = get_local_size(0);

	int avg = avgTotal[0] / N; // getting mean value

	//cache all N values from global memory to local memory
	scratch[lid] = ((A[id] - avg) * (A[id] - avg)) / 10; // claucualting stand dev

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < lN; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < lN))
		{
			scratch[lid] += scratch[lid + i];
		}


		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_add(&B[0], scratch[lid]);
	}
}

// code for vector sorting 
// http://www.bealto.com/gpu-sorting_parallel-selection.html
// date accessed 6th march 2019

kernel void sort(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0); // current thread
	int N = get_global_size(0); // input size
	int lN = get_local_size(0); // workgroup size
	int iKey = A[id]; // input key for current thread

	// Compute position of iKey in output
	int pos = 0;
	// Loop on blocks of size BLOCKSIZE keys (BLOCKSIZE must divide N)
	for (int j = 0; j < N; j += lN)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int index = get_local_id(0); index < lN; index += lN)
		{
			scratch[index] = A[j + index];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop on all values in local
		for (int index = 0; index < lN; index++)
		{
			int jKey = scratch[index]; // broadcasted, local memory
			bool smaller = (jKey < iKey) || (jKey == iKey && (j + index) < id); // in[j] < in[i] ?

			if (smaller)
			{
				pos += 1;
			}
		}
	}
	B[pos] = iKey;
}


//__kernel void ParallelSelection_Blocks(__global const data_t * in, __global data_t * out, __local uint * aux)
//{
//	int i = get_global_id(0); // current thread
//	int n = get_global_size(0); // input size
//	int wg = get_local_size(0); // workgroup size
//	data_t iData = in[i]; // input record for current thread
//	uint iKey = keyValue(iData); // input key for current thread
//	int blockSize = BLOCK_FACTOR * wg; // block size
//
//	// Compute position of iKey in output
//	int pos = 0;
//	// Loop on blocks of size BLOCKSIZE keys (BLOCKSIZE must divide N)
//	for (int j = 0;j < n;j += blockSize)
//	{
//		// Load BLOCKSIZE keys using all threads (BLOCK_FACTOR values per thread)
//		barrier(CLK_LOCAL_MEM_FENCE);
//		for (int index = get_local_id(0);index < blockSize;index += wg)
//			aux[index] = keyValue(in[j + index]);
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		// Loop on all values in AUX
//		for (int index = 0;index < blockSize;index++)
//		{
//			uint jKey = aux[index]; // broadcasted, local memory
//			bool smaller = (jKey < iKey) || (jKey == iKey && (j + index) < i); // in[j] < in[i] ?
//			pos += (smaller) ? 1 : 0;
//		}
//	}
//	out[pos] = iData;
//}

