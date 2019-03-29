
//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
kernel void reduce_add_4(global const float* A, global float* B, local float* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int lN = get_local_size(0);
	int N = get_global_size(0);
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int stride = lN / 2; stride > 0; stride /= 2) 
	{
		if (lid < stride)
		{
			scratch[lid] += scratch[lid + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// coalesced memory

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	barrier(CLK_LOCAL_MEM_FENCE);
	if (!lid) 
	{
		//atomic_add(&B[0],scratch[lid]);
		B[gid] = scratch[lid];
		// saving sum of workgroup (scrath[lid]) into the output array relative to group id
		// sum of work groups are all next to each other
		// e.g. first workgroup sum gets saved to the first output array index

	}
}

kernel void reduce_max_4(global const float* A, global float* C, local float* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int lN = get_local_size(0);
	int N = get_global_size(0);
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int stride = lN / 2; stride > 0; stride /= 2)
	{
		if (lid < stride)
		{
			if (scratch[lid + stride] > scratch[lid])
			{
				scratch[lid] = scratch[lid + stride];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if (!lid)
	{
		//atomic_max(&C[0], scratch[lid]);
		C[gid] = scratch[lid];
	}
}

kernel void reduce_min_4(global const float* A, global float* D, local float* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int lN = get_local_size(0);
	int N = get_global_size(0);
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int stride = lN / 2; stride > 0; stride /= 2)
	{
		if (lid < stride)
		{
			if (scratch[lid + stride] < scratch[lid])
			{
				scratch[lid] = scratch[lid + stride];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if (!lid)
	{
		//atomic_max(&C[0], scratch[lid]);

		D[gid] = scratch[lid];

	}
}

kernel void reduce_standDev_4(global const float* A, global float* B, global float* avgTotal, local float* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int lN = get_local_size(0);
	int N = get_global_size(0);
	int gid = get_group_id(0);

	float avg = avgTotal[0] / N; // getting mean value

	//cache all N values from global memory to local memory
	scratch[lid] =  ((A[id] - avg) * (A[id] - avg)); // claucualting stand dev

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int stride = lN / 2; stride > 0; stride /= 2)
	{
		if (lid < stride)
		{
			scratch[lid] += scratch[lid + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	barrier(CLK_LOCAL_MEM_FENCE);
	if (!lid)
	{
		//atomic_add(&B[0],scratch[lid]);
		B[gid] = scratch[lid];
	}
}

// code for vector sorting 
// http://www.bealto.com/gpu-sorting_parallel-selection.html
// date accessed 6th march 2019

kernel void sort(global const float* A, global float* B, local float* scratch)
{
	int id = get_global_id(0); // current thread
	int N = get_global_size(0); // input size
	int lN = get_local_size(0); // workgroup size
	float iKey = A[id]; // input key for current thread

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
			float jKey = scratch[index]; // broadcasted, local memory
			bool smaller = (jKey < iKey) || (jKey == iKey && (j + index) < id);

			if (smaller)
			{
				pos += 1;
			}
		}
	}
	B[pos] = iKey;
}

// sorting NOT using local memory
kernel void ParallelSelection(global const int* A, global int* B)
{
	int id = get_global_id(0); // current thread
	int n = get_global_size(0); // input size
	int iKey = A[id];
	// Compute position of in[i] in output
	int pos = 0;
	for (int j = 0;j < n;j++)
	{
		uint jKey = A[j]; // broadcasted
		bool smaller = (jKey < iKey) || (jKey == iKey && j < id);  // in[j] < in[i] ?
		pos += (smaller) ? 1 : 0;
	}
	B[pos] = iKey;
}


void cmpxchg(global float* A, global float* B)
{
	if (*A > *B)
	{
		float t = *A; *A = *B; *B = t;
	}
}

kernel void sort_oddeven(global float* A, global float* B) 
{
	int id = get_global_id(0); 
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 0; i < N; i += 2) 
	{	//step

		if (id % 2 == 0 && id + 1 < N)
		{
			//odd
			cmpxchg(&A[id], &A[id + 1]);
			barrier(CLK_GLOBAL_MEM_FENCE);
		}
			
		if (id % 2 == 1 && id + 1 < N)
		{
			//even
			cmpxchg(&A[id], &A[id + 1]);
			barrier(CLK_GLOBAL_MEM_FENCE);
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

