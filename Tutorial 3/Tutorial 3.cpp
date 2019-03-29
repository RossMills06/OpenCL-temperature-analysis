#pragma comment(lib, "OpenCl.lib")

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() 
{
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) 
{
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) 
		{ 
			platform_id = atoi(argv[++i]); 
		}
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) 
		{ 
			device_id = atoi(argv[++i]); 
		}
		else if (strcmp(argv[i], "-l") == 0) 
		{ 
			std::cout << ListPlatformsDevices() << std::endl; 
		}
		else if (strcmp(argv[i], "-h") == 0) 
		{ 
			print_help(); 
		}
	}

	//reading file in
	fstream file;
	string fileDir, word;

	std::vector<string> tempInfoString;
	std::vector<float> tempInfo;

	//fileDir = "C:\\Users\\Student\\Desktop\\OpenCL- Assignment\\temp_lincolnshire_short.txt";
	fileDir = "C:\\Users\\Student\\Desktop\\OpenCL- Assignment\\temp_lincolnshire.txt";

	file.open(fileDir.c_str());

	while (file >> word)
	{
		tempInfoString.push_back(word);
		// reading each word into a vector
	}

	for (int i = 5; i < tempInfoString.size(); i += 6)
	{
		tempInfo.push_back(strtof((tempInfoString[i]).c_str(), 0));
		// taking only the temp floats and adding them to another vector
		// converting to float
	}

	int numberOfElements = tempInfo.size();
	// getting number of elements

	//detect any potential exceptions
	try
	{
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels_3.cl");
		// get the kernel code

		cl::Program program(context, sources);

		//build and debug the kernel code
		try 
		{
			program.build();
		}
		catch (const cl::Error& err)
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// typedef int mytype;
		typedef float mytype;

		//host - input
		std::vector<mytype> A;
		for (int i = 0; i < tempInfo.size(); i++)
		{
			A.push_back(tempInfo[i]);
		}

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 128; // workgroup size
		// work group size may result in different values on different devices

		size_t padding_size = A.size() % local_size;

		// if the input vector is not a multiple of the local_size
		// insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) 
		{
			//create an extra vector with neutral values
			std::vector<mytype> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size() * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		//std::vector<mytype> B(input_elements);
		std::vector<mytype> Bavg(input_elements);
		std::vector<mytype> Cmax(input_elements);
		std::vector<mytype> Dmin(input_elements);
		std::vector<mytype> Estanddev(input_elements);
		std::vector<mytype> sortedVec(input_elements);
		// output vectors for the results

		size_t output_size = Bavg.size() * sizeof(mytype);//size in bytes
		size_t sorted_size = A.size() * sizeof(mytype);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size); // buffer for average
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size); // buffer for max
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size); // buffer for min
		cl::Buffer buffer_standdev(context, CL_MEM_READ_WRITE, output_size); // buffer for standdev
		cl::Buffer buffer_sorted(context, CL_MEM_READ_WRITE, sorted_size); // buffer for sorting

		//copy arrays to buffer and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);//zero C buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);//zero D buffer on device memory
		queue.enqueueFillBuffer(buffer_standdev, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_sorted, 0, 0, sorted_size);

		// ********** AVERAGE KERNEL **********
		cl::Kernel kernel_add = cl::Kernel(program, "reduce_add_4");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Event prof_event_AVG;
		float AVG_kernel_time;
		float AVG_single_kernel;
		cl::Event prof_event_AVG_mem;
		float AVG_memory_time;
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_AVG); // call the kernel
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &Bavg[0], NULL, &prof_event_AVG_mem); // copy the output to the host
		AVG_kernel_time = prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // get kernel profiling time
		AVG_single_kernel = prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // get kernel profiling time
		AVG_memory_time = prof_event_AVG_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_AVG_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // get memory transfer time

		// loop until vector elements is less than workgroup size
		// kernel redcues for only the workgroup size
		// breaking down into chunks
		// reduces until only one value
		// TILING!
		while (Bavg[1] != 0.0f)
		{
			kernel_add.setArg(0, buffer_B);
			kernel_add.setArg(1, buffer_B);
			kernel_add.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_AVG);
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &Bavg[0], NULL, &prof_event_AVG_mem);
			AVG_kernel_time += prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			AVG_memory_time += prof_event_AVG_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_AVG_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>(); // get memory transfer time
		}
		// loop through kernel calls until fully reduced

		

		float totalTemp = Bavg[0];
		float avgTemp = totalTemp / numberOfElements;
		// calcualting avg temp

		// ********** MAX KERNEL **********
		cl::Kernel kernel_max = cl::Kernel(program, "reduce_max_4");
		kernel_max.setArg(0, buffer_A);
		kernel_max.setArg(1, buffer_C);
		kernel_max.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Event prof_event_MAX;
		float max_kernel_time;
		float max_single_kernel;
		cl::Event prof_event_MAX_mem;
		float max_memory_time;
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_MAX);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &Cmax[0], NULL, &prof_event_MAX_mem);
		max_kernel_time = prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		max_single_kernel = prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		max_memory_time = prof_event_MAX_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MAX_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// reduce until left with a single value
		while (Cmax[1] != 0.0f)
		{
			kernel_max.setArg(0, buffer_C);
			kernel_max.setArg(1, buffer_C);
			kernel_max.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_MAX);
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &Cmax[0], NULL, &prof_event_MAX_mem);
			max_kernel_time += prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			max_memory_time += prof_event_MAX_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MAX_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		}

		

		float maxTemp = Cmax[0];

		// ********** MIN KERNEL **********
		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min_4");
		kernel_min.setArg(0, buffer_A);
		kernel_min.setArg(1, buffer_D);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Event prof_event_MIN;
		float min_kernel_time;
		float min_single_kernel;
		cl::Event prof_event_MIN_mem;
		float min_memory_time;
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_MIN);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &Dmin[0], NULL, &prof_event_MIN_mem);
		min_kernel_time = prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		min_single_kernel = prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		min_memory_time = prof_event_MIN_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MIN_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		// reduce until left with a single value
		while (Dmin[1] != 0.0f)
		{
			kernel_min.setArg(0, buffer_D);
			kernel_min.setArg(1, buffer_D);
			kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_MIN);
			queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &Dmin[0], NULL, &prof_event_MIN_mem);
			min_kernel_time += prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			min_memory_time += prof_event_MIN_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MIN_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		}

		float minTemp = Dmin[0];

		// ********** STAND DEV KERNEL **********
		cl::Kernel kernel_standDev = cl::Kernel(program, "reduce_standDev_4");
		kernel_standDev.setArg(0, buffer_A);
		kernel_standDev.setArg(1, buffer_standdev);
		kernel_standDev.setArg(2, buffer_B); // pass through the output from reduce add to get mean for stand dev
		kernel_standDev.setArg(3, cl::Local(local_size * sizeof(mytype)));

		cl::Event prof_event_STANDDEV;
		float standdev_kernel_time;
		float standdev_single_kernel;
		cl::Event prof_event_STANDDEV_mem;
		float standdev_memory_time;
		queue.enqueueNDRangeKernel(kernel_standDev, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_STANDDEV);
		queue.enqueueReadBuffer(buffer_standdev, CL_TRUE, 0, output_size, &Estanddev[0], NULL, &prof_event_STANDDEV_mem);
		standdev_kernel_time = prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		standdev_single_kernel = prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		standdev_memory_time = prof_event_STANDDEV_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_STANDDEV_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		

		while (Estanddev[1] != 0.0f)
		{
			// call add kernel to sum up all the (value - mean)^2 values
			kernel_add.setArg(0, buffer_standdev);
			kernel_add.setArg(1, buffer_standdev);
			kernel_add.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_STANDDEV);
			queue.enqueueReadBuffer(buffer_standdev, CL_TRUE, 0, output_size, &Estanddev[0], NULL, &prof_event_STANDDEV_mem);
			standdev_kernel_time += prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			standdev_memory_time += prof_event_STANDDEV_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_STANDDEV_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		}

		float variance = (Estanddev[0] / numberOfElements);
		float standDev = sqrt(variance);

		// ********** SORT KERNEL **********
		cl::Kernel kernel_sort = cl::Kernel(program, "sort");
		//cl::Kernel kernel_sort = cl::Kernel(program, "sort_oddeven");
		kernel_sort.setArg(0, buffer_A);
		kernel_sort.setArg(1, buffer_sorted);
		kernel_sort.setArg(2, cl::Local(local_size * sizeof(mytype)));

		cl::Event prof_event_SORT;
		float sort_kernel_time;
		cl::Event prof_event_SORT_mem;
		float sort_memory_time;
		//queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_SORT);
		//queue.enqueueReadBuffer(buffer_sorted, CL_TRUE, 0, sorted_size, &sortedVec[0], NULL, &prof_event_SORT_mem);
		//sort_kernel_time = prof_event_SORT.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_SORT.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		//sort_memory_time = prof_event_SORT_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_SORT_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		//for (int i = 0; i < tempInfo.size() / 2; i++)
		//{
		//	kernel_sort.setArg(0, buffer_sorted);
		//	kernel_sort.setArg(1, buffer_sorted);

		//	queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_SORT);
		//	queue.enqueueReadBuffer(buffer_sorted, CL_TRUE, 0, sorted_size, &sortedVec[0], NULL, &prof_event_SORT_mem);
		//	sort_kernel_time += prof_event_SORT.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_SORT.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		//	sort_memory_time = prof_event_SORT_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_SORT_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		//}
		//// call kernel N/2 times
		
		//float medianTemp = sortedVec[A.size() / 2]; // may be wrong because of extra padded 0's
		//float firstQaut = sortedVec[A.size() / 4];
		//float thirdQuat = sortedVec[3 * A.size() / 4];
		//float interQuatRange = thirdQuat - firstQaut;
		// calculating median and quatiles by taking their values from the sorted vector

		//cout << sortedVec;
		//std::cout << std::endl;


		// *********** OUTPUTS **********

		//std::cout << "Input = " << A << std::endl;
		//std::cout << "Output = " << B << std::endl;

		// outputting all the calculated values
		std::cout << std::endl;
		std::cout << "Total Temp: " << totalTemp << std::endl;
		std::cout << "Average Temp: " << avgTemp << std::endl;
		std::cout << "Max Temp: " << maxTemp << std::endl;
		std::cout << "Min Temp: " << minTemp << std::endl;
		std::cout << std::endl;
		std::cout << "Variance: " << variance << std::endl;
		std::cout << "Standard Deviation: " << standDev << std::endl;
		std::cout << std::endl;
		/*std::cout << "Median Temp: " << medianTemp << std::endl;
		std::cout << "First Quartile: " << firstQaut << std::endl;
		std::cout << "Third Quartile: " << thirdQuat << std::endl;
		std::cout << "Interquartile Range: " << interQuatRange << std::endl;*/
		std::cout << std::endl;

		// outputting profiling info
		std::cout << std::endl;
		std::wcout << "Work Group Size: " << local_size << std::endl;
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		std::cout << "Preferred work group multiple: " << kernel_add.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << std::endl;
		std::cout << "Kernel_AVG:	execution time [ns]: " << AVG_kernel_time << ",		single exuctuion time: " << AVG_single_kernel << std::endl << "		total memory transfer [ns]: " << AVG_memory_time << std::endl << std::endl;
		std::cout << "Kernel_MAX:	execution time [ns]: " << max_kernel_time << ",		single exuction time: " << max_single_kernel << std::endl << "		total memory transfer [ns]: " << max_memory_time << std::endl << std::endl;
		std::cout << "Kernel_MIN:	execution time [ns]: " << min_kernel_time << ",		single exuction time: " << min_single_kernel << std::endl << "		total memory transfer [ns]: " << min_memory_time << std::endl << std::endl;
		std::cout << "Kernel_STANDDEV:execution time [ns]: " << standdev_kernel_time << ",		single kernel time: " << standdev_single_kernel <<  std::endl << "		total memory transfer [ns]: " << standdev_memory_time << std::endl << std::endl;
		//std::cout << "Kernel_SORT:		total execution time [ns]: " << sort_kernel_time << ",		total memory transfer time [ns]: " << sort_memory_time << std::endl;
		//std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) <<  endl;
		std::cout << std::endl;

		std::cout << std::endl;
		std::cout << "Number of Local Values: " << tempInfo.size() << std::endl;
		std::cout << "Length of Vector (may be padded): " << A.size() << std::endl;
		std::cout << std::endl;

		
		// std::cout << "Total TEST = " << totalTempTEST << std::endl;
		// std::wcout << "Average Temp TEST = " << avgTempTEST << std::endl;
		// std::cout << "Number of elements = " << A.size() << std::endl;
		//std::cout << "Max Temp TEST: " << maxTempTEST << std::endl;
		//std::cout << "Min Temp TEST: " << minTempTEST << std::endl;

		//std::cout << "input: " << A << std::endl;
		//std::cout << "Sorted: " << sortedVec << std::endl;

		std::cout << "ASSIGNMENT DONE USING FLOATING POINT VALUES" << std::endl;
		std::cout << std::endl;
	}
	catch (cl::Error err) 
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	// stopping the command window closing
	system("pause");
	return 0;
}
