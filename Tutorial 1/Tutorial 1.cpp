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

		AddSources(sources, "my_kernels_1.cl");
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

		typedef int mytype;

		//host - input
		std::vector<mytype> A;

		for (int i = 0; i < tempInfo.size(); i++)
		{
			A.push_back(tempInfo[i] * 10);
		}
		// times values by 10 (to make values int)

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 128; // workgroup size

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
		std::vector<mytype> Bavg(1);
		std::vector<mytype> Cmax(1);
		std::vector<mytype> Dmin(1);
		std::vector<mytype> Estanddev(1);
		std::vector<mytype> sortedVec(input_elements);
		// output vectors for the results (most only one value)

		size_t output_size = Bavg.size() * sizeof(mytype);//size in bytes
		size_t sorted_size = A.size() * sizeof(mytype);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size); // buffer for average
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size); // buffer for max
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size); // buffer for min
		cl::Buffer buffer_standdev(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_sorted(context, CL_MEM_READ_WRITE, sorted_size);

		//copy arrays to buffer and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);//zero C buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_size);//zero D buffer on device memory
		queue.enqueueFillBuffer(buffer_standdev, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_sorted, 0, 0, sorted_size);

		// Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_avg = cl::Kernel(program, "reduce_add_4");
		kernel_avg.setArg(0, buffer_A);
		kernel_avg.setArg(1, buffer_B);
		kernel_avg.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Kernel kernel_max = cl::Kernel(program, "reduce_max_4");
		kernel_max.setArg(0, buffer_A);
		kernel_max.setArg(1, buffer_C);
		kernel_max.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Kernel kernel_min = cl::Kernel(program, "reduce_min_4");
		kernel_min.setArg(0, buffer_A);
		kernel_min.setArg(1, buffer_D);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size

		cl::Kernel kernel_standDev = cl::Kernel(program, "reduce_standDev_4");
		kernel_standDev.setArg(0, buffer_A);
		kernel_standDev.setArg(1, buffer_standdev);
		kernel_standDev.setArg(2, buffer_B); // pass through the output from reduce add to get mean for stand dev
		kernel_standDev.setArg(3, cl::Local(local_size * sizeof(mytype)));

		cl::Kernel kernel_sort = cl::Kernel(program, "sort");
		kernel_sort.setArg(0, buffer_A);
		kernel_sort.setArg(1, buffer_sorted);
		kernel_sort.setArg(2, cl::Local(local_size * sizeof(mytype)));

		//create profiling events
		cl::Event prof_event_AVG; cl::Event prof_event_AVG_mem;
		cl::Event prof_event_MAX; cl::Event prof_event_MAX_mem;
		cl::Event prof_event_MIN; cl::Event prof_event_MIN_mem;
		cl::Event prof_event_STANDDEV; cl::Event prof_event_STANDDEV_mem;
		cl::Event prof_event_SORT; cl::Event prof_event_SORT_mem;

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_avg, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_AVG);
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_MAX);
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_MIN);
		queue.enqueueNDRangeKernel(kernel_standDev, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_STANDDEV);
		//queue.enqueueNDRangeKernel(kernel_sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event_SORT);

		//Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &Bavg[0], NULL, &prof_event_AVG_mem);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_size, &Cmax[0], NULL, &prof_event_MAX_mem);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_size, &Dmin[0], NULL, &prof_event_MIN_mem);
		queue.enqueueReadBuffer(buffer_standdev, CL_TRUE, 0, output_size, &Estanddev[0], NULL, &prof_event_STANDDEV_mem);
		//queue.enqueueReadBuffer(buffer_sorted, CL_TRUE, 0, sorted_size, &sortedVec[0], NULL, &prof_event_SORT_mem);

		float totalTemp = Bavg[0] / 10.0f;
		float avgTemp = totalTemp / tempInfo.size();
		// calcualting avg temp (divide by 10 to account for the int) (dividing by original vector size so padding deosnt affect it)

		float maxTemp = Cmax[0] / 10.0f;
		float minTemp = Dmin[0] / 10.0f;
		// getting min and max, divide by 10 to account for converting to int by x10

		float variance = (Estanddev[0] / A.size()) / 10.0f;
		// divide by 10 to account for int, (already divided by 10 in kernel) (have to divide by 10 twice due to sqauring in stand dev formula)
		float standDev = sqrt(variance);

		float medianTemp = sortedVec[A.size() / 2] / 10.0f; // may be wrong because of extra padded 0's
		float firstQaut = sortedVec[A.size() / 4] / 10.0f;
		float thirdQuat = sortedVec[3 * A.size() / 4] / 10.0f;
		float interQuatRange = thirdQuat - firstQaut;
		// calculating median and quatiles by taking their values from the sorted vector

		//std::cout << "Input = " << A << std::endl;
		//std::cout << "Output = " << B << std::endl;

		// outputting all the calculated values
		std::cout << std::endl;
		std::cout << "Average Temp: " << avgTemp << std::endl;
		std::cout << "Max Temp: " << maxTemp << std::endl;
		std::cout << "Min Temp: " << minTemp << std::endl;
		std::cout << std::endl;
		std::cout << "Variance: " << variance << std::endl;
		std::cout << "Standard Deviation: " << standDev << std::endl;
		std::cout << std::endl;
		std::cout << "Median Temp: " << medianTemp << std::endl;
		std::cout << "First Quartile: " << firstQaut << std::endl;
		std::cout << "Third Quartile: " << thirdQuat << std::endl;
		std::cout << "Interquartile Range: " << interQuatRange << std::endl;
		std::cout << std::endl;

		// outputting profiling info
		std::cout << std::endl;
		std::wcout << "Work Group Size: " << local_size << std::endl;
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		std::cout << "Preferred work group size: " << kernel_min.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << std::endl;
		std::cout << "Kernel_AVG execution time [ns]: " << prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_AVG.getProfilingInfo<CL_PROFILING_COMMAND_START>() << ",			";
		std::cout << "Kernel_AVG memory transfer time [ns]: " << prof_event_AVG_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_AVG_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Kernel_MAX execution time [ns]: " << prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MAX.getProfilingInfo<CL_PROFILING_COMMAND_START>() << ",			";
		std::cout << "Kernel_MAX memory transfer time [ns]: " << prof_event_MAX_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MAX_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Kernel_MIN execution time [ns]: " << prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MIN.getProfilingInfo<CL_PROFILING_COMMAND_START>() << ",			";
		std::cout << "Kernel_MIN memory transfer time [ns]: " << prof_event_MIN_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_MIN_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		std::cout << "Kernel_STANDDEV execution time [ns]: " << prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_STANDDEV.getProfilingInfo<CL_PROFILING_COMMAND_START>() << ",		";
		std::cout << "Kernel_STANDDEV memory transfer time [ns]: " << prof_event_STANDDEV_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_STANDDEV_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;

		//std::cout << "Kernel_SORT execution time [ns]: " << prof_event_SORT.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_SORT.getProfilingInfo<CL_PROFILING_COMMAND_START>() << ",		";
		//std::cout << "Kernel_SORT memory transfer time [ns]: " << prof_event_SORT_mem.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event_SORT_mem.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		//std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) <<  endl;
		std::cout << std::endl;

		std::cout << std::endl;
		std::cout << "Number of Local Values: " << tempInfo.size() << std::endl;
		std::cout << "Length of Vector (may be padded): " << A.size() << std::endl;
		std::cout << std::endl;

		std::cout << std::endl;
		// std::cout << "Total TEST = " << totalTempTEST << std::endl;
		// std::wcout << "Average Temp TEST = " << avgTempTEST << std::endl;
		// std::cout << "Number of elements = " << A.size() << std::endl;
		
		//std::cout << "input: " << A << std::endl;
		//std::cout << "Sorted: " << sortedVec << std::endl;

		std::cout << "ASSIGNMENT USING INTEGER VALUES" << std::endl;
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
