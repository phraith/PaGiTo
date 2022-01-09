#include "gtest/gtest.h"

#include <vector>
#include <fstream>

#include <gpu/core/gpu_memory_provider.h>
#include <gpu/util/util.h>
#include <util/hardware_information.h>

class MemoryProviderTest : public ::testing::TestWithParam<int> {};

TEST_P(MemoryProviderTest, SumUp)
{
	/*std::shared_ptr<HardwareInformation> hw_info = std::make_shared<HardwareInformation>();

	auto& devices = hw_info->GpuInfo();

	if (devices.size() == 0)
		return;

	auto& dev0 = devices.at(0);

	int array_size = GetParam();

	std::vector<my_type2> h_data(array_size, 1);
	
	const auto &dev_partial_sums = dev0.ProvideMemory<my_type2>(256);
	const auto &dev_sum = dev0.ProvideMemory<my_type2>(1);

	const auto& dev_data = dev0.ProvideMemory<my_type2>(array_size);
	dev_data->InitializeHtD(h_data);

	sum(dev_data->Get(), dev_data->Size(), dev_partial_sums->Get(), dev_sum->Get(), dev0.WorkStream());
	gpuErrchk(cudaStreamSynchronize(dev0.WorkStream()));

	my_type2 h_sum;
	cudaMemcpy(&h_sum, dev_sum->Get(), sizeof(my_type2), cudaMemcpyDeviceToHost);

	EXPECT_EQ(h_sum, array_size * 1.0);*/
}

INSTANTIATE_TEST_SUITE_P(MemoryProviderTester,
	MemoryProviderTest,
	::testing::Values(1024, 32768, 131072));