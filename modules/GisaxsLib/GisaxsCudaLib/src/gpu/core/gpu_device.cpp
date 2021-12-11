#include "gpu/core/gpu_device.h"

#include <memory>

#include "vector_types.h"
#include "gpu/core/gpu_helper.h"
#include "gpu/util/cuda_numerics.h"
#include "gpu/core/gpu_helper.h"
#include "gpu/util/test.h"
#include "gpu/util/util.h"

#include "common/unitcell.h"
#include "common/standard_constants.h"
#include "common/standard_defs.h"

GpuDevice::GpuDevice(gpu_info_t info, int device_id)
	:
	info_(info),
	device_id_(device_id),
	work_status_(WorkStatus::kIdle),
	dev_unitcell_(nullptr),
	fitness_(0),
	dev_fitness_(nullptr),
	event_provider_(device_id_),
	stream_provider_(device_id_),
	memory_provider_f_(device_id_),
	memory_provider_f2_(device_id_),
	memory_provider_f3_(device_id_),
	memory_provider_f4_(device_id_),
	memory_provider_c4_(device_id_),
	memory_provider_i_(device_id_),
	memory_provider_s_(device_id_),
	memory_provider_uchar_(device_id_),
	workers(0),
	complete_runtime_(0),
	kernel_runtime_(0),
	runs_(0)
{
	auto status = curandCreateGenerator(&gen_,
		CURAND_RNG_PSEUDO_DEFAULT);

	if (status != CURAND_STATUS_SUCCESS) {
		printf("Error encountered in generating handle\n");
	}

	gpuErrchk(cudaHostRegister(&fitness_, sizeof(MyType), cudaHostRegisterMapped));
	gpuErrchk(cudaHostGetDevicePointer(&dev_fitness_, &fitness_, 0));

	gpuErrchk(cudaHostRegister(&scale_denom_, sizeof(MyType), cudaHostRegisterMapped));
	gpuErrchk(cudaHostGetDevicePointer(&dev_scale_denom_, &scale_denom_, 0));

	gpuErrchk(cudaHostRegister(&scale_prod_, sizeof(MyType), cudaHostRegisterMapped));
	gpuErrchk(cudaHostGetDevicePointer(&dev_scale_prod_, &scale_prod_, 0));
}

GpuDevice::~GpuDevice()
{	
	gpuErrchk(cudaHostUnregister(&fitness_));
	gpuErrchk(cudaHostUnregister(&scale_denom_));
	gpuErrchk(cudaHostUnregister(&scale_prod_));
	curandDestroyGenerator(gen_);
}

SimData GpuDevice::RunGISAXS(const SimJob& descr, const ImageData *real_img, bool copy_intensities)
{
	complete_timer_.Start();

	//set current device
	int device_id = Bind();

	std::string test_id_xy = "XY_TEST";
	std::string test_id_zcoeffs = "ZCOEFFS_TEST";
	std::string qpar_id = "QPAR";
	std::string q_id = "Q";
	std::string test_id_coeffs = "COEFFS_TEST";

	int qcount = descr.GetQGrid().QCount();
	const auto& qpoints_xy = descr.GetQGrid().QPointsXY();
	const auto& qpoints_z_coeffs = descr.GetQGrid().QPointsZCoeffs();
	const auto& qpar = descr.GetQGrid().QPar();
	const auto& q = descr.GetQGrid().Q();
	const auto& coefficients = descr.GetPropagationCoefficients();
	const auto& current_params = descr.HUnitcell().CurrentParams();

	const auto work_stream = ProvideStream();
	auto start = event_provider_.ProvideEvent(work_stream->Get());
	auto stop = event_provider_.ProvideEvent(work_stream->Get());

	//get cuda memory for work
	std::shared_ptr<GpuMemoryBlock<MyType>> dev_sim_intensities = ProvideMemory<MyType>(qcount);
	std::shared_ptr<GpuMemoryBlock<MyType>> dev_sim_intensities_prep = ProvideMemory<MyType>(qcount);
	std::shared_ptr<GpuMemoryBlock<unsigned char>> dev_sim_intensities_uchar = ProvideMemory<unsigned char>(qcount);
	std::shared_ptr<GpuMemoryBlock<MyType>> dev_partial_sums = ProvideMemory<MyType>(256);
	std::shared_ptr<GpuMemoryBlock<MyType>> dev_sum = ProvideMemory<MyType>(1);
	std::shared_ptr<GpuMemoryBlock<MyType>> dev_max = ProvideMemory<MyType>(1);
	std::shared_ptr<GpuMemoryBlock<MyComplex>> dev_sfs = ProvideMemory<MyComplex>(4 * qcount);
	std::shared_ptr<GpuMemoryBlock<ShapeType>> dev_shapes = ProvideMemory<ShapeType>(descr.HUnitcell().ShapeCount());
	std::shared_ptr<GpuMemoryBlock<MyType>> dev_rands = ProvideMemory<MyType>(descr.HUnitcell().RvsCount() * RANDOM_DRAWS);
	std::shared_ptr<GpuMemoryBlock<MyType>> dev_params = ProvideMemory<MyType>(current_params.size());

	std::shared_ptr<const GpuMemoryBlock<MyComplex>> dev_qpoints_xy = ProvideConstantMemory<MyComplex>(test_id_xy, qpoints_xy);
	std::shared_ptr<const GpuMemoryBlock<MyComplex>> dev_qpoints_z_coeffs = ProvideConstantMemory<MyComplex>(test_id_zcoeffs, qpoints_z_coeffs);
	std::shared_ptr<const GpuMemoryBlock<MyComplex>> dev_qpar = ProvideConstantMemory<MyComplex>(qpar_id, qpar);
	std::shared_ptr<const GpuMemoryBlock<MyComplex>> dev_q = ProvideConstantMemory<MyComplex>(q_id, q);
	std::shared_ptr<const GpuMemoryBlock<MyComplex>> dev_coefficients = ProvideConstantMemory<MyComplex>(test_id_coeffs, coefficients);
	
	

	cudaMemset(dev_sim_intensities->Get(), 0, qcount * sizeof(MyType));
	dev_shapes->InitializeHtD(descr.HUnitcell().Types());
	dev_params->InitializeHtD(current_params);

	
	if (dev_unitcell_ == nullptr)
	{
		auto locations = descr.HUnitcell().Locations();
		auto location_counts = descr.HUnitcell().LocationCounts();
		std::shared_ptr<GpuMemoryBlock<MyType3>> dev_locations = ProvideMemory<MyType3>(locations.size());
		dev_locations->InitializeHtD(locations);

		std::shared_ptr<GpuMemoryBlock<int>> dev_location_counts = ProvideMemory<int>(location_counts.size());
		dev_location_counts->InitializeHtD(location_counts);

		MyType3I repetitions = descr.HUnitcell().Repetitions();
		MyType3 distances = descr.HUnitcell().Distances();
		cudaMalloc(&dev_unitcell_, sizeof(DevUnitcell*));
		Gisaxs::CreateUnitcell(dev_unitcell_, dev_shapes->Get(), dev_shapes->Size(), dev_locations->Get(), dev_location_counts->Get(), RANDOM_DRAWS, repetitions, distances, work_stream->Get());
	}

	start->Record();
	GenerateRandoms(dev_rands->Get(), dev_rands->Size(), 0, 1);
	Gisaxs::Update(dev_rands->Get(), qcount, dev_params->Get(), descr.HUnitcell().ShapeCount(), work_stream->Get());
	Gisaxs::RunSim(dev_qpar->Get(), dev_q->Get(), dev_qpoints_xy->Get(), dev_qpoints_z_coeffs->Get(), qcount, dev_coefficients->Get(), dev_sim_intensities->Get(), descr.HUnitcell().ShapeCount(), dev_sfs->Get(),  work_stream->Get());
	
	max(dev_sim_intensities->Get(), dev_sim_intensities->Size(), dev_partial_sums->Get(), dev_max->Get(), work_stream->Get());

	Preprocess(dev_sim_intensities->Get(), dev_sim_intensities->Size(), dev_sim_intensities_prep->Get(), dev_max->Get(), work_stream->Get());
	Normalize(dev_sim_intensities_prep->Get(), dev_sim_intensities_prep->Size(), dev_sim_intensities_uchar->Get(), work_stream->Get());

	float scale = 1;
	if (real_img != nullptr)
	{
		std::shared_ptr<const GpuMemoryBlock<MyType>> dev_real_intensities = ProvideConstantMemory(real_img->Id(), real_img->Intensities());
		
		SumReduce(dev_real_intensities->Get(), dev_real_intensities->Size(), dev_partial_sums->Get(), dev_scale_prod_, work_stream->Get());
		SumReduce(dev_sim_intensities->Get(), dev_sim_intensities->Size(), dev_partial_sums->Get(), dev_scale_denom_, work_stream->Get());
		
		gpuErrchk(cudaStreamSynchronize(work_stream->Get()));
		scale = scale_prod_ / scale_denom_;

		ScaledDiffSum(dev_real_intensities->Get(), dev_sim_intensities->Get(), dev_real_intensities->Size(), dev_partial_sums->Get(), dev_fitness_, scale, work_stream->Get());
	}
	stop->Record();
	gpuErrchk(cudaDeviceSynchronize());

	float milliseconds = 0;
	
	gpuErrchk(cudaEventElapsedTime(&milliseconds, start->Get(), stop->Get()));
	kernel_runtime_ += milliseconds;
	
	dev_sim_intensities->Unlock();
	dev_partial_sums->Unlock();
	dev_params->Unlock();
	dev_sum->Unlock();
	dev_rands->Unlock();
	dev_max->Unlock();
	dev_shapes->Unlock();
	dev_sfs->Unlock();

	start->Unlock();
	stop->Unlock();

	work_stream->Unlock();

	complete_timer_.End();
	complete_runtime_ += complete_timer_.Duration();
	runs_ += 1;

	if (copy_intensities)
	{
		std::vector<unsigned char> copied_intensities(dev_sim_intensities_uchar->Size());
		gpuErrchk(cudaMemcpy(&copied_intensities[0], dev_sim_intensities_uchar->Get(), dev_sim_intensities_uchar->Size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		return { fitness_, std::vector<float>(), copied_intensities, descr.GetQGrid().Qx(), descr.GetQGrid().Qy(), descr.GetQGrid().Qz(),  descr.GetQGrid().Resolution(), scale};
	}
	dev_sim_intensities_prep->Unlock();
	dev_sim_intensities_uchar->Unlock();
	return  { fitness_, {}, std::vector<unsigned char>(), {}, {}, {}, {0,0}, scale };
}

int GpuDevice::Bind() const
{
	gpuErrchk(cudaSetDevice(device_id_));

	return device_id_;
}

const WorkStatus GpuDevice::Status() const
{
	return work_status_;
}

void GpuDevice::SetStatus(WorkStatus status) const
{
	work_status_ = status;
}

int GpuDevice::DeviceID() const
{
	return device_id_;
}

void GpuDevice::GenerateRandoms(float *rands, int size, float mean, float stddev) const
{
	curandGenerateNormal(gen_, rands, size, mean, stddev);
}

std::shared_ptr<Stream> GpuDevice::ProvideStream()
{
	return stream_provider_.ProvideStream();
}

void GpuDevice::UnlockAllMemory()
{
	memory_provider_f_.UnlockAll();
	memory_provider_f2_.UnlockAll();
	memory_provider_f3_.UnlockAll();
	memory_provider_f4_.UnlockAll();

	memory_provider_c4_.UnlockAll();
	memory_provider_i_.UnlockAll();
	memory_provider_s_.UnlockAll();
	memory_provider_uchar_.UnlockAll();
}

void GpuDevice::UnlockAllEvents()
{
	event_provider_.UnlockAll();
}

void GpuDevice::UnlockAllStreams()
{
	stream_provider_.UnlockAll();
}

void GpuDevice::ResetTimers()
{
	runs_ = 0;
	complete_runtime_ = 0;
	kernel_runtime_ = 0;
}

void GpuDevice::CleanUp()
{
	
	if (dev_unitcell_ != nullptr)
	{ 
		auto delete_stream = stream_provider_.ProvideStream()->Get();
		Gisaxs::DestroyUnitcell(delete_stream);

		gpuErrchk(cudaStreamSynchronize(delete_stream));
		gpuErrchk(cudaFree(dev_unitcell_));

		dev_unitcell_ = nullptr;
	}


	memory_provider_f_.FreeAllMemory();
	memory_provider_f2_.FreeAllMemory();
	memory_provider_f3_.FreeAllMemory();
	memory_provider_f4_.FreeAllMemory();
	memory_provider_c4_.FreeAllMemory();
	memory_provider_i_.FreeAllMemory();
	memory_provider_s_.FreeAllMemory();
	memory_provider_uchar_.FreeAllMemory();

	event_provider_.DestroyAllEvents();
	stream_provider_.DestroyAllStreams();

	ResetTimers();
}

double GpuDevice::AverageKernelTime() const
{
	return kernel_runtime_ / runs_;
}

double GpuDevice::AverageFullTime() const
{
	return complete_runtime_ / runs_;
}

double GpuDevice::KernelTime() const
{
	return kernel_runtime_;
}

double GpuDevice::FullTime() const
{
	return complete_runtime_;
}

int GpuDevice::Runs() const
{
	return runs_;
}

std::string GpuDevice::Name() const
{
	return std::string(info_.name);
}