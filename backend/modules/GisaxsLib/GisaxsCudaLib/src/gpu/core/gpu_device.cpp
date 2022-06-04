#include "gpu/core/gpu_device.h"

#include <memory>

#include "gpu/core/gpu_helper.h"
#include "gpu/util/util.h"

#include "common/standard_constants.h"
#include "common/standard_defs.h"
#include "gpu/core/gpu_memory_provider_v2.h"
#include "common/propagation_coefficients.h"
#include "gpu/util/conversion_helper.h"


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
        workers(0),
        complete_runtime_(0),
        kernel_runtime_(0),
        runs_(0) {


    gpuErrchk(cudaHostRegister(&fitness_, sizeof(MyType), cudaHostRegisterMapped));
    gpuErrchk(cudaHostGetDevicePointer(&dev_fitness_, &fitness_, 0));

    gpuErrchk(cudaHostRegister(&scale_denom_, sizeof(MyType), cudaHostRegisterMapped));
    gpuErrchk(cudaHostGetDevicePointer(&dev_scale_denom_, &scale_denom_, 0));

    gpuErrchk(cudaHostRegister(&scale_prod_, sizeof(MyType), cudaHostRegisterMapped));
    gpuErrchk(cudaHostGetDevicePointer(&dev_scale_prod_, &scale_prod_, 0));
}

GpuDevice::~GpuDevice() {
    gpuErrchk(cudaHostUnregister(&fitness_));
    gpuErrchk(cudaHostUnregister(&scale_denom_));
    gpuErrchk(cudaHostUnregister(&scale_prod_));
}

SimData GpuDevice::RunGISAXS(const SimJob &descr, const ImageData *real_img, bool copy_intensities) {
    complete_timer_.Start();

    //set current device
    int device_id = Bind();
    int qcount = descr.ExperimentInfo().DetectorConfig().PixelCount();

    auto unitcell = descr.ExperimentInfo().Unitcell();

    const auto coefficients = GpuConversionHelper::Convert(PropagationCoefficientsCpu::PropagationCoeffsTopBuried(
            descr.ExperimentInfo().SampleConfig(), descr.ExperimentInfo().DetectorConfig(),
            descr.ExperimentInfo().BeamConfig()));
    const auto &current_params = unitcell.Parameters();

    const auto work_stream = ProvideStream();
    auto start = event_provider_.ProvideEvent(work_stream->Get());
    auto stop = event_provider_.ProvideEvent(work_stream->Get());

    //get cuda memory for work
    GpuMemoryProviderV2 memoryProviderV2(0);
    MemoryBlock<unsigned char> dev_sim_intensities_uchar = memoryProviderV2.RequestMemory<unsigned char>(qcount);
    MemoryBlock<MyType> dev_sim_intensities = memoryProviderV2.RequestMemory<MyType>(qcount);
    MemoryBlock<MyType> dev_sim_intensities_prep = memoryProviderV2.RequestMemory<MyType>(qcount);
    MemoryBlock<MyType> dev_partial_sums = memoryProviderV2.RequestMemory<MyType>(256);
    MemoryBlock<MyType> dev_max = memoryProviderV2.RequestMemory<MyType>(1);
    MemoryBlock<MyComplex> dev_sfs = memoryProviderV2.RequestMemory<MyComplex>(4 * qcount);
    MemoryBlock<ShapeTypeV2> dev_shapes = memoryProviderV2.RequestMemory<ShapeTypeV2>(unitcell.ShapeTypes().size());
    MemoryBlock<MyType> dev_rands = memoryProviderV2.RequestMemory<MyType>(
            unitcell.Parameters().size() * 2 * RANDOM_DRAWS);
    MemoryBlock<MyType2> dev_params = memoryProviderV2.RequestMemory<MyType2>(current_params.size());

    MemoryBlock<MyComplex> dev_coefficients = memoryProviderV2.RequestConstantMemory(ConstantMemoryId::QGRID_COEFFS,
                                                                                     coefficients);
    MemoryBlock<MyComplex> container_xy = memoryProviderV2.RequestMemory<MyComplex>(2 * qcount);
    MemoryBlock<MyComplex> container_zcoeffs = memoryProviderV2.RequestMemory<MyComplex>(4 * qcount);
    MemoryBlock<MyComplex> container_qpar = memoryProviderV2.RequestMemory<MyComplex>(qcount);
    MemoryBlock<MyComplex> container_q = memoryProviderV2.RequestMemory<MyComplex>(4 * qcount);
    MemoryBlock<MyComplex> container_coeffs = memoryProviderV2.RequestMemory<MyComplex>(qcount);
    MemoryBlock<MyType> container_alpha_fs = memoryProviderV2.RequestMemory<MyType>(qcount);
    MemoryBlock<MyType> container_theta_fs = memoryProviderV2.RequestMemory<MyType>(qcount);
    MemoryBlock<MyType> container_qx = memoryProviderV2.RequestMemory<MyType>(qcount);
    MemoryBlock<MyType> container_qy = memoryProviderV2.RequestMemory<MyType>(qcount);
    MemoryBlock<MyType> container_qz = memoryProviderV2.RequestMemory<MyType>(qcount);


    GpuQGrid::GpuQGridContainer container{container_xy.Get(), container_zcoeffs.Get(), container_qpar.Get(),
                                          container_q.Get(), container_coeffs.Get(), container_alpha_fs.Get(),
                                          container_theta_fs.Get(), container_qx.Get(), container_qy.Get(),
                                          container_qz.Get()};

    auto alpha_i = descr.ExperimentInfo().BeamConfig().AlphaI();
    auto k0 = descr.ExperimentInfo().BeamConfig().K0();
    auto pixelsize = descr.ExperimentInfo().DetectorConfig().Pixelsize();
    auto sample_distance = descr.ExperimentInfo().DetectorConfig().SampleDistance();
    auto direct_beam = descr.ExperimentInfo().DetectorConfig().Directbeam();
    auto detector_width = descr.ExperimentInfo().DetectorConfig().Resolution().x;
    auto detector_height = descr.ExperimentInfo().DetectorConfig().Resolution().y;

    GpuQGrid::CreateQGridFull(alpha_i, k0, pixelsize, sample_distance, GpuConversionHelper::Convert(direct_beam),
                              detector_width, detector_height,
                              container, work_stream->Get());

    gpuErrchk(cudaDeviceSynchronize());
    cudaMemset(dev_sim_intensities.Get(), 0, qcount * sizeof(MyType));
    dev_shapes.InitializeHtD(unitcell.ShapeTypes());
    dev_params.InitializeHtD(GpuConversionHelper::Convert(current_params));


    if (dev_unitcell_ == nullptr) {
        auto locations = unitcell.Positions();

        auto location_counts = unitcell.LocationCounts();
        MemoryBlock<MyType3> dev_locations = memoryProviderV2.RequestMemory<MyType3>(locations.size());
        dev_locations.InitializeHtD(GpuConversionHelper::Convert(locations));

        MemoryBlock<int> dev_location_counts = memoryProviderV2.RequestMemory<int>(location_counts.size());
        dev_location_counts.InitializeHtD(location_counts);

        MyType3I repetitions = GpuConversionHelper::Convert(unitcell.Repetitions());
        MyType3 distances = GpuConversionHelper::Convert(unitcell.Translation());
        cudaMalloc(&dev_unitcell_, sizeof(DevUnitcell *));
        Gisaxs::CreateUnitcell(dev_unitcell_, dev_shapes.Get(), dev_shapes.Size(), dev_locations.Get(),
                               dev_location_counts.Get(), RANDOM_DRAWS, repetitions, distances, work_stream->Get());
    }

    start->Record();
    random_generator_.GenerateRandoms(dev_rands.Get(), dev_rands.Size(), 0, 1);
    Gisaxs::Update(dev_rands.Get(), qcount, dev_params.Get(), unitcell.ShapeTypes().size(), work_stream->Get());

    Gisaxs::RunSim(container.dev_qpar, container.dev_q, container.dev_qpoints_xy, container.dev_qpoints_z_coeffs,
                   qcount,
                   dev_coefficients.Get(), dev_sim_intensities.Get(), unitcell.ShapeTypes().size(), dev_sfs.Get(),
                   work_stream->Get());

    CalculateMaximumIntensity(dev_sim_intensities.Get(), dev_sim_intensities.Size(), dev_partial_sums.Get(),
                              dev_max.Get(),
                              work_stream->Get());

    Preprocess(dev_sim_intensities.Get(), dev_sim_intensities.Size(), dev_sim_intensities_prep.Get(), dev_max.Get(),
               work_stream->Get());
    Normalize(dev_sim_intensities_prep.Get(), dev_sim_intensities_prep.Size(), dev_sim_intensities_uchar.Get(),
              work_stream->Get());
    gpuErrchk(cudaDeviceSynchronize());
    float scale = 1;
    if (real_img != nullptr) {


        MemoryBlock<MyType> dev_real_intensities = memoryProviderV2.RequestMemory<MyType>(
                real_img->LineProfiles()[0].intensities.size());
        dev_real_intensities.InitializeHtD(real_img->LineProfiles()[0].intensities);

        SumReduce(dev_real_intensities.Get(), dev_real_intensities.Size(), dev_partial_sums.Get(), dev_scale_prod_,
                  work_stream->Get());
        SumReduce(dev_sim_intensities.Get(), dev_sim_intensities.Size(), dev_partial_sums.Get(), dev_scale_denom_,
                  work_stream->Get());

        gpuErrchk(cudaStreamSynchronize(work_stream->Get()));
        scale = scale_prod_ / scale_denom_;

        ScaledDiffSum(dev_real_intensities.Get(), dev_sim_intensities.Get(), dev_real_intensities.Size(),
                      dev_partial_sums.Get(), dev_fitness_, scale, work_stream->Get());
    }
    stop->Record();
    gpuErrchk(cudaDeviceSynchronize());

    float milliseconds = 0;

    gpuErrchk(cudaEventElapsedTime(&milliseconds, start->Get(), stop->Get()));
    kernel_runtime_ += milliseconds;

    start->Unlock();
    stop->Unlock();

    work_stream->Unlock();

    complete_timer_.End();
    complete_runtime_ += complete_timer_.Duration();
    runs_ += 1;

    if (copy_intensities) {
        std::vector<unsigned char> copied_normalized_intensities(dev_sim_intensities_uchar.Size());
        gpuErrchk(cudaMemcpy(&copied_normalized_intensities[0], dev_sim_intensities_uchar.Get(),
                             dev_sim_intensities_uchar.Size() * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        std::vector<MyType> copied_intensities(dev_sim_intensities.Size());
        gpuErrchk(cudaMemcpy(&copied_intensities[0], dev_sim_intensities.Get(),
                             dev_sim_intensities.Size() * sizeof(MyType), cudaMemcpyDeviceToHost));

        return {fitness_, {copied_intensities.begin(), copied_intensities.end()}, copied_normalized_intensities,
                container_qx.CopyToHost(),
                container_qy.CopyToHost(), container_qz.CopyToHost(),
                descr.ExperimentInfo().DetectorConfig().Resolution(), scale};
    }

    memoryProviderV2.UnlockAll();
    return {fitness_, {}, std::vector<unsigned char>(), {}, {}, {}, {0, 0}, scale};
}

int GpuDevice::Bind() const {
    gpuErrchk(cudaSetDevice(device_id_));

    return device_id_;
}

WorkStatus GpuDevice::Status() const {
    return work_status_;
}

void GpuDevice::SetStatus(WorkStatus status) const {
    work_status_ = status;
}

int GpuDevice::DeviceID() const {
    return device_id_;
}

std::shared_ptr<Stream> GpuDevice::ProvideStream() {
    return stream_provider_.ProvideStream();
}

void GpuDevice::UnlockAllMemory() {

}

void GpuDevice::UnlockAllEvents() {
    event_provider_.UnlockAll();
}

void GpuDevice::UnlockAllStreams() {
    stream_provider_.UnlockAll();
}

void GpuDevice::ResetTimers() {
    runs_ = 0;
    complete_runtime_ = 0;
    kernel_runtime_ = 0;
}

void GpuDevice::CleanUp() {

    if (dev_unitcell_ != nullptr) {
        auto delete_stream = stream_provider_.ProvideStream()->Get();
        Gisaxs::DestroyUnitcell(delete_stream);

        gpuErrchk(cudaStreamSynchronize(delete_stream));
        gpuErrchk(cudaFree(dev_unitcell_));

        dev_unitcell_ = nullptr;
    }

    event_provider_.DestroyAllEvents();
    stream_provider_.DestroyAllStreams();

    ResetTimers();
}

double GpuDevice::AverageKernelTime() const {
    return kernel_runtime_ / runs_;
}

double GpuDevice::AverageFullTime() const {
    return complete_runtime_ / runs_;
}

double GpuDevice::KernelTime() const {
    return kernel_runtime_;
}

double GpuDevice::FullTime() const {
    return complete_runtime_;
}

int GpuDevice::Runs() const {
    return runs_;
}

std::string GpuDevice::Name() const {
    return info_.name;
}