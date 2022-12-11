//
// Created by Phil on 01.03.2022.
//

#include "gpu/core/gpu_device_v2.h"

#include <memory>
#include <spdlog/spdlog.h>

#include "gpu/core/gpu_helper.h"
#include "gpu/util/cuda_numerics.h"
#include "gpu/util/test.h"
#include "gpu/util/util.h"

#include "common/standard_constants.h"
#include "common/standard_defs.h"
#include "gpu/core/gpu_memory_provider_v2.h"
#include "common/propagation_coefficients.h"
#include "gpu/core/gisaxs_functions_v2.h"
#include "gpu/util/conversion_helper.h"

namespace GpuDeviceV2 {
    GpuDeviceV2::GpuDeviceV2(gpu_info_t info, int device_id)
            :
            info_(info),
            device_id_(device_id),
            work_status_(WorkStatus::kIdle),
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

    GpuDeviceV2::~GpuDeviceV2() {
        gpuErrchk(cudaHostUnregister(&fitness_));
        gpuErrchk(cudaHostUnregister(&scale_denom_));
        gpuErrchk(cudaHostUnregister(&scale_prod_));
    }

    SimData GpuDeviceV2::RunGISAXS(const SimJob &sim_job, const ImageData *real_img, bool copy_intensities) {
        Timer local_timer;

        complete_timer_.Start();

        //set current device
        int device_id = Bind();
        auto detector_positions = sim_job.DetectorPositions();

        int qcount = detector_positions.size() > 0 ? detector_positions.size()
                                                   : sim_job.ExperimentInfo().DetectorConfig().PixelCount();

        auto unitcell = sim_job.ExperimentInfo().Unitcell();

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
        MemoryBlock<MyType2> dev_params = memoryProviderV2.RequestMemory<MyType2>(unitcell.Parameters().size());
        MemoryBlock<int> dev_parameters_indices = memoryProviderV2.RequestMemory<int>(
                unitcell.ParameterIndices().size());
        MemoryBlock<MyType3> dev_positions = memoryProviderV2.RequestMemory<MyType3>(unitcell.Positions().size());
        MemoryBlock<int> dev_positions_indices = memoryProviderV2.RequestMemory<int>(unitcell.PositionIndices().size());
        MemoryBlock<MyComplex> dev_coefficients = memoryProviderV2.RequestMemory<MyComplex>(4 * qcount);

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

        MemoryBlock<MyType2I> dev_detector_positions = memoryProviderV2.RequestMemory<MyType2I>(
                detector_positions.size());
        dev_detector_positions.InitializeHtD(GpuConversionHelper::Convert(detector_positions));


        local_timer.Start();
        if (detector_positions.size() == 0) {
            GisaxsV2::CalculatePropagationCoefficientsTopBuriedFull(qcount,
                                                                    GpuConversionHelper::Convert(
                                                                            sim_job.ExperimentInfo().DetectorConfig().Resolution()),
                                                                    GpuConversionHelper::Convert(
                                                                            sim_job.ExperimentInfo().BeamConfig().BeamDirection()),
                                                                    sim_job.ExperimentInfo().DetectorConfig().Pixelsize(),
                                                                    sim_job.ExperimentInfo().DetectorConfig().SampleDistance(),
                                                                    sim_job.ExperimentInfo().BeamConfig().K0(),
                                                                    GpuConversionHelper::Convert(
                                                                            sim_job.ExperimentInfo().SampleConfig().Layers().at(
                                                                                    0).N2MinusOne()),
                                                                    sim_job.ExperimentInfo().BeamConfig().AlphaI(),
                                                                    dev_coefficients.Get(),
                                                                    work_stream->Get());
        } else {
            GisaxsV2::CalculatePropagationCoefficientsTopBuried(qcount,
                                                                dev_detector_positions.Get(),
                                                                GpuConversionHelper::Convert(
                                                                        sim_job.ExperimentInfo().DetectorConfig().Resolution()),
                                                                GpuConversionHelper::Convert(
                                                                        sim_job.ExperimentInfo().BeamConfig().BeamDirection()),
                                                                sim_job.ExperimentInfo().DetectorConfig().Pixelsize(),
                                                                sim_job.ExperimentInfo().DetectorConfig().SampleDistance(),
                                                                sim_job.ExperimentInfo().BeamConfig().K0(),
                                                                GpuConversionHelper::Convert(
                                                                        sim_job.ExperimentInfo().SampleConfig().Layers().at(
                                                                                0).N2MinusOne()),
                                                                sim_job.ExperimentInfo().BeamConfig().AlphaI(),
                                                                dev_coefficients.Get(),
                                                                work_stream->Get());
        }

        gpuErrchk(cudaDeviceSynchronize());

        local_timer.End();
        spdlog::info("Calculating coefficients took {} ms", local_timer.Duration());


        GpuQGrid::GpuQGridContainer container{container_xy.Get(), container_zcoeffs.Get(), container_qpar.Get(),
                                              container_q.Get(), container_coeffs.Get(), container_alpha_fs.Get(),
                                              container_theta_fs.Get(), container_qx.Get(), container_qy.Get(),
                                              container_qz.Get()};

        auto alpha_i = sim_job.ExperimentInfo().BeamConfig().AlphaI();
        auto k0 = sim_job.ExperimentInfo().BeamConfig().K0();
        auto pixelsize = sim_job.ExperimentInfo().DetectorConfig().Pixelsize();
        auto sample_distance = sim_job.ExperimentInfo().DetectorConfig().SampleDistance();
        auto direct_beam = sim_job.ExperimentInfo().DetectorConfig().Directbeam();
        auto detector_width = sim_job.ExperimentInfo().DetectorConfig().Resolution().x;
        auto detector_height = sim_job.ExperimentInfo().DetectorConfig().Resolution().y;

        if (detector_positions.size() == 0) {
            GpuQGrid::CreateQGridFull(alpha_i, k0, pixelsize, sample_distance,
                                      GpuConversionHelper::Convert(direct_beam),
                                      detector_width, detector_height,
                                      container, work_stream->Get());
        } else {

            GpuQGrid::CreateQGrid(alpha_i, k0, pixelsize, sample_distance, dev_detector_positions.Get(),
                                  GpuConversionHelper::Convert(direct_beam),
                                  detector_width, qcount,
                                  container, work_stream->Get());
        }

        gpuErrchk(cudaDeviceSynchronize());
        cudaMemset(dev_sim_intensities.Get(), 0, qcount * sizeof(MyType));
        dev_shapes.InitializeHtD(unitcell.ShapeTypes());
        dev_params.InitializeHtD(GpuConversionHelper::Convert(unitcell.Parameters()));
        dev_parameters_indices.InitializeHtD(unitcell.ParameterIndices());
        dev_positions.InitializeHtD(GpuConversionHelper::Convert(unitcell.Positions()));
        dev_positions_indices.InitializeHtD(unitcell.PositionIndices());

        start->Record();
        random_generator_.GenerateRandoms(dev_rands.Get(), dev_rands.Size(), 0, 1);

        GisaxsV2::DeviceFlatUnitcell dev_unitcell{dev_params.Get(), dev_parameters_indices.Get(), dev_positions.Get(),
                                                  dev_positions_indices.Get(), dev_shapes.Get(),
                                                  GpuConversionHelper::Convert(unitcell.Repetitions()),
                                                  GpuConversionHelper::Convert(unitcell.Translation())};

        GisaxsV2::RunSim(container.dev_qpar, container.dev_q, container.dev_qpoints_xy, container.dev_qpoints_z_coeffs,
                         qcount, dev_coefficients.Get(), dev_sim_intensities.Get(), unitcell.ShapeTypes().size(),
                         dev_sfs.Get(), dev_unitcell, dev_rands.Get(), work_stream->Get());

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
            //memoryProviderV2.UnlockAll();
            return {fitness_, {copied_intensities.begin(), copied_intensities.end()}, copied_normalized_intensities,
                    container_qx.CopyToHost(),
                    container_qy.CopyToHost(), container_qz.CopyToHost(), 0};
        }

        //memoryProviderV2.UnlockAll();
        return {fitness_, {}, std::vector<unsigned char>(), {}, {}, {}, 0};//scale};
    }

    int GpuDeviceV2::Bind() const {
        gpuErrchk(cudaSetDevice(device_id_));
        return device_id_;
    }

    WorkStatus GpuDeviceV2::Status() const {
        return work_status_;
    }

    void GpuDeviceV2::SetStatus(WorkStatus status) const {
        spdlog::info("Gpu: status {}", WorkStatusToStr(status));
        work_status_ = status;
    }

    std::string GpuDeviceV2::WorkStatusToStr(WorkStatus status) const {
        switch (status) {
            case WorkStatus::kIdle:
                return "idle";
            case WorkStatus::kWorking:
                return "working";
        }
        return "";
    }

    int GpuDeviceV2::DeviceID() const {
        return device_id_;
    }

    std::shared_ptr<Stream> GpuDeviceV2::ProvideStream() {
        return stream_provider_.ProvideStream();
    }

    void GpuDeviceV2::UnlockAllMemory() {

    }

    void GpuDeviceV2::UnlockAllEvents() {
        event_provider_.UnlockAll();
    }

    void GpuDeviceV2::UnlockAllStreams() {
        stream_provider_.UnlockAll();
    }

    void GpuDeviceV2::ResetTimers() {
        runs_ = 0;
        complete_runtime_ = 0;
        kernel_runtime_ = 0;
    }

    void GpuDeviceV2::CleanUp() {
        event_provider_.DestroyAllEvents();
        stream_provider_.DestroyAllStreams();

        ResetTimers();
    }

    double GpuDeviceV2::AverageKernelTime() const {
        return kernel_runtime_ / runs_;
    }

    double GpuDeviceV2::AverageFullTime() const {
        return complete_runtime_ / runs_;
    }

    double GpuDeviceV2::KernelTime() const {
        return kernel_runtime_;
    }

    double GpuDeviceV2::FullTime() const {
        return complete_runtime_;
    }

    int GpuDeviceV2::Runs() const {
        return runs_;
    }

    std::string GpuDeviceV2::Name() const {
        return info_.name;
    }
}