#include "core/connector.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <memory>

#include <zmq.hpp>

#include <capnp/compat/std-iterator.h>
#include <capnp/message.h>
#include <capnp/serialize-packed.h>

#include <util/hardware_information.h>
#include <core/model_simulator.h>
#include <common/unitcell.h>
#include <nlohmann/json.hpp>

#include "standard_vector_types.h"
#include "util/utility.h"

using json = nlohmann::json;

Connector::Connector(const std::string ip)
	:
	ip_(ip),
	connection_handler_(&Connector::Listen, this),
    quit_work_(false)
{
}

Connector::~Connector()
{
    connection_handler_.join();
}

void Connector::Listen()
{
    zmq::context_t context;
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind("tcp://" + ip_);
    std::cout << "tcp://" + ip_  + "\n" << std::endl;

    while (true) {
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);
        socket.send(zmq::buffer("Server received fitting job..."));
        capnp::word* buffer;
        if (reinterpret_cast<uintptr_t>(reply.data()) % sizeof(void*) == 0)
        {
            buffer = reinterpret_cast<capnp::word*>(reply.data());
            std::cout << "Aligned" << std::endl;
        }
        else
        {
            buffer = new capnp::word[reply.size() / sizeof(capnp::word)];
            memcpy(buffer, reply.data(), reply.size());
            std::cout << "Not Aligned" << std::endl;
        }
        const kj::ArrayPtr<const capnp::word> v(buffer, reply.size() / sizeof(capnp::word));
        capnp::FlatArrayMessageReader message2(v);
        SerializedFittingDescription::Reader x = message2.getRoot<SerializedFittingDescription>();

        auto descr = ConstructFittingDescription(x);

        InsertFittingJob(descr);

        if (descr->IsLast())
            break;
    }
    socket.unbind("tcp://" + ip_);
}

std::shared_ptr<FitJob> Connector::ConstructFittingDescription(const SerializedFittingDescription::Reader &data)
{
    std::string clientId = std::string(data.getClientId().cStr());
    std::string configData = std::string(data.getConfigData().cStr());

    std::string instData = std::string(data.getInstrumentationData().cStr());

    bool is_last = data.getIsLast();

    std::vector<MyType> intensities(data.getIntensities().begin(), data.getIntensities().end());
    std::vector<int> offsets(data.getOffsets().begin(), data.getOffsets().end());

    std::unique_ptr<ImageData> i = std::make_unique<ImageData>(intensities, offsets);
    std::shared_ptr<ModelSimulator> ms = std::make_shared<ModelSimulator>();

    json gisaxs_in = json::parse(configData);
    json inst_in = json::parse(instData);

    const auto& detector_data = inst_in.at("detector");
    const auto& scattering = inst_in.at("scattering");
    const auto& fitting = inst_in.at("fitting");

    const auto& unitcell = gisaxs_in.at("unitcell");
    const auto& substrate = gisaxs_in.at("substrate");
    const auto& sub_ref_index = substrate.at("refindex");

    MyType sub_delta = sub_ref_index.at("delta");
    MyType sub_beta = sub_ref_index.at("beta");

    MyType pixelsize = detector_data.at("pixelsize");
    MyType sdd = detector_data.at("sdd");
    MyType2 directbeam = { detector_data.at("directbeam")[0], detector_data.at("directbeam")[1] };
    MyType2I resolution = { detector_data.at("resolution")[0], detector_data.at("resolution")[1] };
    MyType alphai = scattering.at("alphai");

    //MyType wavelength = scattering.at("wavelength");
    MyType ev = scattering.at("photon").at("ev");
    MyType wavelength = 1239.84 / ev;
    MyType3I repetitions = { unitcell.at("repetitions")[0], unitcell.at("repetitions")[1], unitcell.at("repetitions")[2] };
    MyType3 distances = { unitcell.at("distances")[0], unitcell.at("distances")[1], unitcell.at("distances")[2] };

    std::shared_ptr<ExperimentalModel> experimental_model = std::make_shared<ExperimentalModel>(
        Detector{ pixelsize, resolution, directbeam },
        offsets,
        BeamConfiguration{ alphai * 0.017453, {1,1}, wavelength, 0.1 },
        Sample{ Layer {sub_delta, sub_beta, -1, 0} , std::vector<Layer>{} }, sdd, 0);

    //experimental_model->PrintInfo();

    std::shared_ptr<Unitcell> h_unitcell = std::make_shared<Unitcell> (repetitions, distances);
    for (const auto& component : unitcell.at("components"))
    {
        const std::string& shape_key = component.at("shape");
        const auto& shape = gisaxs_in.at(shape_key);
        
        std::vector<MyType3> locations;
        for (const auto& location : component.at("locations"))
        {
            locations.emplace_back(MyType3{ location[0], location[1], location[2] });
        }

        ShapeType type = Utility::ConvertStringToShapeType(shape.at("type"));

        switch (type)
        {
        case ShapeType::kCylinder:
        {
            const auto& radius_mean_range = shape.at("params").at("radius").at("mean_range");
            const auto& radius_stddev_range = shape.at("params").at("radius").at("stddev_range");
            const auto& height_mean_range = shape.at("params").at("height").at("mean_range");
            const auto& height_stddev_range = shape.at("params").at("height").at("stddev_range");;

            h_unitcell->InsertShape(
                std::make_unique<Cylinder>(
                    MyType2{ radius_mean_range[0], radius_mean_range[1] }, MyType2{ radius_stddev_range[0], radius_stddev_range[1] },
                    MyType2{ height_mean_range[0], height_mean_range[1] }, MyType2{ height_stddev_range[0], height_stddev_range[1] }, locations), ShapeType::kCylinder);
            break;
        }
        case ShapeType::kSphere:
        {
            const auto& radius_mean_range = shape.at("params").at("radius").at("mean_range");
            const auto& radius_stddev_range = shape.at("params").at("radius").at("stddev_range");

            h_unitcell->InsertShape(
                std::make_unique<Sphere>(
                    MyType2{ radius_mean_range[0], radius_mean_range[1] }, MyType2{ radius_stddev_range[0], radius_stddev_range[1] }, locations), ShapeType::kSphere);
            break;
        }
        case ShapeType::kTrapezoid:
        {
            const auto& beta_mean_range = shape.at("params").at("beta").at("mean_range");
            const auto& beta_stddev_range = shape.at("params").at("beta").at("stddev_range");
            const auto& L_mean_range = shape.at("params").at("length").at("mean_range");
            const auto& L_stddev_range = shape.at("params").at("length").at("stddev_range");
            const auto& h_mean_range = shape.at("params").at("height").at("mean_range");
            const auto& h_stddev_range = shape.at("params").at("height").at("stddev_range");

            h_unitcell->InsertShape(
                std::make_unique<Trapezoid>(
                    MyType2{ beta_mean_range[0], beta_mean_range[1] }, MyType2{ beta_stddev_range[0], beta_stddev_range[1] },
                    MyType2{ L_mean_range[0], L_mean_range[1] }, MyType2{ L_stddev_range[0], L_stddev_range[1] },
                    MyType2{ h_mean_range[0], h_mean_range[1] }, MyType2{ h_stddev_range[0], h_stddev_range[1] },
                    locations), ShapeType::kSphere);
            break;
        }
        default:
            break;
        }
    }

    int evolutions = fitting.at("evolutions");
    int populations = fitting.at("populations");
    int individuals = fitting.at("individuals");

    return std::make_shared<FitJob>(experimental_model, std::move(i), h_unitcell, clientId, evolutions, individuals, populations, is_last);
}

void Connector::InsertFittingJob(std::shared_ptr<FitJob> fitting_job)
{
    std::lock_guard <std::mutex> lock(job_mutex_);
    fitting_jobs_.emplace_back(fitting_job);

    job_cv_.notify_one();
}

std::shared_ptr<FitJob> Connector::TakeFittingJob()
{
    auto lk = std::unique_lock<std::mutex>(job_mutex_);
    job_cv_.wait(lk, [&]
        {
            return !fitting_jobs_.empty() || quit_work_;
        });

    if (fitting_jobs_.empty())
        return nullptr;

    auto fitting_job = fitting_jobs_.front();
    fitting_jobs_.pop_front();

    if (fitting_job->IsLast())
        quit_work_ = true;

    return fitting_job;
}