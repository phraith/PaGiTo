#include "util/sim_connector.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

#include <zmq.hpp>
#include <capnp/compat/std-iterator.h>
#include <capnp/message.h>
#include <capnp/serialize-packed.h>

#include <util/hardware_information.h>
#include <common/unitcell.h>
#include <nlohmann/json.hpp>

#include "standard_vector_types.h"
#include "util/utility.h"

using json = nlohmann::json;

SimConnector::SimConnector()
{
}

SimConnector::SimConnector(const std::string ip)
    :
    ip_(ip),
    connection_handler_(&SimConnector::Listen, this),
    quit_work_(false)
{
}

SimConnector::~SimConnector()
{
    if (connection_handler_.joinable())
        connection_handler_.join();
}

void SimConnector::InsertSimulationJob(std::shared_ptr<SimJob> simulation_job)
{
    std::lock_guard <std::mutex> lock(job_mutex_);
    simulation_jobs_.emplace_back(simulation_job);

    job_cv_.notify_one();
}

std::shared_ptr<SimJob> SimConnector::TakeSimulationJob()
{
    auto lk = std::unique_lock<std::mutex>(job_mutex_);
    job_cv_.wait(lk, [&]
        {
            return !simulation_jobs_.empty() || quit_work_;
        });

    if (simulation_jobs_.empty())
        return nullptr;

    auto simulation_job = simulation_jobs_.front();
    simulation_jobs_.pop_front();

    if (simulation_job->IsLast())
        quit_work_ = true;

    return simulation_job;
}

void SimConnector::Listen()
{
    zmq::context_t context;
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind("tcp://" + ip_);
    std::cout << "tcp://" + ip_ + "\n" << std::endl;

    while (true) {
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);
        socket.send(zmq::buffer("Server received simulation job..."));
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
        SerializedSimulationDescription::Reader x = message2.getRoot<SerializedSimulationDescription>();

        auto descr = ConstructSimulationDescription(x);

        InsertSimulationJob(descr);

        if (descr->IsLast())
            break;
    }
    socket.unbind("tcp://" + ip_);
}

std::shared_ptr<SimJob> SimConnector::ConstructSimulationDescription(const SerializedSimulationDescription::Reader& data)
{
    std::string clientId = std::string(data.getClientId().cStr());
    std::string configData = std::string(data.getConfigData().cStr());
    std::string instData = std::string(data.getInstrumentationData().cStr());
    bool is_last = data.getIsLast();

    json gisaxs_in = json::parse(configData);
    json inst_in = json::parse(instData);

    const auto& detector_data = inst_in.at("detector");
    const auto& scattering = inst_in.at("scattering");
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
        std::vector<int> {},
        BeamConfiguration{ alphai * 0.017453, {1,1}, wavelength, 0.1 },
        Sample{ Layer {sub_delta, sub_beta, -1, 0} , std::vector<Layer>{} }, sdd, 0);

        //Layer {0.001, 1e-05, 1, 5} 

    //experimental_model->PrintInfo();

    std::shared_ptr<Unitcell> h_unitcell = std::make_shared<Unitcell>(Unitcell{ repetitions, distances } );

    for (const auto& component : unitcell.at("components"))
    {
        const std::string& shape_key = component.at("shape");
        const auto& shape = gisaxs_in.at(shape_key);

        std::vector<MyType3> locations;
        for (const auto& location : component.at("locations"))
        {
            locations.emplace_back( MyType3 {location[0], location[1], location[2]} );
        }

        ShapeType type = Utility::ConvertStringToShapeType(shape.at("type"));
        std::cout << shape.at("type") << std::endl;
        switch (type)
        {
        case ShapeType::kCylinder:
        {
            const auto& radius_mean = shape.at("params").at("radius").at("mean");
            const auto& radius_stddev = shape.at("params").at("radius").at("stddev");
            const auto& height_mean = shape.at("params").at("height").at("mean");
            const auto& height_stddev = shape.at("params").at("height").at("stddev");;

            h_unitcell->InsertShape(
                std::make_unique<Cylinder>(
                    MyType2{ radius_mean, radius_stddev }, MyType2{ height_mean, height_stddev }, locations), ShapeType::kCylinder);
            break;
        }
        case ShapeType::kSphere:
        {
            const auto& radius_mean = shape.at("params").at("radius").at("mean");
            const auto& radius_stddev = shape.at("params").at("radius").at("stddev");

            h_unitcell->InsertShape(
                std::make_unique<Sphere>(
                    MyType2{ radius_mean, radius_stddev }, locations), ShapeType::kSphere);
            break;
        }
        case ShapeType::kTrapezoid:
        {
            const auto& beta_mean = shape.at("params").at("beta").at("mean");
            const auto& beta_stddev = shape.at("params").at("beta").at("stddev");
            const auto& L_mean = shape.at("params").at("length").at("mean");
            const auto& L_stddev = shape.at("params").at("length").at("stddev");
            const auto& h_mean = shape.at("params").at("height").at("mean");
            const auto& h_stddev = shape.at("params").at("height").at("stddev");

            h_unitcell->InsertShape(
                std::make_unique<Trapezoid>(
                    MyType2{ beta_mean, beta_stddev }, MyType2{ L_mean, L_stddev }, MyType2{ h_mean, h_stddev }, locations), ShapeType::kTrapezoid);
            break;
        }
        default:
            break;
        }
    }

    return std::make_shared<SimJob>(clientId, h_unitcell, experimental_model, is_last);
}
