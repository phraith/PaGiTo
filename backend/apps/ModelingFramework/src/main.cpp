#include <core/model_simulator_v2.h>
#include <thread>
#include "broker.h"
#include "worker.h"
#include "core/model_fitter_v2.h"

static void RunSimWorker(const std::string& worker_address, const std::string& broker_address, std::shared_ptr<HardwareInformation> hw_info)
{
    std::unique_ptr<Service> v2 = std::make_unique<ModelSimulatorV2>(hw_info);
    Worker w(v2, worker_address, broker_address);
    w.Start();
}

static void RunFitWorker(const std::string& worker_address, const std::string& broker_address)
{
    std::unique_ptr<Service> v2 = std::make_unique<ModelFitterV2>(broker_address);
    Worker w(v2, worker_address, broker_address);
    w.Start();
}

int main(int argc, char** argv)
{
    std::string broker_address = "tcp://0.0.0.0:5555";
    std::string broker_bind_address = "tcp://*:5555";

    std::shared_ptr<HardwareInformation> hw_info = std::make_shared<HardwareInformation>();

    std::thread t1( majordomo::BrokerActor, broker_bind_address);
    std::thread t2(RunSimWorker, "tcp://*:5558", broker_address, hw_info);
    std::thread t4(RunSimWorker, "tcp://*:5559", broker_address, hw_info);
    std::thread t5(RunSimWorker, "tcp://*:5560", broker_address, hw_info);
    std::thread t3(RunFitWorker, "tcp://*:5557", broker_address);

    t1.join();
    t2.join();
    t3.join();
	return 0;
}
