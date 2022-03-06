#include <core/model_simulator_v2.h>
#include <thread>
#include "broker.h"
#include "worker.h"

static void RunSimWorker(const std::string& worker_address, const std::string& broker_address)
{
    std::unique_ptr<Service> v2 = std::make_unique<ModelSimulatorV2>();
    Worker w(v2, worker_address, broker_address);
    w.Start();
}

int main(int argc, char** argv)
{
    std::string broker_address = "tcp://0.0.0.0:5555";
    std::string broker_bind_address = "tcp://*:5555";

    std::thread t1( majordomo::BrokerActor,broker_bind_address);
    std::thread t2(RunSimWorker, "tcp://*:5558", broker_address);

    t1.join();
    t2.join();
	return 0;
}
