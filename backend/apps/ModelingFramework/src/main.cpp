#include <core/model_simulator_v2.h>
#include <thread>
#include <spdlog/spdlog.h>
#include "core/model_fitter_v2.h"
#include "rabbitmq_client.h"
#include "rabbitmq_worker.h"

static std::string Username() {
    auto user = std::getenv("RABBITMQ_USER");

    if (user == nullptr) {
        spdlog::error("RABBITMQ_USER is not set!");
    }
    return user;
}

static std::string Password() {
    auto password = std::getenv("RABBITMQ_PASSWORD");
    if (password == nullptr) {
        spdlog::error("RABBITMQ_PASSWORD is not set!");
    }
    return password;
}

static std::string Host() {
    auto host = std::getenv("RABBITMQ_HOST");
    if (host == nullptr) {
        spdlog::error("RABBITMQ_HOST is not set!");
    }
    return host;
}

static std::shared_ptr<RabbitMqConnection> CreateConnection()
{
    return std::make_shared<RabbitMqConnection>(Host(), 5672, Username(), Password());
}

static void RunSimWorker(std::shared_ptr<HardwareInformation> hw_info) {
    std::unique_ptr<Service> v2 = std::make_unique<ModelSimulatorV2>(hw_info);
    RabbitMqWorker c(CreateConnection(), "Simulation", std::move(v2));
}

static void RunFitWorker() {
    std::unique_ptr<Service> v2 = std::make_unique<ModelFitterV2>(CreateConnection());
    RabbitMqWorker c(CreateConnection(), "Fitting", std::move(v2));
}

int main(int argc, char **argv) {
    std::shared_ptr<HardwareInformation> hw_info = std::make_shared<HardwareInformation>();
    std::thread t1(RunSimWorker, hw_info);
    std::thread t2(RunFitWorker);

    t1.join();
    t2.join();
    return 0;
}
