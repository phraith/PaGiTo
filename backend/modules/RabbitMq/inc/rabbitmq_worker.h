//
// Created by phili on 18.02.2023.
//

#ifndef GISAXSMODELINGFRAMEWORK_RABBITMQ_WORKER_H
#define GISAXSMODELINGFRAMEWORK_RABBITMQ_WORKER_H

#include <string>
#include <memory>
#include <thread>
#include "common/service.h"

class RabbitMqWorker {
public:
    RabbitMqWorker(const std::string &host, int port, const std::string &user, const std::string &password,
                   const std::string &queue_name, std::unique_ptr<Service> service);

private:

    std::vector<std::byte> HandleRequest(std::string message);

    void Run();
    std::string user_name_;
    std::string password_;
    std::string host_;
    int port_;
    std::unique_ptr<Service> service_;
    std::string queue_name_;
};


#endif //GISAXSMODELINGFRAMEWORK_RABBITMQ_WORKER_H
