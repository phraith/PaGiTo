//
// Created by phili on 18.02.2023.
//

#ifndef GISAXSMODELINGFRAMEWORK_RABBITMQ_WORKER_H
#define GISAXSMODELINGFRAMEWORK_RABBITMQ_WORKER_H

#include <string>
#include <memory>
#include "common/service.h"
#include "rabbitmq_client.h"
#include "rabbitmq_connection.h"

class RabbitMqWorker {
public:
    RabbitMqWorker(std::shared_ptr<RabbitMqConnection> connection, const std::string &queue_name, std::unique_ptr<Service> service);

private:

    std::vector<std::byte> HandleRequest(std::string message);

    void Run();
    std::unique_ptr<Service> service_;
    std::shared_ptr<RabbitMqConnection> connection_;
    std::string queue_name_;

};


#endif //GISAXSMODELINGFRAMEWORK_RABBITMQ_WORKER_H
