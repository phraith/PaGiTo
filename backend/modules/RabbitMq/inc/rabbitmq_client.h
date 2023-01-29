//
// Created by phili on 29.01.2023.
//

#ifndef GISAXSMODELINGFRAMEWORK_RABBITMQ_CLIENT_H
#define GISAXSMODELINGFRAMEWORK_RABBITMQ_CLIENT_H

#include <string>

class RabbitMqClient {
public:
    RabbitMqClient(const std::string &host_name, int port, const std::string &queue_name);
private:
    std::string host_name_;
    std::string queue_name_;
    int port_;
};


#endif //GISAXSMODELINGFRAMEWORK_RABBITMQ_CLIENT_H
