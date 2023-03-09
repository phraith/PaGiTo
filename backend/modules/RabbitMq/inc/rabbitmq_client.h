//
// Created by phili on 29.01.2023.
//

#ifndef GISAXSMODELINGFRAMEWORK_RABBITMQ_CLIENT_H
#define GISAXSMODELINGFRAMEWORK_RABBITMQ_CLIENT_H

#include <string>
#include <amqp.h>
#include <memory>
#include "common/service.h"

class RabbitMqClient {
public:
    RabbitMqClient(const std::string &host_name, int port, const std::string &queue_name);


private:
    void SetUpConnection();
    bool EvaluateRpcOperation(amqp_rpc_reply_t reply);

    std::string host_name_;
    std::string queue_name_;
    int port_;
    amqp_connection_state_t connection_;
};


#endif //GISAXSMODELINGFRAMEWORK_RABBITMQ_CLIENT_H
