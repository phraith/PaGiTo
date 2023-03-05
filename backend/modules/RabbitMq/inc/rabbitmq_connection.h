//
// Created by phili on 19.02.2023.
//

#ifndef GISAXSMODELINGFRAMEWORK_RABBITMQ_CONNECTION_H
#define GISAXSMODELINGFRAMEWORK_RABBITMQ_CONNECTION_H


#include <string>
#include <amqp.h>
#include <vector>
#include <functional>
#include "common/standard_defs.h"

class RabbitMqConnectionException : public std::exception {
public:
    RabbitMqConnectionException() : message("Connection failed!") {}

    RabbitMqConnectionException(char *msg) : message(msg) {}

    char *what() {
        return message;
    }

private:
    char *message;
};

class RabbitMqConnection {
public:
    RabbitMqConnection(std::string host_name, int port, std::string user_name, std::string password);

    uint16_t QueueDeclare(std::string queue_name);

    std::tuple<std::string, uint16_t> QueueDeclare();

    void RegisterConsumer(std::string queue_name);

    void RegisterConsumer(std::string queue_name, uint16_t channel_id);

    std::tuple<std::string, std::string> Consume();

    void
    Publish(int publisher_channel, std::string target_queue_name, std::string reply_to, std::string correlation_id,
            std::vector<std::byte> &bytes);

    void ConsumeAndPublish(int publisher_channel, std::function<std::vector<std::byte>(std::string message)> handler);

    uint16_t NextChannel();

    void ConnectSafe();

private:
    bool EvaluateRpcOperation(amqp_rpc_reply_t reply);

    void Establish();

    std::string host_name_;
    int port_;
    std::string user_name_;
    std::string password_;
    amqp_connection_state_t connection_;
    uint16_t next_channel_id_;
};


#endif //GISAXSMODELINGFRAMEWORK_RABBITMQ_CONNECTION_H
