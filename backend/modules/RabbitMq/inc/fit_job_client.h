//
// Created by phili on 26.03.2023.
//

#ifndef GISAXSMODELINGFRAMEWORK_FIT_JOB_CLIENT_H
#define GISAXSMODELINGFRAMEWORK_FIT_JOB_CLIENT_H


#include <string>
#include <uv.h>
#include <thread>
#include "gisaxs_tcp_handler.h"

class FitJobClient {
public:
    FitJobClient(const std::string &host, int port, const std::string &user, const std::string &password);

    ~FitJobClient();

    std::vector<std::tuple<std::string, std::string>> PublishBatch(const std::vector<std::vector<std::byte>> &configs);

private:
    std::shared_ptr<uv_loop_s> CreateLoop() const;

    std::string queue_name_;
    std::string user_name_;
    std::string password_;
    std::string host_;
    int port_;
    std::vector<std::tuple<std::string, std::string>> results_;
    std::shared_ptr<uv_loop_s> loop_;
    std::shared_ptr<GisaxsTcpHandler> handler_;
    std::shared_ptr<AMQP::TcpConnection> connection_;
    std::shared_ptr<AMQP::TcpChannel> consumer_channel_;
    std::shared_ptr<AMQP::TcpChannel> publisher_channel_;
    std::stop_source source_;
    std::jthread heartbeat_runner_;
};


#endif //GISAXSMODELINGFRAMEWORK_FIT_JOB_CLIENT_H
