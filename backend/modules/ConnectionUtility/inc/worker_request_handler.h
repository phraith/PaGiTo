//
// Created by Phil on 30.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_WORKER_REQUEST_HANDLER_H
#define GISAXSMODELINGFRAMEWORK_WORKER_REQUEST_HANDLER_H


#include <zmq.hpp>
#include <zmq_addon.hpp>
#include "majordomo_utility.h"

class WorkerRequestHandler {
public:
    WorkerRequestHandler(const std::string &address, const std::string &broker_address, std::string service);

    void Receive(zmq::multipart_t &request);
    void Send(zmq::multipart_t &reply);
    zmq::socket_t &Socket();
private:
    void ConnectToBroker(bool reconnect = true);
private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    std::string address_;
    std::string broker_address_;
    std::string service_;
    std::string reply_to_;
    majordomo::ms_time_t heartbeat_at_;
    majordomo::ms_time_t reconnect_;
    int liveness_;
};


#endif //GISAXSMODELINGFRAMEWORK_WORKER_REQUEST_HANDLER_H
