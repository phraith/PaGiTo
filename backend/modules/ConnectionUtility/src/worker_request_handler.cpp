//
// Created by Phil on 30.01.2022.
//

#include "worker_request_handler.h"

#include <utility>

WorkerRequestHandler::WorkerRequestHandler(const std::string &address, const std::string &broker_address, std::string service)
        :
        socket_(context_, zmq::socket_type::dealer),
        address_(address),
        broker_address_(broker_address),
        service_(std::move(service)),
        reply_to_(""),
        heartbeat_at_(0),
        liveness_(majordomo::HEARTBEAT_LIVENESS),
        reconnect_(majordomo::HEARTBEAT_INTERVAL){
    socket_.bind(address);
    //socket_.bind("tcp://*:5558");
    ConnectToBroker(false);
}

void WorkerRequestHandler::Receive(zmq::multipart_t &request) {
    zmq::poller_t poller;
    poller.add(socket_, zmq::event_flags::pollin);

    std::vector<zmq::poller_event<>> events(1);
    int rc = poller.wait_all(events, majordomo::HEARTBEAT_INTERVAL);
    if (rc > 0) {
        zmq::multipart_t multipart_msg;
        majordomo::ReceiveFromDealer(socket_, multipart_msg);
        liveness_ = majordomo::HEARTBEAT_LIVENESS;
        std::string header = multipart_msg.popstr();
        assert(header == majordomo::worker::ident);

        std::string command = multipart_msg.popstr();
        if(majordomo::worker::request == command)
        {
            reply_to_ = multipart_msg.popstr();
            multipart_msg.pop();
            request = std::move(multipart_msg);
            return;
        }
        else if(majordomo::worker::heartbeat == command)
        {
        }
        else if(majordomo::worker::disconnect == command)
        {
            ConnectToBroker();
        }
        else
        {

        }
    }
    else {
        --liveness_;
        if (liveness_ == 0)
        {

        }

        majordomo::ms_sleep(reconnect_);
        ConnectToBroker();
    }

    if (majordomo::ms_now() >= heartbeat_at_) {
        zmq::multipart_t multipart_msg;
        multipart_msg.pushstr(majordomo::worker::heartbeat);
        multipart_msg.pushstr(majordomo::worker::ident);
        majordomo::SendToDealer(socket_, multipart_msg);
        heartbeat_at_ += majordomo::HEARTBEAT_INTERVAL;
    }
}

void WorkerRequestHandler::Send(zmq::multipart_t &reply) {
    if (reply.empty()) {return;}
    reply.pushmem(nullptr, 0);
    reply.pushstr(reply_to_);
    reply.pushstr(majordomo::worker::reply);
    reply.pushstr(majordomo::worker::ident);

    majordomo::SendToDealer(socket_, reply);
}

void WorkerRequestHandler::ConnectToBroker(bool reconnect) {
    if (reconnect)
    {
        socket_.disconnect(broker_address_);
    }

    int linger = 0;
    socket_.set(zmq::sockopt::linger, linger);
    socket_.connect(broker_address_);

    zmq::multipart_t multipart_msg;
    multipart_msg.pushstr(service_);
    multipart_msg.pushstr(majordomo::worker::ready);
    multipart_msg.pushstr(majordomo::worker::ident);
    auto result = majordomo::SendToDealer(socket_, multipart_msg);

    liveness_ = majordomo::HEARTBEAT_LIVENESS;
    heartbeat_at_ = majordomo::ms_now() + majordomo::HEARTBEAT_INTERVAL;

}

zmq::socket_t &WorkerRequestHandler::Socket() {
    return socket_;
}

const std::string &WorkerRequestHandler::ReplyTo() const {
    return reply_to_;
}


