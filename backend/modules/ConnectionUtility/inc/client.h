//
// Created by Phil on 26.03.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_CLIENT_H
#define GISAXSMODELINGFRAMEWORK_CLIENT_H

#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>

namespace majordomo {
    class Client {
    public:
        explicit Client(std::string  broker_address);
        void Send(const std::string &service_name, const std::string& job_payload);
        zmq::multipart_t Recv(const std::string &service_name);
    private:
        std::string broker_address_;
        zmq::context_t context_;
        zmq::socket_t socket_;
    };

}
#endif //GISAXSMODELINGFRAMEWORK_CLIENT_H
