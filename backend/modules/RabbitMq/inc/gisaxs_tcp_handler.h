//
// Created by phili on 25.03.2023.
//

#ifndef GISAXSMODELINGFRAMEWORK_GISAXS_TCP_HANDLER_H
#define GISAXSMODELINGFRAMEWORK_GISAXS_TCP_HANDLER_H

#include <amqpcpp.h>
#include <amqpcpp/libuv.h>

class GisaxsTcpHandler : public AMQP::LibUvHandler {
public:
    GisaxsTcpHandler(uv_loop_t *loop) : AMQP::LibUvHandler(loop) {}

    virtual ~GisaxsTcpHandler() = default;

private:
    virtual void onError(AMQP::TcpConnection *connection, const char *message) override {
        std::cout << "error: " << message << std::endl;
    }

    virtual void onConnected(AMQP::TcpConnection *connection) override {
        std::cout << "connected" << std::endl;
    }
};

#endif //GISAXSMODELINGFRAMEWORK_GISAXS_TCP_HANDLER_H
