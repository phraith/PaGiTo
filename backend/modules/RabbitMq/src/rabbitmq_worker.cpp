//
// Created by phili on 18.02.2023.
//

#include "rabbitmq_worker.h"
#include "gisaxs_tcp_handler.h"
#include <iostream>
#include <spdlog/spdlog.h>
#include <amqpcpp.h>
#include <amqpcpp/libuv.h>
#include <uv.h>

using namespace std::chrono_literals;

RabbitMqWorker::RabbitMqWorker(const std::string &host, int port, const std::string &user, const std::string &password,
                               const std::string &queue_name, std::unique_ptr<Service> service)
        :
        host_(host),
        user_name_(user),
        password_(password),
        port_(port),
        service_(std::move(service)),
        queue_name_(queue_name) {
    Run();
}

void RabbitMqWorker::Run() {

    auto loop = std::make_shared<uv_loop_s>();
    uv_loop_init(loop.get());

    auto login = AMQP::Login(user_name_, password_);
    auto address = AMQP::Address(host_, port_, login, "/");
    GisaxsTcpHandler handler = GisaxsTcpHandler(loop.get());
    AMQP::TcpConnection connection(&handler, address);
    AMQP::TcpChannel consumer_channel(&connection);
    AMQP::TcpChannel publisher_channel(&connection);
    auto request_handler = std::bind(&RabbitMqWorker::HandleRequest, this, std::placeholders::_1);

    auto source = std::stop_source();
    auto heartbeat_runner = std::jthread([&, stop_token = source.get_token()] {
        while (!stop_token.stop_requested()) {
            connection.heartbeat();
            spdlog::info("FitJobClient: heartbeat");
            std::this_thread::sleep_for(2s);
        };
    });

    consumer_channel.declareQueue(queue_name_, AMQP::durable).onSuccess(
                    [&consumer_channel, &publisher_channel, &request_handler](const std::string &queue_name,
                                                                              uint32_t messagecount,
                                                                              uint32_t consumercount) {
                        spdlog::info("declared queue {}", queue_name);
                        consumer_channel.consume(queue_name).onSuccess([](const std::string &tag) {
                                    spdlog::info("started consuming with tag {}", tag);
                                })
                                .onReceived(
                                        [&consumer_channel, &publisher_channel, &request_handler](
                                                const AMQP::Message &message,
                                                uint64_t deliveryTag,
                                                bool redelivered) {
                                            std::string job_config(message.body(), message.bodySize());
                                            spdlog::info("received {}", deliveryTag);
                                            const std::vector<std::byte> &data = request_handler(job_config);
                                            auto result = reinterpret_cast<const char *>(&data[0]);

                                            if (result == nullptr) {
                                                consumer_channel.ack(deliveryTag);
                                                return;
                                            }

                                            AMQP::Envelope envelope(result, data.size());
                                            envelope.setHeaders(message.headers());
                                            envelope.setCorrelationID(message.correlationID());
                                            envelope.setContentType("text/plain");
                                            envelope.setDeliveryMode(2);
                                            publisher_channel.publish("", message.replyTo(), envelope, AMQP::headers);
                                            consumer_channel.ack(deliveryTag);
                                        });

                    })
            .onError([](const char *message) {
                spdlog::error(message);
            });

    uv_run(loop.get(), UV_RUN_DEFAULT);
    source.request_stop();
}

std::vector<std::byte> RabbitMqWorker::HandleRequest(std::string message) {
    return service_->HandleRequest(message).data;
}