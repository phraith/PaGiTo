//
// Created by phili on 26.03.2023.
//

#include "fit_job_client.h"

#include "gisaxs_tcp_handler.h"
#include <spdlog/spdlog.h>
#include <amqpcpp.h>
#include <amqpcpp/libuv.h>
#include <uv.h>

using namespace std::chrono_literals;

FitJobClient::FitJobClient(const std::string &host, int port, const std::string &user, const std::string &password)
        :
        host_(host),
        user_name_(user),
        password_(password),
        port_(port) {


    loop_ = CreateLoop();
    handler_ = std::make_shared<GisaxsTcpHandler>(loop_.get());
    connection_ = std::make_shared<AMQP::TcpConnection>(handler_.get(),
                                                        AMQP::Address(host_, port_, AMQP::Login(user_name_, password_),
                                                                      "/"));
    consumer_channel_ = std::make_shared<AMQP::TcpChannel>(connection_.get());
    publisher_channel_ = std::make_shared<AMQP::TcpChannel>(connection_.get());

    heartbeat_runner_ = std::jthread([&, stop_token = source_.get_token()] {
        while (!stop_token.stop_requested()) {
            connection_->heartbeat();
            spdlog::info("FitJobClient: heartbeat");
            std::this_thread::sleep_for(2s);
        };
    });

    consumer_channel_->declareQueue(AMQP::exclusive)
            .onSuccess([&](const std::string &queue_name, uint32_t messagecount, uint32_t consumercount) {
                queue_name_ = queue_name;
                consumer_channel_->consume(queue_name_).onSuccess([](const std::string &tag) {
                            spdlog::info("started consuming with tag {}", tag);
                        })
                        .onReceived([&](const AMQP::Message &message, uint64_t deliveryTag, bool redelivered) {
                            std::string job_result(message.body(), message.bodySize());
                            results_.emplace_back(
                                    std::tuple<std::string, std::string>{message.correlationID(), job_result});
                            consumer_channel_->ack(deliveryTag);
                        });
            })
            .onError([](const char *message) {
                spdlog::error("FitJobClient: {}", message);
            });

    while (queue_name_ == "") {
        spdlog::info("Waiting for queue to appear...");
        uv_run(loop_.get(), UV_RUN_ONCE);
        std::this_thread::sleep_for(2s);
    }
}

std::shared_ptr<uv_loop_s> FitJobClient::CreateLoop() const {
    auto loop = std::make_shared<uv_loop_s>();
    uv_loop_init(loop.get());
    return loop;
}

std::vector<std::tuple<std::string, std::string>>
FitJobClient::PublishBatch(const std::vector<std::vector<std::byte>> &configs) {
    results_.clear();
    for (int i = 0; i < configs.size(); ++i) {
        auto message = configs.at(i);
        auto message_pointer = reinterpret_cast<const char *>(&message[0]);
        AMQP::Envelope envelope(message_pointer, message.size());
        envelope.setCorrelationID(std::to_string(i));
        envelope.setContentType("text/plain");
        envelope.setReplyTo(queue_name_);
        envelope.setDeliveryMode(2);
        publisher_channel_->publish("", "Simulation", envelope);
    }

    while (results_.size() < configs.size()) {
        uv_run(loop_.get(), UV_RUN_ONCE);
    }

    return results_;
}

FitJobClient::~FitJobClient() {
    source_.request_stop();
}
