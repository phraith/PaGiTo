//
// Created by phili on 18.02.2023.
//

#include "rabbitmq_worker.h"

#include <sys/time.h>
#include "rabbitmq_client.h"
#include "amqp.h"
#include <iostream>
#include <spdlog/spdlog.h>

#define SUMMARY_EVERY_US 1000000

uint64_t now_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t) tv.tv_sec * 1000000 + (uint64_t) tv.tv_usec;
}

RabbitMqWorker::RabbitMqWorker(std::shared_ptr<RabbitMqConnection> connection, const std::string &queue_name,
                               std::unique_ptr<Service> service)
        :
        connection_(connection),
        service_(std::move(service)),
        queue_name_(queue_name) {
    Run();
}

void RabbitMqWorker::Run() {
    uint64_t start_time = now_microseconds();
    int received = 0;
    int previous_received = 0;
    uint64_t previous_report_time = start_time;
    uint64_t next_summary_time = start_time + SUMMARY_EVERY_US;

    connection_->RegisterConsumer(queue_name_);
    uint16_t publisher_channel_id = connection_->NextChannel();

    auto function_pointer = std::bind(&RabbitMqWorker::HandleRequest, this, std::placeholders::_1);

    for (;;) {
        amqp_rpc_reply_t ret;
        amqp_envelope_t envelope;

        uint64_t now = now_microseconds();
        if (now > next_summary_time) {
            int countOverInterval = received - previous_received;
            double intervalRate =
                    countOverInterval / ((now - previous_report_time) / 1000000.0);

            int current_ms = (int) (now - start_time) / 1000;
            spdlog::info("{} ms: Received {} - {} since last report ({} Hz)", current_ms, received, countOverInterval,
                         (int) intervalRate);

            previous_received = received;
            previous_report_time = now;
            next_summary_time += SUMMARY_EVERY_US;
        }

        try {
            connection_->ConsumeAndPublish(publisher_channel_id, function_pointer);
        }
        catch (const RabbitMqConnectionException &e) {
            connection_->ConnectSafe();
            connection_->RegisterConsumer(queue_name_);
            publisher_channel_id = connection_->NextChannel();
        }
        received++;
    }
}

std::vector<std::byte> RabbitMqWorker::HandleRequest(std::string message) {
    return service_->HandleRequest(message).data;
}
