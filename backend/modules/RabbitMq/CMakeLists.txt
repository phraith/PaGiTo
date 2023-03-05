find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(rabbitmq-c CONFIG REQUIRED)

add_library(RabbitMq STATIC)
target_sources(RabbitMq
        PRIVATE
        src/rabbitmq_client.cpp
        inc/rabbitmq_client.h
        src/rabbitmq_worker.cpp
        inc/rabbitmq_worker.h
        src/rabbitmq_connection.cpp
        inc/rabbitmq_connection.h
        inc/rabbitmq_message.h
        )

target_include_directories(RabbitMq
        PUBLIC
        inc
        )

target_link_libraries(RabbitMq
        PRIVATE
        nlohmann_json
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        rabbitmq::rabbitmq-static
        CommonLib
        )