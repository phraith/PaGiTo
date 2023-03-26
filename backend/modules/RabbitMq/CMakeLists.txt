find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(amqpcpp REQUIRED)
find_package(libuv REQUIRED)


add_library(RabbitMq STATIC)
target_sources(RabbitMq
        PRIVATE
        src/gisaxs_tcp_handler.cpp
        inc/gisaxs_tcp_handler.h
        src/fit_job_client.cpp
        inc/fit_job_client.h
        #        src/rabbitmq_client.cpp
        #        inc/rabbitmq_client.h
        src/rabbitmq_worker.cpp
        inc/rabbitmq_worker.h
        #        src/rabbitmq_connection.cpp
        #        inc/rabbitmq_connection.h
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
        amqpcpp
        uv_a
        CommonLib
        )