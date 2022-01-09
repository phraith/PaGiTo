#ifndef MODEL_SIMULATOR_CORE_CONNECTORV2_H
#define MODEL_SIMULATOR_CORE_CONNECTORV2_H

#include <string>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>

class ConnectorV2
{
public:
	ConnectorV2(const std::string ip);
	~ConnectorV2();
private:
	void Listen();

	const std::string ip_;
	std::thread connection_handler_;
};

#endif