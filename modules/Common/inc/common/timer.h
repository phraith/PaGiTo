#ifndef MODEL_SIMULATOR_UTIL_TIMER_H
#define MODEL_SIMULATOR_UTIL_TIMER_H

#include <chrono>

class Timer
{
public:
	Timer();

	void Start();
	void End();

	double Duration();

private:
	std::chrono::time_point<std::chrono::steady_clock> start_;
	std::chrono::time_point<std::chrono::steady_clock> end_;
};
#endif