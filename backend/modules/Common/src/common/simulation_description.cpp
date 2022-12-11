#include "common/simulation_description.h"

SimJob::SimJob(const JobMetaInformation &meta_information, const ExperimentalData &experimental_information)
        :
        meta_information_(meta_information),
        experimental_information_(experimental_information) {}

const JobMetaInformation &SimJob::JobInfo() const {
    return meta_information_;
}

const ExperimentalData &SimJob::ExperimentInfo() const {
    return experimental_information_;
}

const int SimJob::Qcount() const {
    if (meta_information_.SimulationTargets().size() == 0) {
        return experimental_information_.DetectorConfig().PixelCount();
    }

    int qcount = 0;
    for (const auto &simulation_target: meta_information_.SimulationTargets()) {
        assert(simulation_target.start.y <= simulation_target.end.y);
        assert(simulation_target.start.x <= simulation_target.end.x);
        int x_diff = simulation_target.end.x - simulation_target.start.x + 1;
        int y_diff = simulation_target.end.y - simulation_target.start.y + 1;

        qcount += x_diff * y_diff;
    }

    return qcount;
}

std::vector<Vector2<int>> SimJob::DetectorPositions() const {
    if (meta_information_.SimulationTargets().size() == 0) {
        return std::vector<Vector2<int>>();
    }

    std::vector<Vector2<int>> detector_positions;
    for (const auto &simulation_target: meta_information_.SimulationTargets()) {
        assert(simulation_target.start.y >= simulation_target.end.y);
        assert(simulation_target.start.x <= simulation_target.end.x);
        for (int y = simulation_target.start.y; y >= simulation_target.end.y; --y) {
            for (int x = simulation_target.start.x; x <= simulation_target.end.x; ++x) {

                detector_positions.emplace_back(Vector2<int>{x, y});
            }
        }
    }
    return detector_positions;
}