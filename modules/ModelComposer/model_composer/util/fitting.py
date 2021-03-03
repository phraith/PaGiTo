import matplotlib.pyplot as plt
import numpy as np

from .. util.util import get_config


class FittingJob:
    def __init__(self, path_to_config, path_to_inst, intensities, indices, is_last):
        self.pathToConfig = path_to_config
        self.path_to_inst = path_to_inst
        _, self.config_data = get_config(self.pathToConfig)
        _, self.inst_data = get_config(self.path_to_inst)

        self.timestamp = 0

        self.intensities = intensities
        self.pixels = len(self.intensities)
        self.indices = indices
        self.is_last = is_last


class FittingResult:
    def __init__(self, result):
        self.intensities = np.array(result.simulatedIntensities)
        self.scale = result.scale
        self.device_timings = result.deviceTimingData
        self.fitted_shapes = result.fittedShapes
        self.fitting_time = result.deviceTimingData[0].fittingTime if len(
            result.deviceTimingData) > 0 else 0
        self.scale = result.scale
        self.fitness = result.fitness

    def __str__(self):
        return '\n'.join([str(shape) for shape in self.fitted_shapes])

    def show(self, real_intensities):
        plt.plot(real_intensities)
        plt.plot(self.intensities * self.scale, '--')
        plt.yscale('log')
        plt.show()
