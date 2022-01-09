import numpy as np
import fabio

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from .. util.util import get_config


class SimulationJob:
    def __init__(self, path_to_config, path_to_inst):
        self.path_to_config = path_to_config
        self.path_to_inst = path_to_inst
        _, self.config_data = get_config(self.path_to_config)
        _, self.inst_data = get_config(self.path_to_inst)
        self.timestamp = 0
        self.is_last = False


class SimulationResult:
    def __init__(self, result):
        self.imgx = result.xDim
        self.imgy = result.yDim
        self.intensities = np.array(result.simulatedIntensities)
        self.img2D = np.reshape(self.intensities, (self.imgy, self.imgx))
        self.img2D_flipped = np.flip(self.img2D)
        self.qz_reshape = np.reshape(result.simulatedQz, (self.imgy, self.imgx))
        self.qx = result.simulatedQx
        self.qy = result.simulatedQy
        self.qz = result.simulatedQz

        self.extent_qy_qz = np.min(self.qy), np.max(
            self.qy), np.min(self.qz), np.max(self.qz)

    def show(self):
        pass
        plt.clf()
        plt.xlabel('Qy [1/nm]')
        plt.ylabel('Qz [1/nm]')

        zmax = np.amax(self.intensities)
        zmin = 1e-6*zmax

        if zmin == zmax == 0.0:
            norm = Normalize(0, 1)
        else:
            norm = LogNorm(zmin, zmax)

        plt.imshow(self.img2D_flipped, cmap='twilight_shifted',
                   norm=norm, extent=self.extent_qy_qz)

        plt.ylim(0.0, plt.ylim()[1])

        plt.colorbar()
        plt.show()

    def save_edf(self, path_to_folder):
        fabio.edfimage.edfimage(data=self.img2D).write(
            path_to_folder.joinpath(path_to_folder.name + '.edf'))

    def save_pdf(self, path_to_folder):
        plt.xlabel('Qy [1/nm]')
        plt.ylabel('Qz [1/nm]')

        zmax = np.amax(self.intensities)
        zmin = 1e-6*zmax

        if zmin == zmax == 0.0:
            norm = Normalize(0, 1)
        else:
            norm = LogNorm(zmin, zmax)

        plt.imshow(self.img2D_flipped, cmap='twilight_shifted',
                   norm=norm, extent=self.extent_qy_qz)

        plt.ylim(0.0, plt.ylim()[1])

        plt.colorbar()
        plt.savefig(path_to_folder.joinpath(
            path_to_folder.name + '.pdf'), bbox_inches='tight')
