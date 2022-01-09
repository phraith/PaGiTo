import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class DataContainer:
    def __init__(self, sim_runs, kernel_time, sim_time, fit_time):
        self.sim_runs = sim_runs
        self.kernel_time = kernel_time
        self.sim_time = sim_time
        self.fit_time = fit_time


class DataAnalyzer:
    def __init__(self):
        self.data = []

    def add_entry(self, job, result):
        self.data.append((job, result))

    def create_data_frames(self, path_to_folder):
        if len(self.data) == 0:
            return

        _, result_0 = self.data[0]

        df_dict = {}
        device_names = [x.deviceName for x in result_0.DeviceTimings()]

        header = ['pixels', 'runs', 'full_kernel_time', 'full_sim_time',
                  'fitting_time', 'kernel_time', 'sim_time',
                  'norm_kernel_time', 'norm_sim_time',
                  'inv_norm_kernel_time', 'inv_norm_sim_time'
                  ]

        for idx, name in enumerate(device_names):
            new_dev_name = name.replace(' ', '_') + '_' + str(idx)
            df_dict[new_dev_name] = pd.DataFrame(columns=header)

        for job, result in self.data:
            for idx, device_data in enumerate(result.device_timings):
                name = device_data.deviceName
                new_dev_name = name.replace(' ', '_') + '_' + str(idx)
                runs = device_data.simRuns
                full_kernel_time = device_data.kernelTime
                full_sim_time = device_data.simulationTime
                fitting_time = device_data.fittingTime

                kernel_time = device_data.averageKernelTime
                sim_time = device_data.averageSimulationTime

                norm_kernel_time = kernel_time / job.pixels
                norm_sim_time = sim_time / job.pixels

                inv_norm_kernel_time = 1 / norm_kernel_time
                inv_norm_sim_time = 1 / norm_sim_time

                dataList = [job.pixels, runs, full_kernel_time, full_sim_time, fitting_time,
                            kernel_time, sim_time, norm_kernel_time, norm_sim_time,
                            inv_norm_kernel_time, inv_norm_sim_time]

                df_dict[new_dev_name] = df_dict[new_dev_name].append(
                    dict(zip(header, dataList)), ignore_index=True)

        datestr = str(datetime.now().strftime('%m-%d_%H-%M-%S'))
        path_to_folder.joinpath('results').mkdir(parents=True, exist_ok=True)
        path_to_folder.joinpath('results', device_names[0]).mkdir(
            parents=True, exist_ok=True)
        path_to_folder.joinpath('results', device_names[0], datestr).mkdir(
            parents=True, exist_ok=True)

        for ddf in df_dict:
            df_dict[ddf].to_csv(path_to_folder.joinpath(
                'results', device_names[0], datestr, str(ddf) + '.csv'))

            plt.plot(df_dict[ddf]['pixels'], df_dict[ddf]
                     ['norm_sim_time'])
            plt.plot(df_dict[ddf]['pixels'], df_dict[ddf]
                     ['norm_kernel_time'])
            plt.savefig(path_to_folder.joinpath(
                'results', device_names[0], datestr, 'measurement.png'))
