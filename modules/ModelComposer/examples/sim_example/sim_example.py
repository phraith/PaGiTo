from pathlib import Path

from model_composer.core.model_composer import ModelComposer
from model_composer.util.simulation import SimulationJob

if __name__ == '__main__':
    #mc = ModelComposer([])
    mc = ModelComposer(["philipp@yoneda.dhcp.lbl.gov"], True, 'C:\\Users\\Phil\\.ssh\\id_rsa')

    current_path = Path(__file__).parent
    path_to_config_folder = current_path.joinpath('sim_cylinder_10nm_0sd')
    path_to_config = path_to_config_folder.joinpath('simulation.json')
    path_to_inst = current_path.joinpath('instrumentation.json')

    result = mc.issue_job(SimulationJob(path_to_config, path_to_inst))

    result.show()
    result.save_edf(path_to_config_folder)
    result.save_pdf(path_to_config_folder)
