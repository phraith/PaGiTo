import fabio
from pathlib import Path

from model_composer.core.model_composer import ModelComposer
from model_composer.util.util import concat_vert_lps
from model_composer.util.fitting import FittingJob

if __name__ == '__main__':
    mc = ModelComposer([])
    #mc = ModelComposer(["philipp@yoneda.dhcp.lbl.gov"], True, 'C:\\Users\\Phil\\.ssh\\id_rsa')

    current_path = Path(__file__).parent
    path_to_config_folder = current_path.joinpath('fit_sphere_10nm_0sd')
    path_to_config = path_to_config_folder.joinpath('fitting.json')
    path_to_inst = current_path.joinpath('instrumentation.json')

    obj = fabio.open(path_to_config_folder.joinpath('sphere_10nm_0sd.edf'))
    intensities, indices = concat_vert_lps([737], obj)

    result = mc.issue_job(FittingJob(
        path_to_config, path_to_inst, intensities.tolist(), indices.tolist(), False))

    print(result)

    result.show(intensities)

    