from ..util.simulation import SimulationJob
from ..util.fitting import FittingJob
from .connection_handler import FittingConnector, SimulationConnector

from os import path
import uuid
import capnp


class ModelComposer:
    def __init__(self, ips, ssh_mode=False, ssh_keyfile=None):
        self.fit_path = path.join(path.dirname(
            __file__), 'serialized_fitting_description.capnp')
        self.sim_path = path.join(path.dirname(
            __file__), 'serialized_simulation_description.capnp')

        self.fitting_description_template = capnp.load(self.fit_path)
        self.simulation_description_template = capnp.load(self.sim_path)
        self.uuid = str(uuid.uuid4())
        self.ips = ips
        self.ssh_mode = ssh_mode
        self.ssh_keyfile = ssh_keyfile
        self.fitting_connector = None
        self.sim_connector = None
        self.jobs = []

    def issue_job(self, job):
        if isinstance(job, FittingJob):
            self.start_fit_server()
            return self.fitting_connector.issue_job(job)
        elif isinstance(job, SimulationJob):
            self.start_sim_server()
            return self.sim_connector.issue_job(job)

    def start_fit_server(self):
        if self.fitting_connector is None:
            self.fitting_connector = FittingConnector(
                self.ips, self.uuid, self.ssh_mode, self.ssh_keyfile)

    def start_sim_server(self):
        if self.sim_connector is None:
            self.sim_connector = SimulationConnector(
                self.ips, self.uuid, self.ssh_mode, self.ssh_keyfile)
