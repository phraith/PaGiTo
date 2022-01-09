import capnp

import zmq
import time
import capnp

from .. util.fitting import FittingResult
from .. util.simulation import SimulationResult
from zmq import ssh

from os import path


class SimulationConnector:
    def __init__(self, ips, client_id, ssh_mode=False, ssh_keyfile=None):
        self.ips = ips
        self.client_id = client_id
        self.ssh_keyfile = ssh_keyfile
        self.ssh_mode = ssh_mode

        self.sim_path = path.join(path.dirname(
            __file__), 'serialized_simulation_description.capnp')

        self.sim_template = capnp.load(self.sim_path)

        self.description_template = self.sim_template.SerializedSimulationDescription
        self.result_template = self.sim_template.SerializedSimResult

        self.req_context = zmq.Context()
        self.req_socket = self.req_context.socket(zmq.REQ)
        self.sub_context = zmq.Context()
        self.sub_socket = self.sub_context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, str(self.client_id).encode())

        self.local_req_ip = 'tcp://127.0.0.1:5558'
        self.local_sub_ip = 'tcp://127.0.0.1:5559'

        self.connect()
        time.sleep(5)

    def connect(self):
        if self.ssh_mode:
            for ip in self.ips:
                ssh.tunnel_connection(
                    self.req_socket, self.local_req_ip, ip, keyfile=self.ssh_keyfile)
                ssh.tunnel_connection(
                    self.sub_socket, self.local_sub_ip, ip, keyfile=self.ssh_keyfile)
        else:
            self.req_socket.connect(self.local_req_ip)
            self.sub_socket.connect(self.local_sub_ip)

    def issue_job(self, job):

        is_last = job.is_last

        sfd = self.description_template.new_message(
            timestamp=job.timestamp, clientId=self.client_id,
            instrumentationData=job.inst_data,
            configData=job.config_data, isLast=is_last)

        self.req_socket.send(sfd.to_bytes())
        reply = self.req_socket.recv()
        print(reply.decode())

        if is_last:
            self.req_socket.close()

        _ = self.sub_socket.recv()
        message = self.sub_socket.recv()

        if is_last:
            self.sub_socket.close()

        dssd = self.result_template.from_bytes(message)

        return SimulationResult(dssd)


class FittingConnector:
    def __init__(self, ips, clientId, ssh_mode=False, ssh_keyfile=None):
        self.ips = ips
        self.client_id = clientId
        self.ssh_keyfile = ssh_keyfile
        self.ssh_mode = ssh_mode

        self.fit_path = path.join(path.dirname(
            __file__), 'serialized_fitting_description.capnp')

        self.fit_template = capnp.load(self.fit_path)

        self.description_template = self.fit_template.SerializedFittingDescription
        self.result_template = self.fit_template.SerializedFittingResult

        self.req_context = zmq.Context()
        self.req_socket = self.req_context.socket(zmq.REQ)
        self.sub_context = zmq.Context()
        self.sub_socket = self.sub_context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, str(self.client_id).encode())

        self.local_req_ip = 'tcp://127.0.0.1:5556'
        self.local_sub_ip = 'tcp://127.0.0.1:5557'

        self.connect()
        time.sleep(5)

    def connect(self):
        if self.ssh_mode:
            for ip in self.ips:
                print(self.ssh_keyfile)
                ssh.tunnel_connection(
                    self.req_socket, self.local_req_ip, ip, keyfile=self.ssh_keyfile)
                ssh.tunnel_connection(
                    self.sub_socket, self.local_sub_ip, ip, keyfile=self.ssh_keyfile)
        else:
            self.req_socket.connect(self.local_req_ip)
            self.sub_socket.connect(self.local_sub_ip)

    def issue_job(self, job):

        is_last = job.is_last

        sfd = self.description_template.new_message(
            timestamp=job.timestamp,
            instrumentationData=job.inst_data,
            clientId=self.client_id,
            configData=job.config_data, isLast=is_last,
            intensities=job.intensities, offsets=job.indices)

        self.req_socket.send(sfd.to_bytes())
        _ = self.req_socket.recv()

        if is_last:
            self.req_socket.close()

        _ = self.sub_socket.recv()
        message = self.sub_socket.recv()

        if is_last:
            self.sub_socket.close()

        dsfd = self.result_template.from_bytes(message)
        return FittingResult(dsfd)
