from mpi4py import MPI
from typing import Dict
from repast4py import parameters
from My_Model import Model

model = None


def run(params: Dict):
    # generate_network_file('SIT_network', params['network_file'], params['n_kernels'], params['n_agents'])
    # uncomment to generate new network. Currently using 1000 agents on 16 kernels
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
