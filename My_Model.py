# My_Model.py

import networkx as nx
from dataclasses import dataclass
import random
import numpy as np

from repast4py import context as ctx
from repast4py import schedule, logging
from repast4py.network import write_network, read_network
from Agent import create_AgentSIT, restore_agent


def generate_network_file(fname: str, fpath: str, n_ranks: int, n_agents: int):
    """Generates a network file using repast4py.network.write_network.
     Args:
        fname: the name of the file to write to
        fpath: the path to the file
        n_ranks: the number of process ranks to distribute the file over
        n_agents: the number of agents (node) in the network
    """
    g = nx.watts_strogatz_graph(n_agents, k=4, p=0.2, seed=42)
    print(g)
    write_network(g, fname, fpath, n_ranks, partition_method='random', rng='default')


@dataclass
class OpinionDistribution:
    mean_opinion: float = 777
    min_opinion: float = 777
    max_opinion: float = 777


class Model:
    """The Model itself

    Defines
         duration of simulation,
         logging,
         what happens at each step


       """

    def __init__(self, comm, params):

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)  # executing 'step' every tick
        self.runner.schedule_repeating_event(0, 1, self.log_agents)  # currently logging opinion of single agents
        # every 10 ticks, starting at tick 0
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        fpath = params['network_file']
        self.context = ctx.SharedContext(comm)
        read_network(fpath, self.context, create_AgentSIT, restore_agent)
        self.net = self.context.get_projection('SIT_network')

        self.rank = comm.Get_rank()

        self.agent_logger = logging.TabularLogger(comm, params['distribution_file'],
                                                  ['tick', 'agent_id', 'empathy', 'ojs', 'vjs',  'ambigtol', 'outrage',
                                                   'opinion1', 'opinion2',
                                                   'opinion3', 'opinion4', 'opinion5'])

        self.all_agents = {}

        self.params = params
        # generating dictionary in which all agents are stored with their id and their opinion,
        # so I can access the t-1 opinion to update agents
        for agent_id in self.context.agents():  # random selection of agents with (count = n, shuffle = True)
            self.all_agents[agent_id.uid] = 0

    def at_end(self):
        self.agent_logger.close()

    def update_agent(self):

        """ Updating Agents

        At each step, agents are updated in the following way
        - a topic is chosen randomly.
        - agents interact with all their neighbours that lie within their individual confidence bound for this topic.
        - agents are then influenced towards the mean of the opinion of all these neighbours.

        """
        for agent_id in self.all_agents:
            self.context.agent(agent_id).old_opinion = self.context.agent(agent_id).opinion
        for agent_id in self.all_agents:
            agent = self.context.agent(agent_id)
            count = 0
            val_sum = 0
            i = random.choice(range(5))
            for ngh in self.net.graph.neighbors(agent):
                if agent.group == ngh.group:
                    group_influence = 0.9  # having the neighbour in the same group as the agent decreases the
                else:  # agents' self-influence
                    group_influence = 1.0
                if abs(ngh.old_opinion[i] - agent.old_opinion[i]) <= (agent.c_ind):  # currently c_ind around 5
                    val_sum += ngh.old_opinion[i]  # currently using all neighbours
                    count += 1
            if count > 0:
                val_mean = val_sum / count
            else:
                val_mean = agent.old_opinion[i]
            agent.opinion[i] = (agent.m[i] * agent.old_opinion[i]) + ((1 - agent.m[i]) * val_mean)
            self.all_agents[agent_id] = agent.opinion

        opinions = []
        for i in range(4):
            opinions.append(np.asarray([float(item[i]) for item in (list(self.all_agents.values()))]))

        self.context.synchronize(restore_agent)

    def step(self):

        """ The updating process

        at each step, an agent updates their belief according to their neighbours' beliefs.
        Only neighbours are chosen that differ from agent by less than c.
        The weight of the neighbours' opinion in comparison to own opinion is m.

           """

        print('Hello, I\'m working!')
        self.update_agent()

    def log_agents(self):
        tick = self.runner.schedule.tick
        for agent in self.context.agents():
            self.agent_logger.log_row(tick, agent.id, agent.empathy, agent.ojs, agent.vjs,
                                      agent.ambigtol, agent.outrage, agent.opinion[0],
                                      agent.opinion[1], agent.opinion[2], agent.opinion[3], agent.opinion[4])
        self.agent_logger.write()

    def start(self):
        self.runner.execute()
