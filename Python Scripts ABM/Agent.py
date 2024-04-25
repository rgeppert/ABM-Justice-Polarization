import numpy as np
from repast4py import core, parameters
import pickle


def create_AgentSIT(nid, type, rank, **kwargs):
    """Calls the initialization of a new instance of the agent class
    """
    return AgentSIT(nid, type, rank)



def restore_agent(agent_data):
    """Creates an agent from the specified agent_data.

    Args:
    agent_data: This is the tuple returned from the agent's save() method
                where the first element is the agent id tuple,
                followed by their opinion.
    """

    uid = agent_data[0]
    opinion = agent_data[1]
    return AgentSIT(uid[0], uid[1], uid[2], opinion[0])


class AgentSIT(core.Agent):
    """The SIT Agent

       Args:
           nid: an integer that uniquely identifies this agent on its
                 starting rank
           rank: the starting MPI rank of this agent
           opinion: a vector of length d, in which the agent stores their opinion on d different topics
           d: the number of topics the agent has an opinion on

    """

    TYPE = 0

    def __init__(self, nid: int, rank: int, d: int):
        super().__init__(id=nid, type=AgentSIT.TYPE, rank=rank)

        parser = parameters.create_args_parser()
        args = parser.parse_args()
        params = parameters.init_params(args.parameters_file, args.parameters)

        self.old_opinion = 0
        # generate correlated arrays (these include opinion and certainty for five topics)
        num_samples = params['n_topics']
        # The desired mean values of the sample.
        mu = np.array([5, 0.5])
        # The desired variances of the sample.
        sigma = np.array([2, 0.2])
        # The desired correlation matrix.
        cor = np.array([[1, 0.7],
                        [0.7, 1]])
        # create covariance matrix
        r = sigma * cor
        # Generate the random samples.
        rng = np.random.default_rng()
        y = rng.multivariate_normal(mu, r, size=num_samples)
        yt = np.transpose(y)

        # self.opinion = []
        # for o in yt[0]:
        # for o in o_uni:
        #     if o % 2 == 0 and o > 5:
        #         self.opinion.append(10 - o)
        #     else:
        #         self.opinion.append(o)

        # attempt what happens if the opinions are initially uniformly distributed
        o_uni = np.random.uniform(1, 10, 5)
        self.opinion = o_uni
        # bnd_o = [0, 10]
        # for o in self.opinion:
        #     self.opinion[list(self.opinion).index(o)] = \
        #         np.interp(self.opinion[list(self.opinion).index(o)], bnd_o, [0, 10])

        self.certainty = yt[1]
        bnd_c = [min(self.certainty) - 0.2, max(self.certainty) + 0.2]
        for c in self.certainty:
            self.certainty[list(self.certainty).index(c)] = \
                np.interp(self.certainty[list(self.certainty).index(c)], bnd_c, [0, 1])

        # read the empirical distribution
        with open('./../resources/empathy.pkl', 'rb') as f:
            empathy = pickle.load(f)
        with open('./../resources/ambigtol.pkl', 'rb') as f:
            ambigtol = pickle.load(f)
        with open('./../resources/outrage.pkl', 'rb') as f:
            outrage = pickle.load(f)
        with open('./../resources/ojs.pkl', 'rb') as f:
            ojs = pickle.load(f)
        with open('./../resources/vjs.pkl', 'rb') as f:
            vjs = pickle.load(f)
        with open('./../resources/soccomp.pkl', 'rb') as f:
            socialcomp = pickle.load(f)

        n = 1

        self.empathy = float(np.random.choice(empathy._mcpts, n, replace=False))
        self.ambigtol = float(np.random.choice(ambigtol._mcpts, n, replace=False))
        self.outrage = float(np.random.choice(outrage._mcpts, n, replace=False))
        self.ojs = float(np.random.choice(ojs._mcpts, n, replace=False))
        self.vjs = float(np.random.choice(vjs._mcpts, n, replace=False))
        self.socialcomp = float(np.random.choice(socialcomp._mcpts, n, replace=False))

        # how much agents are influences by other agents is dependent on opinion certainty and empathy
        self.m = np.exp(self.empathy + (-1 * self.certainty)) / (1 + np.exp(self.empathy + (-1 * self.certainty)))
        # print('m = ', self.m)

        # who agents talk to is dependent on js-measures, outrage and ambigtol
        self.c_ind = -4 + 6.62793 + 0.75194*self.ambigtol -1.17661*self.outrage + 0.28118*self.ojs -1.19446*self.vjs - \
                     0.47566*self.socialcomp

        self.group = None

    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its opinion using this
        method, and any ghost agents of this agent will
        be updated with that data (self.opinion).

        Returns:
            The agent's uid and their opinion
        """
        return (self.uid, self.opinion)

    def update(self, data: int):
        """Updates the opinion of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent opinion (opinion_new)
        """
        if self.opinion != data:
            # only update if the opinion has changed
            self.opinion = data
