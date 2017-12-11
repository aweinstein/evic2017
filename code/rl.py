"""Fitting the parameters of an action-value model.

This module is the companion of the presentation "Modelos de toma de decisiones
de humanos basado en aprendizaje reforzado", presented by Alejandro Weinstein
at the XIIIIEEE Escuela de Verano Latino-Americana en Inteligencia
Computacional (EVIC 2017).
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 14


def softmax(Qs, beta):
    """Compute softmax probabilities for all actions.

    Parameters
    ----------
    Qs: array-like
        Action values for all the actions
    beta: float
        Inverse temperature

    Returns
    -------
    array-like
        Probabilities for each action
    """

    num = np.exp(Qs * beta)
    den = np.exp(Qs * beta).sum()
    return num / den


class Environment(object):
    "Environment implementation."
    def __init__(self):
        "Initialization of the actions and its associated probabilities."
        self.actions = ('square', 'triangle', 'circle')
        self.prob_win = {'square': 0.8,
                         'triangle': 0.5,
                         'circle': 0.2}
        self.n = len(self.actions)

    def reward(self, action):
        """Return the reward for a given action.

        Parameters
        ----------
        actions: {'square', 'triangle', 'circle'}
            Selected actions.

        Returns
        -------
        float
            Reward
        """
        if action not in self.actions:
            print('Error: action not in', self.actions)
            sys.exit(-1)
        p = self.prob_win[action]
        if np.random.rand() < p:
            r = 1.
        else:
            r = -1.
        return r


class Agent(object):
    """Implementation of an agent.

    The agent takes decisions based on the action-value model.
    """
    def __init__(self, environment, alpha, beta):
        """Set the agent parameters.

        Parameters
        ----------
        environment: Environment
            Instance of the Environment with which the agent will interact.
        alpha : float
            Learnig rate.
        beta : float
            Inverse temperature.
        """

        self.alpha = alpha
        self.beta = beta
        self.environment = environment
        self.actions = self.environment.actions
        self.n = environment.n
        # init with small random numbers to avoid ties
        self.Q = np.random.uniform(0, 1e-4, self.n)

        self.log = {'reward': [], 'action': [], 'Q': []}

    def run(self):
        """Choose an action."""
        action = self.choose_action()
        reward = self.environment.reward(self.actions[action])

        # Update action-value
        self.update_action_value(action, reward)

        # Keep track of performance
        self.log['reward'].append(reward)
        self.log['action'].append(self.actions[action])
        self.log['Q'].append(self.Q.copy())

    def choose_action(self):
        """Choose action following the softmax rule."""
        p = softmax(self.Q, self.beta)
        actions = range(self.n)
        action = np.random.choice(actions, p=p)
        return action

    def update_action_value(self, action, reward):
        """Update the action-value function.

        Parameters
        ----------
        action : int
            Index of the corresponding action.
        reward : float
            Observed reward.
        """
        self.Q[action] += self.alpha * (reward - self.Q[action])


def plot_simulation(agent):
    """Plot a realization of an agent-environment interaction.

    The plot shows the evolution of Q, the events, and the reward.
    """
    Q = np.stack(agent.log['Q'])
    actions = np.array(agent.log['action'])
    plt.close('all')
    fig, axs = plt.subplots(2, 1, figsize=(12, 10),
                            gridspec_kw={'height_ratios': [2, 1]})
    ax = axs[0]
    ax.plot(Q)
    ax.legend(('square', 'triangle', 'circle'), loc=2)
    ax.set_xlim((0, T - 1))
    ax.set_ylabel('Q')

    lo, ll = -0.9, 0.2
    ax.eventplot(np.where(actions == 'square')[0], colors='#1f77b4',
                 lineoffsets=lo, linelengths=ll)
    ax.eventplot(np.where(actions == 'triangle')[0], colors='#ff7f0e',
                 lineoffsets=lo, linelengths=ll)
    ax.eventplot(np.where(actions == 'circle')[0], colors='#2ca02c',
                 lineoffsets=lo, linelengths=ll)

    ax = axs[1]
    ax.stem(agent.log['reward'], basefmt='b-')
    ax.set_xlim((0, T - 1))
    ax.set_xlabel('trial')
    ax.set_yticks([-1, 1])
    ax.set_ylabel('reward')

    alpha, beta = agent.alpha, agent.beta
    title = r'Simulación con $\alpha=%.2f$ y $\beta=%.2f$' % (alpha, beta)
    fig.suptitle(title, y=0.92, fontsize=14)
    # plt.tight_layout(pad=8.5)
    plt.savefig('../figures/action_value_sim.pdf')
    plt.show()


class ML(object):
    """Estimation of the parameters of the action-value model.

    The parameters are estimated using the Maximum-Likielihood (ML) principle.
    """
    def __init__(self, log):
        """
        Parameters
        ----------
        log: dict
           Dictionary with the agent-environment obserbations. The dictionary
           must have the sequence of actions (with key `action`) and rewards
           (with key `reward`).
        """
        self.rewards = log['reward']
        label_to_n = {'square': 0, 'triangle': 1, 'circle': 2}
        self.actions = [label_to_n[a] for a in log['action']]
        self.n_actions = 3

    def neg_log_likelihood(self, alphabeta):
        """Compute the negative log likelihood function.

        Parameters
        ----------
        alphabeta : Tuple of floats
            Tuple of the form (alpha, beta), where alpha is the learning rate,
            and beta the inverse temperature parameters.

        Returns
        -------
        float :
            Negative log likelihood of the alpha-beta parameters.
        """
        alpha, beta = alphabeta
        prob_log = 0
        Q = np.zeros(self.n_actions)
        for action, reward in zip(self.actions, self.rewards):
            Q[action] += alpha * (reward - Q[action])
            prob_log += np.log(softmax(Q, beta)[action])

        return -prob_log

    def fit(self):
        """Fit the model."""
        bounds = ((0, 1), (0, 2))
        r = minimize(self.neg_log_likelihood, [0.1, 0.1],
                     method='L-BFGS-B',
                     bounds=bounds)
        return r

    def plot_ml(self, alpha, beta, alpha_hat, beta_hat):
        """Plot the likelihood function.

        Parameters
        ----------
        alpha, beta : float
            Real value of the parameters.
        alpha_hat, beta_hat : float
            Estimated value of the parameters.
        """

        from itertools import product
        n = 50
        alpha_max = 0.2
        beta_max = 1.5
        if alpha is not None:
            alpha_max = alpha_max if alpha < alpha_max else 1.1 * alpha
            beta_max = beta_max if beta < beta_max else 1.1 * beta
        if alpha_hat is not None:
            alpha_max = alpha_max if alpha_hat < alpha_max else 1.1 * alpha_hat
            beta_max = beta_max if beta_hat < beta_max else 1.1 * beta_hat
        alphas = np.linspace(0, alpha_max, n)
        betas = np.linspace(0, beta_max, n)
        Alpha, Beta = np.meshgrid(alphas, betas)
        Z = np.zeros(len(Alpha) * len(Beta))
        for i, (a, b) in enumerate(product(alphas, betas)):
            Z[i] = self.neg_log_likelihood((a, b))
        Z.resize((len(alphas), len(betas)))
        _, ax = plt.subplots(1, 1)
        ax.contourf(Alpha, Beta, Z.T, 50, cmap=plt.cm.jet)
        if alpha is not None:
            ax.plot(alpha, beta, 'rs', ms=7)
        if alpha_hat is not None:
            ax.plot(alpha_hat, beta_hat, 'r^', ms=7)
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'$\beta$', fontsize=20)
        return


if __name__ == '__main__':
    np.random.seed(42)
    env = Environment()
    alpha = 0.1
    beta = 0.5
    agent = Agent(env, alpha=alpha, beta=beta)

    T = 200
    for _ in range(T):
        agent.run()

    # plot_simulation(agent)
    ml = ML(agent.log)
    r = ml.fit()
    alpha_hat, beta_hat = r.x


    ml.plot_ml(alpha, beta, alpha_hat, beta_hat)
    plt.show()
