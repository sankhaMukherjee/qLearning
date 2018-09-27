import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = 1
        self.alpha = 0.1 # constant alpha training
        
        # For the epsilon-greedy policy, find a good method
        # for decreasing the epsilon over time.
        self.epsilon     = 0.5
        self.delEpsilon  = 0.1
        self.minEpsilon  = 0
        self.N           = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        # for a simple implementation, we shall use a simple Q-learning
        # policy. Remember that for Q-learning, the agent takes a step 
        # in an epsilon-greedy manner. 
        nA      = self.nA
        epsilon = self.epsilon
        
        greedyAction = np.argmax( self.Q[state] )
        greedyProb = 1 - epsilon + epsilon/nA
        otherProb  = epsilon/self.nA
        
        actionProbs = [(greedyProb if i == greedyAction else otherProb) for i in range(nA)]
        action      = np.random.choice(range(nA), p = actionProbs)
        return action
    
    def getProbs(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        # for a simple implementation, we shall use a simple Q-learning
        # policy. Remember that for Q-learning, the agent takes a step 
        # in an epsilon-greedy manner. 
        nA      = self.nA
        epsilon = self.epsilon
        
        greedyAction = np.argmax( self.Q[state] )
        greedyProb = 1 - epsilon + epsilon/nA
        otherProb  = epsilon/self.nA
        
        actionProbs = [(greedyProb if i == greedyAction else otherProb) for i in range(nA)]
        return actionProbs

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        nA  = self.nA
        Qs  = self.Q[state]
        Qsn = self.Q[next_state]
        g   = self.gamma
        probs = self.getProbs(next_state)
        
        if done:
            QsaV = (1-self.alpha)*Qs[action] + self.alpha*( reward )
        else:
            # SarsaMax
            # QsaV = (1-self.alpha)*Qs[action] + self.alpha*( reward + g*max(Qsn) )
            # Expected Sarsa
            QsaV = (1-self.alpha)*Qs[action] + self.alpha*( reward + g*np.sum(Qsn*probs) )
        
        self.Q[state][action] = QsaV
        
        self.N += 1
        
        self.epsilon = 1/self.N
        # if self.epsilon < self.minEpsilon:
        #     self.epsilon = 1/self.N
        # if self.epsilon < self.minEpsilon:
        #     self.epsilon *= self.delEpsilon
        
        