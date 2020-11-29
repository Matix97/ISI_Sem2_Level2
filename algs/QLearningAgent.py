import random
import matplotlib.pyplot as plt
from collections import defaultdict


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        # print()
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):  # najlepsza akcja dla danego stanu
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #
        # print(possible_actions)
        # Q(state,action) = self.get_qvalue(state,action)
        max_value = self.get_qvalue(state, possible_actions[0])
        for i in range(1, len(possible_actions)):
            max_value = max(max_value, self.get_qvalue(
                state, possible_actions[i]))

        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value for the given state and action
        #
# Q(state,action) = self.get_qvalue(state,action)

        self.set_qvalue(state, action,  # akutalizujemy naszą obecną wartość o learning_rate*(reward+gamma*self.get_value(next_state))
                        (1-learning_rate)*self.get_qvalue(state, action)+learning_rate*(reward+gamma*self.get_value(next_state)))

    def get_best_action(self, state):  # slajd 31
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #
        best_value = self.get_qvalue(state, possible_actions[0])
        best_act = [possible_actions[0]]
        # best_act.append(possible_actions[0])
        for i in range(1, len(possible_actions)):
            temp = self.get_qvalue(state, possible_actions[i])
            if temp > best_value:
                best_act = [possible_actions[i]]
                # best_act.append(possible_actions[i])
                best_value = temp
            elif temp == best_value:
                best_act.append(possible_actions[i])
        # no i jak mamy więc to rand:
        #print("BEST ACTIONS:",best_act)
        best_action = random.choice(best_act)
        #print("ONE BEST ACTION:",best_action)

        return best_action

    def get_action(self, state):  # slajd 31
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #
        # chosen_action = self.get_best_action(state) w sumie w else będzie szybciej
        if random.random() < epsilon:  # random.random() - domyślnie losuje z zakresu 0,1
            chosen_action = random.choice(
                possible_actions)  # nasz element losowści
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0
