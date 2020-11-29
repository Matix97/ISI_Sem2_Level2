import random
import matplotlib.pyplot as plt
from collections import defaultdict

class SARSAAgent:
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
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    # ---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
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
        # max_value = self.get_qvalue(state,possible_actions[0])
        # for i in range(1,len(possible_actions)):            
        #     max_value = max(max_value,self.get_qvalue(state, possible_actions[i]))
        values =[]
        for action in possible_actions:
            values.append(self.get_qvalue(state,action))
        return max(values)

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * Q(s', a'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value for the given state and action
        #
        #self.set_qvalue(state,action,
        #    (1-learning_rate)*self.get_qvalue(state,action)+learning_rate*(reward+gamma*self.get_qvalue(next_state,action)))
        next_action = self.get_action(next_state)
        self.set_qvalue(state,action,
            (1-learning_rate)*self.get_qvalue(state,action)+learning_rate*(reward+gamma*self.get_qvalue(next_state,next_action)))

        # function returns selected action for next state
        return next_action

    def get_best_action(self, state):
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
        best_value = self.get_qvalue(state,possible_actions[0])
        best_act = [possible_actions[0]]
        for i in range(1,len(possible_actions)):            
            temp = self.get_qvalue(state,possible_actions[i])
            if temp > best_value:
                best_act = [possible_actions[i]]
                best_value = temp
            elif temp == best_value:
                best_act.append(possible_actions[i])
        best_action = random.choice(best_act)

        return best_action

    def get_action(self, state):
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
        if random.random()<epsilon:
            chosen_action = random.choice(possible_actions)  
        else:   
            chosen_action = self.get_best_action(state)  

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0

class ExpectedSARSAAgent:
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

    def get_value(self, state):
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
        max_value = self.get_qvalue(state,possible_actions[0])
        for i in range(1,len(possible_actions)):            
            max_value = max(max_value,self.get_qvalue(state, possible_actions[i]))
        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * \sum_a \pi(a|s') Q(s', a))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value for the given state and action
        #
    
   
        #sprawdzenie prawdopodbieństwa
        # p=0
        # for a in possible_actions:
        #     if a==best_action:
        #         p+=(1-self.epsilon)
        #     else:
        #         p+=(self.epsilon/(len(possible_actions)-1))
        # print("Prawdopodbieśńtwo: ",p)

        sum = 0.0
        best_action = self.get_best_action(next_state)
        possible_actions = self.get_legal_actions(next_state)

        #p_best = 1 - self.epsilon #+ (self.epsilon / len(possible_actions))
        #p_other = self.epsilon/(len(possible_actions)-1)     #najlepszą akcję wybieramy prawie zawsze a pozostałe w tym espilonem równo 
        p_best = 1 - self.epsilon #+ (self.epsilon / len(possible_actions))
        if len(possible_actions)-1 != 0:
            p_other = self.epsilon/(len(possible_actions)-1)     #najlepszą akcję wybieramy prawie zawsze a pozostałe w tym espilonem równo 
        else:
            p_other = 0
            
        for a in possible_actions:
            if a==best_action:
                sum+=p_best*self.get_qvalue(next_state,a)
            else:
                sum+=p_other*self.get_qvalue(next_state,a)

        self.set_qvalue(state,action,
            (1-learning_rate)*self.get_qvalue(state,action)+learning_rate*(reward+gamma*sum))

  
    def get_best_action(self, state):
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

        best_value = self.get_qvalue(state,possible_actions[0])
        best_act = [possible_actions[0]]
        for i in range(1,len(possible_actions)):            
            temp = self.get_qvalue(state,possible_actions[i])
            if temp > best_value:
                best_act = [possible_actions[i]]
                best_value = temp
            elif temp == best_value:
                best_act.append(possible_actions[i])
        best_action = random.choice(best_act)

        return best_action

    def get_action(self, state):
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
        if random.random()<epsilon:#random.random() - domyślnie losuje z zakresu 0,1
            chosen_action = random.choice(possible_actions)#nasz element losowści       
        else:   
            chosen_action = self.get_best_action(state) 

        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0
