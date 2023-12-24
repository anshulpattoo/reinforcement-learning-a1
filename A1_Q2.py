import numpy as np

class Iterative_Policy_Evaluation():
    """This class implements iterative policy evaluation of policy iteration for the problem posed by the first section of the assignment. The pseudocode for this algorithm is indicated in pg. 6 of lecture 3.1 (https://onq.queensu.ca/d2l/le/content/718240/viewContent/4281953/View).
    
    The default settings for this class, i.e., executed without parameters, are from the 5x5 case with a discount
    rate of 0.9, i.e., matching the example from the course textbook."""
    def __init__(self, discount_rate = 0.9, state_a = (0, 1), state_b = (0, 3),
    state_a_prime = (4, 1), state_b_prime = (2, 3), height = 5, width = 5, a_reward = 10, b_reward = 5, debug = False):
        self.action_choice_probability = 0.25
        self.discount_rate = discount_rate
        self.state_a = state_a
        self.state_b = state_b
        self.state_a_prime = state_a_prime
        self.state_b_prime = state_b_prime
        self.wall_reward = -1
        self.a_reward = a_reward
        self.b_reward = b_reward
        self.actions = ["NORTH", "WEST", "EAST", "SOUTH"]
        self.height = height
        self.width = width
        self.num_of_states = self.height * self.width
        self.grid_dim = self.height # Given that our grid is N x N, we can arbitrarily set `grid_dim` to `self.height`
        self.debug = debug

    def reward_state_s_action(self, state_s, action):
        """
        Returns the reward for a given state s and action, taking into account special states, wall regions, and corner cells.
        """
        grid_dim = self.height # Given that our grid is N x N, we can arbitrarily set `grid_dim` to `self.height`
        s_row = state_s // grid_dim
        s_col = state_s % grid_dim

        # If state s is a special state. Only one action is relevant, which is arbitrarily chosen to be "SOUTH" since the agent would move in a downward direction
        if s_row == self.state_a[0] and s_col == self.state_a[1]:
            if action == "SOUTH":
                return self.a_reward
            else:
                return 0
        elif s_row == self.state_b[0] and s_col == self.state_b[1]:
            if action == "SOUTH":
                return self.b_reward
            else:
                return 0

        # If state s is any one of the four corners
        if (s_row == 0 and s_col == 0) and ((action == "NORTH") or (action == "WEST")):
            return -1
        elif (s_row == 0 and s_col == (grid_dim - 1)) and ((action == "NORTH") or (action == "EAST")):
            return -1
        elif (s_row == (grid_dim - 1) and s_col == 0) and ((action == "SOUTH") or (action == "WEST")):
            return -1
        elif (s_row == (grid_dim - 1) and s_col == (grid_dim - 1)) and ((action == "SOUTH") or (action == "EAST")): 
            return -1
            
        # Wall regions, aside from the four corners above 
        # (i.e., none of the conditions below should be true if state s is a corner cell)
        if s_row == 0 and action == "NORTH":
            return -1
        elif s_row == (grid_dim - 1) and action == "SOUTH":
            return -1
        elif s_col == 0 and action == "WEST":
            return -1
        elif s_col == (grid_dim - 1) and action == "EAST":
            return -1
        
        return 0

    def get_ind_from_coords(self, state):
        """
        Given the row (i) and column (j) indices of a grid cell and the number of columns (self.width), 
        returns the index of the corresponding cell in a flattened grid.
        """
        i = state[0]
        j = state[1]
        return i * self.width + j

    def get_grid_coords(self, index):
        """
        Given the index of a cell in a flattened grid and the number of columns, returns the row and column indices of the corresponding cell in the grid.
        """
        n = self.height
        row = index // n
        col = index % n
        return (row, col)

    def prob_s_to_s_prime(self, state_s, state_s_prime, action):
        """
        Given two states (current and next) and an action, returns the probability of transitioning from the current state to the next state when the given action is taken.
        """
        grid_dim = self.grid_dim
        s_row = state_s // grid_dim
        s_col = state_s % grid_dim
        s_prime_row = state_s_prime // grid_dim
        s_prime_col = state_s_prime % grid_dim

        if s_row == self.state_a[0] and s_col == self.state_a[1]:
            if s_prime_row == self.state_a_prime[0] and s_prime_col == self.state_a_prime[1]:
                if action == "SOUTH": # There is a 100% probability of a single action occurring
                    return 1
            else:
                return 0
        elif s_row == self.state_b[0] and s_col == self.state_b[1]:
            if s_prime_row == self.state_b_prime[0] and s_prime_col == self.state_b_prime[1]:
                if action == "SOUTH": # There is a 100% probability of a single action occurring
                    return 1
            else:
                return 0
        
        # If near a wall, probability of staying inside is 0.25
        if (state_s == state_s_prime): # Whatever the action is, the probability for such particular action would always be 0.25
            if s_row == 0 and action == "NORTH":
                return 0.25
            elif s_row == grid_dim - 1 and action == "SOUTH":
                return 0.25
            elif s_col == 0 and action == "WEST":
                return 0.25
            elif s_col == grid_dim - 1 and action == "EAST":
                return 0.25
            else:
                return 0

        # Manhattan distance between both points on grid must equal exactly 1, i.e.,
        # we are only interested in distance measured by the x and y axis
        s_coords = self.get_grid_coords(state_s)
        s_prime_coords = self.get_grid_coords(state_s_prime)
        if (abs(s_row - s_prime_row) + abs(s_col - s_prime_col)) == 1:
            if action == "NORTH" and s_prime_coords[0] < s_coords[0]:
                return 0.25
            elif action == "SOUTH" and s_prime_coords[0] > s_coords[0]:
                return 0.25
            elif action == "EAST" and s_prime_coords[1] > s_coords[1]:
                return 0.25
            elif action == "WEST" and s_prime_coords[1] < s_coords[1]:
                return 0.25
        
        return 0

    def iterative_policy_evaluation(self):
        """
        Runs the iterative policy evaluation algorithm to compute the state-value function for a given policy.
        """
        # Small threshold theta > 0 determining accuracy of estimation
        threshold_theta = 0.0001
        delta = np.inf
        # Initialized state-value table for all states s; stored as a one-dimensional array
        state_val_table_vec = [0] * self.num_of_states

        while delta >= threshold_theta:
            delta = 0
            for state_s_ind in range(len(state_val_table_vec)):
                var_v = state_val_table_vec[state_s_ind]
                a_sum = 0 # `a_sum` denotes "V(s)", i.e., the sum across all possible actions

                for action in self.actions:
                    s_prime_r_sum = 0
                    for state_s_prime_ind in range(len(state_val_table_vec)):
                        reward = self.reward_state_s_action(state_s_ind, action) 
                        probability = self.prob_s_to_s_prime(state_s_ind, state_s_prime_ind, action)
                        s_prime_r_sum += probability * (reward + (self.discount_rate * state_val_table_vec[state_s_prime_ind])) # 

                        if self.debug == True:
                            print("State s:", state_s_ind)
                            print("State s prime:", state_s_prime_ind)
                            print("Reward:", reward)
                            print("Action:", action)
                            print("Probability:", probability)
                    a_sum += s_prime_r_sum

                state_val_table_vec[state_s_ind] = a_sum
                delta = max(delta, abs(var_v - state_val_table_vec[state_s_ind]))

        num_of_states = self.height * self.width

        print("State-value table for", self.grid_dim, "by", self.grid_dim, "case and discount rate of", str(self.discount_rate) + ":")
        for i in range(0, num_of_states, self.grid_dim):
            print(state_val_table_vec[i:(i+self.grid_dim)])
   
if __name__ == "__main__":
    # Default setting 
    iterative_policy_evaluation_1 = Iterative_Policy_Evaluation()
    iterative_policy_evaluation_1.iterative_policy_evaluation()
    print() # Printing of whitespace for better readability

    # 5x5 case with discount rate of 0.85
    iterative_policy_evaluation_2 = Iterative_Policy_Evaluation(discount_rate = 0.85, state_a = (0, 1), state_b = (0, 3),
    state_a_prime = (4, 1), state_b_prime = (2, 3), height = 5, width = 5, a_reward = 10, b_reward = 5) 
    iterative_policy_evaluation_2.iterative_policy_evaluation()
    print()

    # 5x5 case with discount rate of 0.75
    iterative_policy_evaluation_3 = Iterative_Policy_Evaluation(discount_rate = 0.75, state_a = (0, 1), state_b = (0, 3),
    state_a_prime = (4, 1), state_b_prime = (2, 3), height = 5, width = 5, a_reward = 10, b_reward = 5) 
    iterative_policy_evaluation_3.iterative_policy_evaluation()
    print()

    # 7x7 case with discount rate of 0.85
    iterative_policy_evaluation_4 = Iterative_Policy_Evaluation(discount_rate = 0.85, state_a = (2, 1), state_b = (0, 5),
    state_a_prime = (6, 1), state_b_prime = (3, 5), height = 7, width = 7, a_reward = 10, b_reward = 5)  
    iterative_policy_evaluation_4.iterative_policy_evaluation()
    print()

    # 7x7 case with discount rate of 0.75
    iterative_policy_evaluation_5 = Iterative_Policy_Evaluation(discount_rate = 0.75, state_a = (2, 1), state_b = (0, 5),
    state_a_prime = (6, 1), state_b_prime = (3, 5), height = 7, width = 7, a_reward = 10, b_reward = 5)  
    iterative_policy_evaluation_5.iterative_policy_evaluation()
    print()
