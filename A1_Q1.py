import numpy as np

class Linear_Solver():
    """This class implements a linear solver for the problem posed by the first section of this assignment. The set-up indicated in https://onq.queensu.ca/d2l/le/content/718240/viewContent/4356243/View is followed.
    
    The default settings for this class, i.e., executed without parameters, are from the 5x5 case with a discount
    rate of 0.9, i.e., matching the example from the course textbook."""
    def __init__(self, discount_rate = 0.9, state_a = (0, 1), state_b = (0, 3),
    state_a_prime = (4, 1), state_b_prime = (2, 3), height = 5, width = 5, a_reward = 10, b_reward = 5):
        self.choice_probability = 0.25 
        self.discount_rate = discount_rate
        self.wall_reward = -1
        self.state_a = state_a
        self.state_b = state_b
        self.state_a_prime = state_a_prime
        self.state_b_prime = state_b_prime
        self.a_reward = a_reward
        self.b_reward = b_reward

        # Default height and width of Grid World is 5x5, 
        # but can be set to 7x7 as well (during initialization of Linear_Solver)
        self.height = height
        self.width = width
        self.grid_dim = self.height # Given that our grid is N x N, we can arbitrarily set `grid_dim` to `self.height`

    def prob_s_to_s_prime(self, state_s, state_s_prime, grid_dim):
        """
        Calculates the probability of transitioning from state s to state s' given the size of the grid and the location of the two states.
        """
        s_row = state_s // grid_dim
        s_col = state_s % grid_dim
        s_prime_row = state_s_prime // grid_dim
        s_prime_col = state_s_prime % grid_dim
     
        if s_row == self.state_a[0] and s_col == self.state_a[1]:
            if s_prime_row == self.state_a_prime[0] and s_prime_col == self.state_a_prime[1]:
                return 1
            else:
                return 0
        if s_row == self.state_b[0] and s_col == self.state_b[1]:
            if s_prime_row == self.state_b_prime[0] and s_prime_col == self.state_b_prime[1]:
                return 1
            else:
                return 0
        
        # If near a wall, probability of staying inside is 0.25
        if (state_s == state_s_prime):
            if (s_row == 0 and s_col == 0) or (s_row == grid_dim - 1 and s_col == grid_dim - 1):
                return 0.5
            elif (s_row == 0 and s_col == grid_dim - 1) or (s_row == grid_dim - 1 and s_col == 0):
                return 0.5
            elif s_row == 0 or s_col == grid_dim - 1 or s_row == grid_dim - 1 or s_col == 0:
                return 0.25

        # Manhattan distance between both points on grid must equal exactly 1, i.e.,
        # we are only interested in distance measured by the x and y axis
        if (abs(s_row - s_prime_row) + abs(s_col - s_prime_col)) == 1:
            return 0.25
        else:
            return 0

    def compute_reward_vec(self):
        """
        Computes the reward vector for each state in the gridworld. If a state is a terminal state (i.e., state_a or state_b), then the reward is fixed. Otherwise, the reward is a weighted sum of the rewards of the neighboring states (weighted by the probability of transitioning to each neighboring state).
        """
        num_entries = self.height * self.width
        grid_dim = self.grid_dim
        reward_vec = []
        for state_s in range(num_entries):
            weighted_reward_sum = 0
            s_row = state_s // grid_dim
            s_col = state_s % grid_dim

            if (s_row, s_col) == self.state_a:
                reward_vec.append(self.a_reward)
                continue
            elif (s_row, s_col) == self.state_b:
                reward_vec.append(self.b_reward)
                continue

            if (s_row - 1 == -1):
                weighted_reward_sum += (self.choice_probability * self.wall_reward)
            if (s_row + 1 == self.height):
                weighted_reward_sum += (self.choice_probability * self.wall_reward)
            if (s_col - 1 == -1):
                weighted_reward_sum += (self.choice_probability * self.wall_reward)
            if (s_col + 1 == self.width):
                weighted_reward_sum += (self.choice_probability * self.wall_reward)

            # All other choices of moves lead to cells, resulting in rewards of 0. There is no need to modify `weighted_reward_sum` for rewards of 0.
            reward_vec.append(weighted_reward_sum)

        return reward_vec

    def linear_solver(self):
        """
        Sets up the linear equation Ax=b, where A is a square matrix, x is the vector of state values, and b is the vector of rewards. It then solves for x using `np.linalg.solve`.
        """
        A_mat_dim = self.height * self.width
        # Prdouce a row for each of the (height * width) states
        p_mat_dim = self.height * self.width

        transition_mat = []
        for state_s in range(p_mat_dim):
            trans_mat_s_row = []
            for state_s_prime in range(p_mat_dim):
                s_to_s_prime_prob = self.prob_s_to_s_prime(state_s, state_s_prime, self.grid_dim)
                trans_mat_s_row.append(s_to_s_prime_prob)
            transition_mat.append(trans_mat_s_row)

        transition_mat = np.array(transition_mat)
        reward_vec = np.array(self.compute_reward_vec())
        """
        The following code is the set-up of the linear solver, Ax=b. Matrix
        A represents the identity matrix of a dimension of number of states by
        number of states minus discount factor multiplied by transition matrix of same dimension. b is the reward vector having a size equivalent to the number of states.

        The equation set-up follows the algebraic set-up outlined in the following two links:
            - https://onq.queensu.ca/d2l/le/content/718240/viewContent/4356243/View (Pg. 5)
            - https://cs.stackexchange.com/questions/142128/how-to-setup-the-bellman-equation-as-a-linear-system-of-equation
        """
        A_mat = np.identity(p_mat_dim) - (self.discount_rate * transition_mat)
        x_vec = np.linalg.solve(A_mat, reward_vec) # Given A and b from the matrix-vector product Ax=b, `np.linalg.solve` solves for this vector x 
        x_vec = x_vec.tolist()
        
        num_of_states = self.height * self.width

        print("State-value table for", self.grid_dim, "by", self.grid_dim, "case and discount rate of", str(self.discount_rate) + ":")
        for i in range(0, num_of_states, self.grid_dim):
            print(x_vec[i:(i+self.grid_dim)])

if __name__ == "__main__":   
    # Default setting
    solver_1 = Linear_Solver() 
    solver_1.linear_solver()
    print() # Printing of whitespace for better readability

    # 5x5 case with discount rate of 0.85
    solver_2 = Linear_Solver(discount_rate = 0.85, state_a = (0, 1), state_b = (0, 3),
    state_a_prime = (4, 1), state_b_prime = (2, 3), height = 5, width = 5, a_reward = 10, b_reward = 5) 
    solver_2.linear_solver()
    print()

    # 5x5 case with discount rate of 0.75
    solver_3 = Linear_Solver(discount_rate = 0.75, state_a = (0, 1), state_b = (0, 3),
    state_a_prime = (4, 1), state_b_prime = (2, 3), height = 5, width = 5, a_reward = 10, b_reward = 5) 
    solver_3.linear_solver()
    print()

    # 7x7 case with discount rate of 0.85
    solver_4 = Linear_Solver(discount_rate = 0.85, state_a = (2, 1), state_b = (0, 5),
    state_a_prime = (6, 1), state_b_prime = (3, 5), height = 7, width = 7, a_reward = 10, b_reward = 5)  
    solver_4.linear_solver()
    print()

    # 7x7 case with discount rate of 0.75
    solver_5 = Linear_Solver(discount_rate = 0.75, state_a = (2, 1), state_b = (0, 5),
    state_a_prime = (6, 1), state_b_prime = (3, 5), height = 7, width = 7, a_reward = 10, b_reward = 5)  
    solver_5.linear_solver()
    print()

