class MonteCarloES:
    def __init__(self, num_states, num_actions, state_transition_func):
        import numpy as np
        import random
        self.policies = np.zeros(num_states, np.int)
        self.state_action_values = np.zeros( (num_states, num_actions) , np.float64)
        self.state_action_rewards = np.zeros( (num_states, num_actions, 2) , np.int64 )
        self.state_transition_func = state_transition_func
        self.num_actions = num_actions
        self.num_states = num_states

    def run(self, num_episodes):
        for step in range(0, num_episodes):
            episode, total_reward = self.create_episode()
            self.recalculate_state_action_values(episode, total_reward)
            self.reevaluate_policies(episode)

    def create_episode(self):
        import random
        start_state = random.randint(0, self.num_states - 1)
        start_action = random.randint(0, self.num_actions - 1)
        next_state, reward = self.state_transition_func(start_state, start_action)
        episode = [(start_state, start_action, reward)]
        total_reward = reward

        while next_state != None:
            current_state = next_state
            next_state, reward = self.state_transition_func(current_state, self.policies[current_state])
            episode.append( (current_state, self.policies[current_state], reward) )
            total_reward += reward
        return episode, total_reward

    def recalculate_state_action_values(self, episode, total_reward):
        for state, action, reward in episode:
            self.state_action_rewards[state][action] += total_reward, 1
            self.state_action_values[state][action] = self.state_action_rewards[state][action][0] \
                / self.state_action_rewards[state][action][1]
            total_reward -= reward

    def reevaluate_policies(self, episode):
        import numpy as np
        for state, _, _ in episode:
            self.policies[state] = np.argmax(self.state_action_values[state])