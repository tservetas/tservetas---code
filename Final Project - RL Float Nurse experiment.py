import numpy as np
import random
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

class ClinicQAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((num_states, num_actions))
        self.exploit_count = 0
        self.total_actions = 0

    def choose_action(self, state):
        self.total_actions += 1
        if random.uniform(0, 1) < self.exploration_prob:
            return random.choice(range(self.num_actions))  # Exploration
        else:
            self.exploit_count += 1
            return np.argmax(self.q_table[state, :])  # Exploitation

    def exploitation_rate(self):
        return self.exploit_count / self.total_actions if self.total_actions > 0 else 0

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action])

# Environment parameters
num_float_nurses = 10

#mean_arrival_rate_type1 = 3
std_dev_arrival_rate_type1 = 3
service_time_type1 = 1.5

#mean_arrival_rate_type2 = 1
std_dev_arrival_rate_type2 = 1
service_time_type2 = 2.5

# Define mean arrival rates for different time intervals
mean_arrival_rates = {
    'morning': {'type1': 2, 'type2': 1},
    'midday': {'type1': 5, 'type2': 3},
    'evening': {'type1': 3, 'type2': 2}
}

# RL agent
num_states = (num_float_nurses + 1) * (num_float_nurses + 1) * (num_float_nurses * 2 + 1)  # Considering both patient types
num_actions = num_float_nurses + 1
agent = ClinicQAgent(num_states=num_states, num_actions=num_actions)

# Simulation parameters
sim_duration_hours = 24
time_unit_per_hour = 10

# Function to determine mean arrival rate based on the hour
def get_mean_arrival_rate(hour, patient_type):
    if 0 <= hour < 10:  # Morning (12 AM - 10 AM)
        return mean_arrival_rates['morning'][patient_type]
    elif 10 <= hour < 16:  # Midday (10 AM - 4 PM)
        return mean_arrival_rates['midday'][patient_type]
    else:  # Evening (4 PM onwards)
        return mean_arrival_rates['evening'][patient_type]

    
# State encoding function
def encode_state(Nf, Np1, Np2):
    return Nf * (2 * num_float_nurses + 1) + Np1 * num_float_nurses + Np2

# Create a list to store the chosen actions, number of type 1 and type 2 patients, and reward at each whole hour
results_list = []

# Training loop
num_episodes = 10000
exploitation_rates = []

for episode in range(num_episodes):
    Nf = random.randint(0, num_float_nurses)
    state = encode_state(Nf, 0, 0)  # Initialize with no patients

    for hour in range(sim_duration_hours):
        action = agent.choose_action(state)

        # Simulate patient arrivals at each hour using the dynamic mean arrival rate
        type1_arrivals = max(int(norm.rvs(loc=get_mean_arrival_rate(hour, 'type1'), scale=std_dev_arrival_rate_type1)), 1)
        type2_arrivals = max(int(norm.rvs(loc=get_mean_arrival_rate(hour, 'type2'), scale=std_dev_arrival_rate_type2)), 1)

        # Update the number of patients in the system
        Np1 = type1_arrivals
        Np2 = type2_arrivals

        # Departure of patients after service
        Np1_served = min(Np1, action * 2)
        Np2_served = min(Np2, action)

        Np1 = max(0, Np1 - Np1_served)
        Np2 = max(0, Np2 - Np2_served)

        # Simulate transitioning to the next state after service
        next_Nf = min(num_float_nurses, max(0, Nf + action - (Np1 + Np2) // 2))

        next_state = encode_state(next_Nf, Np1, Np2)

        # Reward structure
        uncovered_penalty = -4 * (((Np1/2) + Np2) - Nf)
        extra_nurse_penalty = -5 * abs(action - Nf)
        optimal_coverage_reward = 10 if action == Nf else 0

        reward = uncovered_penalty + extra_nurse_penalty + optimal_coverage_reward

        # Update Q-table
        agent.update_q_table(state, action, reward, next_state)
        state = next_state

        # Append results to the list at whole hour marks
        results_list.append({
            "Hour": hour + 1,
            "Optimal Float Nurses": action,
            "Type 1 Patients": Np1,
            "Type 2 Patients": Np2,
            "Reward": reward
        })
    
    exploitation_rates.append(agent.exploitation_rate())

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_list)

# Average the results across all episodes for each hour
average_results_df = results_df.groupby("Hour").mean().reset_index()

# Display the DataFrame in the variable explorer
average_results_df

# Plotting the trends for each column
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

average_results_df.plot(x="Hour", y="Optimal Float Nurses", ax=axes[0, 0], legend=False, title="Optimal Float Nurses")
average_results_df.plot(x="Hour", y="Type 1 Patients", ax=axes[0, 1], legend=False, title="Type 1 Patients")
average_results_df.plot(x="Hour", y="Type 2 Patients", ax=axes[1, 0], legend=False, title="Type 2 Patients")
average_results_df.plot(x="Hour", y="Reward", ax=axes[1, 1], legend=False, title="Reward")

plt.tight_layout()
plt.show()

# Plotting the exploitation rate
plt.plot(range(num_episodes), exploitation_rates, label='Exploitation Rate', color='red')
plt.title('Exploitation Rate Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Exploitation Rate')
plt.legend()
plt.show()