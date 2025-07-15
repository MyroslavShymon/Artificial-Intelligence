import numpy as np
import matplotlib.pyplot as plt
import random

# Фіксування seed для відтворюваності результатів
np.random.seed(42)
random.seed(42)

# Середовище "Лабіринт"
class LabyrinthEnv:
    def __init__(self, size, start, goal, traps, stochasticity=0.2,
                 reward_goal=100, reward_trap=-100, cost_move=-1):
        self.size = size
        self.start = start
        self.goal = goal
        self.traps = traps
        self.stochasticity = stochasticity
        self.reward_goal = reward_goal
        self.reward_trap = reward_trap
        self.cost_move = cost_move
        self.reset()

    def reset(self):
        self.agent_position = self.start
        return self.agent_position

    def step(self, action):
        actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Вгору, вниз, ліворуч, праворуч

        # Стохастичність: випадкова дія
        if random.random() < self.stochasticity:
            action = random.choice(list(actions.keys()))

        new_position = (self.agent_position[0] + actions[action][0],
                        self.agent_position[1] + actions[action][1])

        # Перевірка меж лабіринту
        if 0 <= new_position[0] < self.size[0] and 0 <= new_position[1] < self.size[1]:
            self.agent_position = new_position

        # Винагороди
        if self.agent_position == self.goal:
            reward = self.reward_goal
            done = True
        elif self.agent_position in self.traps:
            reward = self.reward_trap
            done = True
        else:
            reward = self.cost_move
            done = False

        return self.agent_position, reward, done

# Q-Learning алгоритм
def q_learning(env, episodes, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99):
    q_table = np.zeros((*env.size, 4))  # Q-таблиця для дій у кожному стані
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy політика
            if random.random() < epsilon:
                action = random.choice(range(4))
            else:
                action = np.argmax(q_table[state[0], state[1]])

            new_state, reward, done = env.step(action)
            total_reward += reward

            # Оновлення Q-значення
            best_next_action = np.max(q_table[new_state[0], new_state[1]])
            q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + \
                alpha * (reward + gamma * best_next_action - q_table[state[0], state[1], action])

            state = new_state

        # Зменшення epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

    return q_table, rewards

# Візуалізація маршруту агента
def visualize_path(env, q_table):
    state = env.reset()
    path = [state]
    done = False

    while not done:
        action = np.argmax(q_table[state[0], state[1]])
        state, _, done = env.step(action)
        path.append(state)

    grid = np.zeros(env.size)
    for trap in env.traps:
        grid[trap] = -1
    grid[env.goal] = 2
    for p in path:
        grid[p] = 0.5

    print("Шлях агента:")
    print(grid)

# Дослідження з різними значеннями винагород
cases = [
    {"reward_goal": 100, "reward_trap": -100, "cost_move": -1},
    {"reward_goal": 200, "reward_trap": -200, "cost_move": -1},
    {"reward_goal": 100, "reward_trap": -100, "cost_move": -5},
]

for i, case in enumerate(cases):
    print(f"\n--- Дослідження {i+1}: {case} ---")
    env = LabyrinthEnv(size=(5, 5), start=(0, 0), goal=(4, 4),
                       traps=[(2, 2), (3, 3)], stochasticity=0.2,
                       **case)
    q_table, rewards = q_learning(env, episodes=2000)

    # Побудова графіка сумарної винагороди
    plt.plot(rewards)
    plt.title(f"Сума винагород за епізод (Дослідження {i+1})")
    plt.xlabel("Епізод")
    plt.ylabel("Сума винагород")
    plt.show()

    # Візуалізація маршруту
    visualize_path(env, q_table)
