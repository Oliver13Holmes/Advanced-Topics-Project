# Basic implementation of a DQN agent for the CartPole problem

# Import Libraries #
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import random
from collections import deque

# Create Environment #
env = gym.make("CartPole-v1", render_mode="human")  
obs_size = env.observation_space.shape[0]  
n_actions = env.action_space.n  

# Hyperparameters #
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
batch_size = 64
memory_size = 10000
episodes = 300

recent_rewards = deque(maxlen=100)
model_name = "cartpole-model.h5"

# Replay Memory #
memory = deque(maxlen=memory_size)

# Build the DQN Model #
def build_model():
    model = tf.keras.Sequential([layers.Dense(128, activation='relu', input_shape=(obs_size,)), layers.Dense(n_actions, activation='linear')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mean_squared_error")
    return model


# Load or Initialize Model #
trained = False
try:
    model = load_model(model_name)
    print("Found saved model. Loading...")
    trained = True
except (IOError, tf.errors.NotFoundError):
    print("No saved model found. Starting training from scratch...")
    model = build_model()

# Epsilon-Greedy Action Selection #
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(q_values[0])

# Training Loop #
if not trained:
    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        step_count = 0

        print(f"\n=== Episode {ep + 1} ===")
        print(f"Epsilon: {epsilon:.3f}")

        for t in range(500):
            action = select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            step_count += 1
            total_reward += reward

            # Step Details #
            #print(f"Step {t+1}:")
            
            #print(f"  State: {np.round(state, 3)}")
            """cp, cv, pa, pv = np.round(state, 3)
            print(f"  State:")
            print(f"    Cart Position           : {cp}")
            print(f"    Cart Velocity           : {cv}")
            print(f"    Pole Angle              : {pa}")
            print(f"    Pole Angular Velocity   : {pv}")"""

            #print(f"  Action: {'LEFT' if action == 0 else 'RIGHT'}")
            #print(f"  Reward: {reward}")
            
            #print(f"  Next State: {np.round(next_state, 3)}")
            """ncp, ncv, npa, npv = np.round(next_state, 3)
            print(f"  Next State:")
            print(f"    Cart Position           : {ncp}")
            print(f"    Cart Velocity           : {ncv}")
            print(f"    Pole Angle              : {npa}")
            print(f"    Pole Angular Velocity   : {npv}")"""

            #print(f"  Done: {done}")

            state = next_state

            # Training Step #
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = np.array(states)
                next_states = np.array(next_states)
                q_values = model.predict(states, verbose=0)
                next_q_values = model.predict(next_states, verbose=0)

                for i in range(batch_size):
                    target = rewards[i]
                    if not dones[i]:
                        target += gamma * np.max(next_q_values[i])
                    q_values[i][actions[i]] = target

                model.fit(states, q_values, epochs=1, verbose=0)

            if done:
                break

        # End of Episode #
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {ep + 1} finished after {step_count} steps.")
        print(f"Total Reward: {total_reward}")
        recent_rewards.append(total_reward)

        # Evaluate Model #
        if len(recent_rewards) == 100:
            avg_reward = np.mean(recent_rewards)
            print(f"Average reward over last 100 episodes: {avg_reward:.2f}")
            if avg_reward >= 195: # Considered solved if average score is >= 195 over the past 100 episodes
                print("\nEarly stopping: Environment solved")
                model.save(model_name)
                trained = True
                break

# Final Evaluation #
print("\nStarting final evaluation with epsilon = 0 (pure exploitation)")
test_episodes = 1
test_rewards = []

for ep in range(test_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    step_count = 0

    print(f"\n=== Test Episode {ep + 1} ===")

    with open('testcases.txt', 'w') as file:
        for t in range(500):
            action = select_action(state, epsilon=0.0)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1

            # Step Details #
            #print(f"Step {t+1}:")
                
            #print(f"  State: {np.round(state, 3)}")
            
            cp, cv, pa, pv = np.round(state, 3)
            
            file.write(f"Input: {cp}, {cv}, {pa}, {pv} ")
            file.write(f"Output: {'LEFT' if action == 0 else 'RIGHT'}\n")
            
            #print(f"Input: {cp}, {cv}, {pa}, {pv}")
            #print(f"Output: {'LEFT' if action == 0 else 'RIGHT'}\n")

            """print(f"  State:")
            print(f"    Cart Position           : {cp}")
            print(f"    Cart Velocity           : {cv}")
            print(f"    Pole Angle              : {pa}")
            print(f"    Pole Angular Velocity   : {pv}")

            print(f"  Action: {'LEFT' if action == 0 else 'RIGHT'}")"""
            #print(f"  Reward: {reward}")
                
            #print(f"  Next State: {np.round(next_state, 3)}")
            """ncp, ncv, npa, npv = np.round(next_state, 3)
            print(f"  Next State:")
            print(f"    Cart Position           : {ncp}")
            print(f"    Cart Velocity           : {ncv}")
            print(f"    Pole Angle              : {npa}")
            print(f"    Pole Angular Velocity   : {npv}")"""

            #print(f"  Done: {done}")

            state = next_state
            
            if done:
                        break
        

    test_rewards.append(total_reward)
    print(f"Test Episode {ep + 1} finished after {step_count} steps. Total Reward: {total_reward}")

#avg_test_reward = np.mean(test_rewards)
#print(f"\nAverage reward over {test_episodes} test episodes: {avg_test_reward:.2f}")

# Cleanup #
env.close()
