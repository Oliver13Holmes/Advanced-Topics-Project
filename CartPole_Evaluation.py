# Import Libraries #
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Create Environment #
env = gym.make("CartPole-v1", render_mode="human")  
obs_size = env.observation_space.shape[0]  
n_actions = env.action_space.n  
model_name = "cartpole-model.h5"
model = load_model(model_name)

# Heuristics #
heuristic1 = """def check_direction(a, b, c, d):
  if d > 0:
    return "RIGHT"
  else:
    return "LEFT"
"""
heuristic2 = """def check_direction(a, b, c, d):
  val = (c - a) * (d - b)
  if val > 0:
    return "RIGHT"
  else:
    return "LEFT"
"""
heuristic3 = """def check_direction(a, b, c, d):
  val = (b - d) * (c - a)
  if val > 0:
    return "RIGHT"
  elif val < 0:
    return "LEFT"
  else:
    return "LEFT"
"""
heuristic4 = """def check_direction(a, b, c, d):
  val = (b - d) * (c - a)
  if val > 0:
    return "RIGHT"
  elif val < 0:
    return "LEFT"
  else:
    return "RIGHT"
"""
heuristics = [heuristic1, heuristic2, heuristic3, heuristic4]

# Run Heuristic #
def run_code(code_to_run, parameters):
    # Runs the current code iteration with the parameters from the test cases #
    cp = parameters[0]
    cv = parameters[1]
    pa = parameters[2]
    pv = parameters[3]
    
    local_scope = {}
    exec(code_to_run, globals(), local_scope)

    func = list(local_scope.values())[0]
    result = func(cp, cv, pa, pv)

    return result

# Modified Action Selection #
def select_heuristic_action(state, heuristic):
    cp, cv, pa, pv = np.round(state, 3)
    parameters = [cp, cv, pa, pv]
    if run_code(heuristic, parameters) == "RIGHT":
       return 1
    else:
       return 0

# DQN Action Selection #
def select_DQN_action(state):
    q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(q_values[0])

# Final Evaluation #
test_episodes = 100
heuristic_test_rewards = []
DQN_test_rewards = []

# Heuristic loop #
for i, code in enumerate(heuristics):
    print(f"\nStarting final evaluation using LLM generated heuristic {i+1}...")
    rewards = []
    for ep in range(test_episodes):
        print("Episode: " + str(ep+1) + "/" + str(test_episodes))
        state = env.reset()[0]
        done = False
        total_reward = 0
        step_count = 0

        for t in range(500):
            action = select_heuristic_action(state, code)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1

            state = next_state
                
            if done:
                break
            
        rewards.append(total_reward)
    heuristic_test_rewards.append(rewards)

# DQN loop #
print("\nStarting final evaluation using DQN for comparison...")
for ep in range(test_episodes):
    print("Episode: " + str(ep+1) + "/" + str(test_episodes))

    state = env.reset()[0]
    done = False
    total_reward = 0
    step_count = 0

    for t in range(500):
        action = select_DQN_action(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

        state = next_state
            
        if done:
            break
        
    DQN_test_rewards.append(total_reward)

# Cleanup #
env.close()

# Calculate Average #
for i, scores in enumerate(heuristic_test_rewards):
    print(f"\nLLM Generated Heuristic {i+1} scored an average of: {np.mean(scores)}")
print(f"\nDQN scored an average of: {np.mean(DQN_test_rewards)}")

# Plot Results #
print("\nPlotting results...")
plt.plot(DQN_test_rewards, label="DQN")
for i, scores in enumerate(heuristic_test_rewards):
    plt.plot(scores, label=f"Heuristic {i+1}")
plt.legend()
#plt.xticks(ticks=range(test_episodes), labels=range(1, test_episodes+1))
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("CartPole Score Comparison")
plt.show()
