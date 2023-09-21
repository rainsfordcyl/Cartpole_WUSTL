import numpy as np
import os
import gym
import random
from model import DQNAgent

EPISODES = 50

def find_next_model_index(directory='src/checkpoints', prefix='cartpole-dqn-'):
    existing_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.h5')]
    existing_indices = [int(f.replace(prefix, '').replace('.h5', '')) for f in existing_files]
    next_index = max(existing_indices, default=-1) + 1
    return next_index

if __name__ == "__main__":
    next_model_index = find_next_model_index()

    env = gym.make('CartPole-v1', render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for e in range(EPISODES):
        # reset the environment
        original_state = env.reset()
        state = np.array(original_state[0])  # Extract the numpy array from the tuple
        state = np.reshape(state, [1, state_size])
        
        
        for time in range(500):
            env.render()
            
            # Control Policy: Decide Action
            action = agent.act(state)

            # Environment: Take Action, Observe Reward and Next State
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Control Policy: Update
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}")
                break

            ## Training Episodes Log: load and save model, value
            # Make it works
            # Save and Run the Training Log
            # Categorize the code to Control Policy, Environment, Training Program, RL Structure

            # update the control policy
            if len(agent.memory) > batch_size: # Batch_size : only part of the trajectory;
                agent.replay(batch_size)
        
        # Save the model every 10 episodes
        if e % 10 == 0:
            checkpoint_path = f"src/checkpoints/cartpole-dqn-{next_model_index}.h5"
            agent.save(checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
            next_model_index += 1

env.close()