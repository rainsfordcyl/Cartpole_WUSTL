import gym
import numpy as np
from model import DQNAgent

env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

agent.load("src/checkpoints/cartpole-dqn-0.h5")

def test_agent():
    episode_scores = []

    for episode in range(10):
        initial_state_tuple = env.reset()
        initial_state = initial_state_tuple[0]
        state = np.reshape(initial_state, [1, state_size])

        for t in range(500):
            env.render()
            
            action = agent.act(state, mode='test')
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state)
            next_state = np.reshape(next_state, [1, state_size])
            
            state = next_state
            
            if done:
                print(f"Episode: {episode}, Score: {t}")
                episode_scores.append(t)
                break

        average_score = np.mean(episode_scores)
        print(f"Average Score Over 10 Episodes: {average_score}")

test_agent()