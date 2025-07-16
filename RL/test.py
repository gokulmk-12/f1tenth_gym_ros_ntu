import time
import numpy as np
from RL.algos.ppo import PPO
from rich.console import Console
from f1tenth_gym_ros.simple_rl import F110Gym

console = Console()

def test():

    env_name = "racecar"
    max_ep_len = 1000 

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 10               # update policy for K epochs
    clip_param = 0.2              # clip parameter for PPO
    gamma = 0.995                # discount factor

    learning_rate = 5e-4           # learning rate

    #####################################################

    env = F110Gym()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo_agent = PPO(
        state_dim, 
        action_dim,
        K_epochs,
        learning_rate,
        gamma,
        clip_param,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        schedule="adaptive")

    weights_dir = f"PPO_pretrained/{env_name}/"
    checkpoint_path = weights_dir + f"/PPO_{env_name}.pth"
    console.print("[RL-Test] Loading network from : " + checkpoint_path, style="green")

    ppo_agent.load(checkpoint_path)

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state, _ = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward

            if done:
                break
            
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        console.print('[RL-Test] Episode: {} Reward: {}'.format(ep, round(ep_reward, 2)), style="red")
        ep_reward = 0


    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    console.print("[RL-Test] Average test reward : " + str(avg_test_reward), style="green")

    print("============================================================================================")


if __name__ == '__main__':

    test()