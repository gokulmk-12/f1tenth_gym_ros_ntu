import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from datetime import datetime
from rich.console import Console

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

console = Console()

class PPO:
    def __init__(self, config: dict, enable_logging: bool = False):
        self.args = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args["cuda"] else "cpu")
        self.enable_logging = enable_logging

        self.args["batch_size"] = int(self.args["num_envs"] * self.args["num_steps"])
        self.args["minibatch_size"] = int(self.args["batch_size"] // self.args["num_minibatches"])

        if self.enable_logging:
            run_name = f"{self.args['gym_id']}__{self.args['exp_name']}__{self.args['seed']}__{int(time.time())}"
            self.writer = SummaryWriter(f"runs/{run_name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in self.args.items()])),
            )
        else:
            self.writer = None
        
        random.seed(self.args["seed"])
        np.random.seed(self.args["seed"])
        torch.manual_seed(self.args["seed"])
        torch.backends.cudnn.deterministic = self.args["torch_deterministic"]

        self.envs = gym.vector.SyncVectorEnv(
            [self.make_env(self.args["gym_id"], self.args["seed"] + i) for i in range(self.args["num_envs"])]
        )

        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.args["learning_rate"], eps=1e-5)

        self.obs = torch.zeros((self.args["num_steps"], self.args["num_envs"]) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.args["num_steps"], self.args["num_envs"]) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.args["num_steps"], self.args["num_envs"])).to(self.device)
        self.rewards = torch.zeros((self.args["num_steps"], self.args["num_envs"])).to(self.device)
        self.dones = torch.zeros((self.args["num_steps"], self.args["num_envs"])).to(self.device)
        self.values = torch.zeros((self.args["num_steps"], self.args["num_envs"])).to(self.device)

        console.print(f"[PPO] Num Env: [{self.args['num_envs']}], Device: [{str(torch.cuda.get_device_name(self.device))}]", style="cyan")
    
    def make_env(self, gym_id, seed):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    def train(self):
        global_step = 0
        start_time = time.time()
        next_obs = torch.tensor(self.envs.reset()[0]).to(self.device)
        next_done = torch.zeros(self.args["num_envs"]).to(self.device)
        num_updates = int(self.args["total_timesteps"] // self.args["batch_size"])

        for update in range(1, num_updates + 1):
            if self.args["anneal_lr"]:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args["learning_rate"]
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.args["num_steps"]):
                global_step += self.args["num_envs"]
                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                next_obs, reward, done, _, _ = self.envs.step(action.cpu().numpy())
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            print(update)
            avg_reward = round(torch.mean(self.rewards).detach().item(), 3)
            console.print(f"[Training] [{timestamp()}] Steps: {global_step} Average Reward: {avg_reward}", style="yellow")

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)

                if self.args["gae"]:
                    advantages = torch.zeros_like(self.rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.args["num_steps"])):
                        if t == self.args["num_steps"] - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]
                        delta = self.rewards[t] + self.args["gamma"] * nextvalues * nextnonterminal - self.values[t]
                        advantages[t] = lastgaelam = delta + self.args["gamma"] * self.args["gae_lambda"] * nextnonterminal * lastgaelam
                    returns = advantages + self.values

                else:
                    returns = torch.zeros_like(self.rewards).to(self.device)
                    for t in reversed(range(self.args["num_steps"])):
                        if t == self.args["num_steps"] - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = self.rewards[t] + self.args["gamma"] * nextnonterminal * next_return
                    advantages = returns - self.values

            self._optimize(returns, advantages, global_step, start_time)

        self.envs.close()
        if self.writer:
            self.writer.close()

    def _optimize(self, returns, advantages, global_step, start_time):
        b_obs = self.obs.reshape((-1, ) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1, ) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        b_inds = np.arange(self.args["batch_size"])
        clipfracs = []

        for epoch in range(self.args["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, self.args["batch_size"], self.args["minibatch_size"]):
                end = start + self.args["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args["clip_coef"], 1 + self.args["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.args["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -self.args["clip_coef"], self.args["clip_coef"])
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args["ent_coef"] * entropy_loss + self.args["vf_coef"] * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args["max_grad_norm"])
                self.optimizer.step()

                if self.args["target_kl"] is not None and approx_kl > self.args["target_kl"]:
                    break

        if self.writer:
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    def save(self, checkpoint_path):
        torch.save({
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.agent.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if __name__ == "__main__":
    
    config = {
        "exp_name": "",
        "gym_id": "HalfCheetah-v5",
        "seed": 1,
        "torch_deterministic": True,
        "cuda": True,
        "total_timesteps": 2e6,
        "num_envs": 1,
        "num_steps": 2048,
        "num_minibatches": 32,
        "learning_rate": 3e-4,
        "anneal_lr": True,
        "gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "update_epochs": 4,
        "norm_adv": True,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None
    }

    ppo = PPO(config, enable_logging=True)
    ppo.train()