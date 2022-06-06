import time

import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import gym
from network import FeedForwardNN


class PPO:
    def __init__(self, env):
        # 环境
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # 神经网络
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        # 超参数
        self._init_hyperparameters()
        # 协方差矩阵
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        # 优化器
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        # 日志记录
        self.logger = {
            'delta_t': time.time_ns(),  # 获取当前的时间，单位为ns（纳秒）
            't_so_far': 0,  # 目前已经训练过多少step
            'i_fo_far': 0,  # 目前已经训练过多少次iteration，每次iteration都会采集一个batch的数据并根据这个batch的数据来进行多次epoch的更新
            'batch_lens': [],  # 记录batch中每个episode的step数目
            'batch_rews': [],  # 记录batch中每个step的reward
            'actor_losses': [],  # 记录当前迭代中actor网络的loss（也就是J(theta)）
        }

    """参数total_timesteps指定了学习任务总共要学多少个step"""
    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # 目前为止已经训练了多少step
        i_so_far = 0  # 目前为止训练了多少个iteration
        while t_so_far < total_timesteps:

            """ 使用rollout来获取一个batch的数据，一个batch中包含多个episode，每个episode包含多个step"""
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)
            i_so_far += 1    # 一个iteration会采集一个batch的数据并用该batch训练actor和critic网络epoch次
            """日志记录"""
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # 使用critic网络计算每个step的V值
            V, _ = self.evaluate(batch_obs, batch_acts)

            """根据rtg和V计算advantage的值，V的值要detach()一下从而从critic网络的计算图中脱离出来
            因为这里V的值是用来计算A_k进而计算actor_loss的，如果不从critic网络的计算图中脱离出来，当执行
            actor_loss.backward()时候将会导致critic网络发生变化 """
            A_k = batch_rtgs - V.detach()
            # advantage的值归一化，这是一个优化的trick
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            """每次update要包含多次epoch，为什么请看：
            https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
            这个回答下面的 2. Multiple epochs for policy updating """
            for _ in range(self.n_updates_per_iteration):

                """计算当前epoch中policy（当前batch中的动作在（也就是旧的actor网络输出的动作）
                经过update后的actor网络对应的策略中的的log_probs，即log pi_{theta}(a_t|s_t)，其中s_t和a_t都是batch中的数据
                这个batch是通过旧的策略log pi_{theta'}(a_t|s_t)生成的"""
                V, current_log_probs = self.evaluate(batch_obs, batch_acts)
                # 这里根据exp(log(P_1) - log(P_2)) = exp(log(P_1/P_2))=P_1/P_2计算新策略与旧策略概率之间的ratio
                ratios = torch.exp(current_log_probs - batch_log_probs)

                # """根据rtg和V计算advantage的值，V的值要detach()一下从而从critic网络的计算图中脱离出来
                # 因为这里V的值是用来计算A_k进而计算actor_loss的，如果不从critic网络的计算图中脱离出来，当执行
                # actor_loss.backward()时候将会导致critic网络发生变化 """
                # A_k = batch_rtgs - V.detach()
                # # advantage的值归一化，这是一个优化的trick
                # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


                """实现Clip-PPO的surrogate objective function，也就是PPO的核心"""
                surr1 = ratios * A_k
                clamp_ratio = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                surr2 = clamp_ratio * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()

                """根据PPO的loss来更新actor (policy)网络"""
                self.actor_optim.zero_grad()
                actor_loss.backward()  # 这里保存计算图是因为多次更新（多个epoch）
                self.actor_optim.step()

                """根据MSELoss function来更新critic网络"""
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                """记录actor的loss"""
                self.logger['actor_losses'].append(actor_loss.detach())
            """完成当前（iteration）batch数据的训练之后，重新采集数据，直到达到训练最大的step数
            下面是打印当前iteration的日志"""
            self._log_summary()

            # 保存模型
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')


    def evaluate(self, batch_obs, batch_acts):
        """critic网络的输入:（batch_step_num,obs_dim）,输出（batch_step_num,1）,
        所以，需要使用squeeze来去掉1这个维度的括号，eg:[[2,3,12,41,51]]->[2,3,12,41,51]"""
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)  # actor网络的输出是高斯分布的均值
        """根据actor网络吐出的参数mean和设定的协方差矩阵（也就是超参数）来使用Pytorch的函数产生一个多元高斯分布，
        然后根据batch中的action来获得action对应的log_probs"""
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048  # 每个batch包含的step数目
        self.max_timesteps_per_episode = 200  # batch中每个episode的最大持续step数
        self.gamma = 0.99  # 计算rtg时的衰减指数
        self.n_updates_per_iteration = 10  # 进行PPO更新时，要重复多少个epoch
        self.clip = 0.2  # PPO-clip超参数，也就是讲policy更新的幅度限制到[0.8, 1.2]之间
        self.lr = 3e-4  # 两个网络的学习率
        self.save_freq = 50  # 模型保存频率，也就是每过100次迭代保存一次模型

    """根据actor网络产生训练数据"""
    def rollout(self):
        """定义每个batch中的数据，包含每个step的状态、动作、
        log_probs、rewards以及每个episode的step长度"""
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []

        t = 0  # 记录当前共产生了多少个step
        while t < self.timesteps_per_batch:
            ep_rews = []   # 存放episode中每个step的reward
            obs = self.env.reset()   # 第一个状态
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)  # 存入状态
                action, log_prob = self.get_action(obs)  # 根据状态产生action以及该动作对应的log_prob
                obs, rew, done, _ = self.env.step(action)  # 与环境互动产生奖励以及下一个状态

                ep_rews.append(rew)  # 存入当前step的奖励
                batch_acts.append(action)  # 存入当前step的状态
                batch_log_probs.append(log_prob)  # 存入当前step对应的动作的log_prob

                if done:
                    break

            batch_lens.append(ep_t + 1)  # 记录刚刚结束的这个episode的step数目，因为for是从0开始的，所以ep_t要+1
            batch_rews.append(ep_rews)  # 记录刚刚结束的这个episode的的reward,所以batch_rew的
                                        # shape应该是（这个batch中episode的数目，每个episode的step数 ）
                                        # 把每个episode的reward分开记录的原因是需要每个step的reward-to-go
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # 根据每个episode中reward来算r-t-g

        # 日志记录
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        """actor网络输入一个状态，输出一个均值，根据这个均值和超参数协方差矩阵生成一个多维正态分布，
        然后从这个多维正态分布中采样动作以及获取该动作对应的log_probs"""
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()  # 根据多元正态分布采样动作
        log_prob = dist.log_prob(action)  # 获取采样的动作所对应的概率并转换成log_prob的形式

        return action.detach().numpy(), log_prob.detach()

    """计算reward-to-go，也就是说从当前step开始到本episode结束
    所有的reward之和(还要乘上折扣因子gamma)"""
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):   # 遍历每个episode（这里正序倒叙无所谓）
            discounted_reward = 0
            for rew in reversed(ep_rews):  # 倒序遍历episode中每个step的reward
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []



env = gym.make('Pendulum-v0')
model = PPO(env)
model.learn(5000000)