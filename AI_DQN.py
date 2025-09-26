import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from collections import deque


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class TetrisAI:
    def __init__(self):
        self.opt = self.get_args()
        self.episode = None
        self.epoch = None
        self.model = None
        self.replay_memory = []
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Use device: {self.DEVICE}')

    def get_args(self):
        parser = argparse.ArgumentParser(
            """Implementation of Deep Q Network to play Tetris""")
        parser.add_argument("--width", type=int, default=10, help="The common width for all images")
        parser.add_argument("--height", type=int, default=21, help="The common height for all images")
        parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
        parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--initial_epsilon", type=float, default=1)
        parser.add_argument("--final_epsilon", type=float, default=1e-3)
        parser.add_argument("--num_decay_epochs", type=float, default=2000)
        parser.add_argument("--num_epochs", type=int, default=3000)
        parser.add_argument("--save_interval", type=int, default=10)
        parser.add_argument("--replay_memory_size", type=int, default=30000,
                            help="Number of epoches between testing phases")
        parser.add_argument("--log_path", type=str, default="tensorboard")
        parser.add_argument("--saved_path", type=str, default="Data")

        args = parser.parse_args()
        return args

    def init_setting(self, state_properties):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)
        if os.path.isdir(self.opt.log_path):
            shutil.rmtree(self.opt.log_path)
        os.makedirs(self.opt.log_path)
        self.writer = SummaryWriter(self.opt.log_path)
        # env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
        # self.model = DeepQNetwork()
        self.load_model(f'{self.opt.saved_path}/tetris.pth')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.criterion = nn.MSELoss()

        self.reset_state(state_properties)
        if torch.cuda.is_available():
            self.model.cuda()
            self.state = self.state.cuda()

        # self.replay_memory = deque(maxlen=self.opt.replay_memory_size)
        # self.epoch = 0

    def reset_state(self, state_properties):
        self.state = state_properties

    def select_action(self, next_steps) -> tuple:
        # Exploration or exploitation
        epsilon = self.opt.final_epsilon + (max(self.opt.num_decay_epochs - self.epoch, 0) * (
                self.opt.initial_epsilon - self.opt.final_epsilon) / self.opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(next_states)[:, 0]
        self.model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        self.next_state = next_states[index, :]
        self.action = next_actions[index]
        return self.action


    def update(self, reward, done, total_reward, tetrominoes, cleared_lines, game_score):
        # reward, done = env.step(self.action, render=True)

        if torch.cuda.is_available():
            self.next_state = self.next_state.cuda()
        self.replay_memory.append([self.state, reward, self.next_state, done])
        if done:
            final_reward = total_reward
            final_tetrominoes = tetrominoes
            final_cleared_lines = cleared_lines
            # state = env.reset()
            if torch.cuda.is_available():
                self.state = self.state.cuda()
            self.episode += 1
        else:
            self.state = self.next_state
            return
        if len(self.replay_memory) < self.opt.replay_memory_size / 10:
            return
        self.epoch += 1
        batch = sample(self.replay_memory, min(len(self.replay_memory), self.opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state.to(self.DEVICE) for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = self.model(state_batch)
        self.model.eval()
        with torch.no_grad():
            next_prediction_batch = self.model(next_state_batch)
        self.model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + self.opt.gamma * prediction for reward, done, prediction in
                zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, y_batch)
        loss.backward()
        self.optimizer.step()

        print("Epoch: {} | Reward: {:4} | Tetrominoes {:3} | Cleared lines: {:3} | Score: {:5} | Loss: {:4.4f}".format(
            self.epoch,
            # self.opt.num_epochs,
            # self.action,
            final_reward,
            final_tetrominoes,
            final_cleared_lines,
            game_score,
            loss.mean().item()))
        self.writer.add_scalar('Train/Score', final_reward, self.epoch - 1)
        self.writer.add_scalar('Train/Tetrominoes', final_tetrominoes, self.epoch - 1)
        self.writer.add_scalar('Train/Cleared lines', final_cleared_lines, self.epoch - 1)

        if self.epoch % self.opt.save_interval == 0 or game_score >= 200000:
            self.save_model(f'{self.opt.saved_path}/tetris.pth')

        # if self.episode % self.opt.save_interval == 0:
        #     self.save_model(f'{self.opt.saved_path}/tetris_.pth')
            # torch.save(self.model, "{}/tetris_{}.pth".format(self.opt.saved_path, self.epoch))

    # torch.save(self.model, "{}/tetris".format(opt.saved_path))

    def save_model(self, model_path: str):
        torch.save({
            'model': self.model.state_dict(),
            'episode': self.episode,
            'epoch': self.epoch,
            'replay_memory': self.replay_memory
            }, model_path)
        print(f"模型已儲存到 {model_path}")

    def load_model(self, model_path: str):
        if not os.path.isdir('./Data'):
            os.mkdir('./Data')
        self.model = DeepQNetwork().to(self.DEVICE)  
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.episode = checkpoint['episode']
            self.epoch = checkpoint['epoch']
            self.replay_memory = checkpoint['replay_memory']
            print(f"模型已從 {model_path} 載入")
        else:
            print("找不到模型檔案，無法載入。")
            self.episode = 1
            self.epoch = 0
            self.replay_memory = deque(maxlen=self.opt.replay_memory_size)


# if __name__ == "__main__":
#     opt = TetrisAI().get_args()
#     TetrisAI().train(opt)
