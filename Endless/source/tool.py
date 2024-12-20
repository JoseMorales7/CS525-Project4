__author__ = 'marble_xu'

import os
import json
from abc import abstractmethod
import pygame as pg
from . import constants as c
from .models.DQNModel import DQN
from .models.DQNAgent import Agent_DQN
from .models.rainbow_dqn import RainbowAgent, experienceReplayBuffer_rainbow, QNetwork_rainbow
from .models import rainbow_dqn

import torch
import torch.nn as nn
import torchvision.transforms
from collections import deque
from tqdm import tqdm
import numpy as np

class State():
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.next = None
        self.persist = {}
    
    @abstractmethod
    def startup(self, current_time, persist):
        '''abstract method'''

    def cleanup(self):
        self.done = False
        return self.persist
    
    @abstractmethod
    def update(self, surface, keys, current_time):
        '''abstract method'''

class Control():
    def __init__(self):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.fps = 60
        self.keys = pg.key.get_pressed()
        self.mouse_pos = None
        self.mouse_click = [False, False]  # value:[left mouse click, right mouse click]
        self.current_time = 0.0
        self.state_dict = {}
        self.state_name = None
        self.state = None
        self.game_info = {c.CURRENT_TIME:0.0,
                          c.LEVEL_NUM:c.START_LEVEL_NUM}
        
        # buffer = experienceReplayBuffer_rainbow(memory_size=100000, burn_in=10000)
        # net = QNetwork_rainbow(device='cpu', use_zombienet=False, use_gridnet=False)
        # self.agent = RainbowAgent(net, buffer, n_iter=-1, batch_size=200, numPlants=4)
        
        self.agent = Agent_DQN(c.GRID_Y_LEN, c.GRID_X_LEN, 4)
        
        # self.model = torch.nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel -> 16 output channels
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x (128x128 -> 64x64)

        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16 -> 32 channels
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by 2x (64x64 -> 32x32)

        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 64 channels
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2x (32x32 -> 16x16)
        # ).to("cuda:0")

        # self.model = DQN().to("cuda:0")

        # self.tsfm = torchvision.transforms.Compose([
        #     torchvision.transforms.Lambda(lambda x: x.permute(2, 1, 0)),
        #     torchvision.transforms.Resize((96, 128)),
        #     # torchvision.transforms.ToTensor()
        # ])
 
    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, self.game_info)

    def update(self, action = None):
        self.current_time = pg.time.get_ticks()
        if self.state.done:
            self.flip_state()
        self.state.update(self.screen, self.current_time, self.mouse_pos, self.mouse_click, self.keys, action)
        self.mouse_pos = None
        self.mouse_click[0] = False
        self.mouse_click[1] = False

    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
                # print(self.keys)
                # print(self.keys[pg.K_w])
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()
            elif event.type == pg.MOUSEBUTTONDOWN:
                self.mouse_pos = pg.mouse.get_pos()
                self.mouse_click[0], _, self.mouse_click[1] = pg.mouse.get_pressed()
                print('pos:', self.mouse_pos, ' mouse:', self.mouse_click)
            
            

    def main(self):
        def step():
            self.event_loop()
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)

        def stepAgent(action):
            self.event_loop()
            self.update(action)
            pg.display.update()
            self.clock.tick(self.fps)

        allScores = []
        scores = deque(maxlen=30)
        total_steps = 0
        device = "cuda:0"

        train_progress_bar = tqdm(range(1000), desc="Training")

        for episode in train_progress_bar:
            step()
            # print(self.game_info[c.LEVEL_NUM])
            state = torch.cat([torch.tensor(self.state.PlantGrid.flatten(), dtype=torch.float32), torch.tensor(self.state.ZombieGrid.flatten(), dtype=torch.float32), torch.tensor([self.state.menubar.sun_value], dtype=torch.float32), torch.tensor(self.state.menubar.getAvailableMoves()[:4], dtype=torch.float32)])
            # print(torch.tensor(self.state.PlantGrid.flatten(), dtype=torch.float32).shape)
            # print(torch.tensor(self.state.ZombieGrid.flatten(), dtype=torch.float32).shape)
            # print(torch.tensor([self.state.menubar.sun_value], dtype=torch.float32).shape)
            # print(torch.tensor(self.state.menubar.getAvailableMoves(), dtype=torch.float32).shape)

            while not self.done:
                
                # pg.image.save(self.screen,"screenshot.jpg")
                # print(pg.surfarray.array3d(self.screen))
                # print(pg.surfarray.array3d(self.screen).shape)

                # if self.state_name == c.LEVEL:
                #     screenTensor = torch.tensor(pg.surfarray.array3d(self.screen), dtype=torch.float32).to(device)
                #     screenTensor = self.tsfm(screenTensor)
                #     # print(screenTensor.shape)
                #     print(self.model(screenTensor.unsqueeze(0)).shape)
                prevScore = self.state.score
                mask = self.state.getActionMask()
                action = self.agent.make_action(state, mask, True)
                # print(action)
                stepAgent(action)
                # print(self.current_time)

                nextState = torch.cat([torch.tensor(self.state.PlantGrid.flatten(), dtype=torch.float32), torch.tensor(self.state.ZombieGrid.flatten(), dtype=torch.float32), torch.tensor([self.state.menubar.sun_value], dtype=torch.float32), torch.tensor(self.state.menubar.getAvailableMoves()[:4], dtype=torch.float32)])
                currentScore = self.state.score
                reward = currentScore - prevScore
                # print(reward)
                if total_steps % self.agent.update_replay_buffer_every == 0:
                    # print(f"Pushing: {state}")
                    # print("Pushing to buffer")
                    self.agent.push(state.cpu(), action, reward, nextState.cpu(), self.done, mask)

                state = nextState
                total_steps += 1



                if len(self.agent.replay_buffer) >= self.agent.batch_size and total_steps % self.agent.update_net_every == 0:
                    # print("Learning")
                    self.agent.learn()

                if total_steps % self.agent.update_target_every == 0:
                    # print("Updating Target")
                    self.agent.q_target.load_state_dict(self.agent.q_net.state_dict())
                    train_progress_bar.set_postfix({"Epsilon": self.agent.epsilon, "Avg Score": 0, "Score": self.state.score})


                

                if self.state.done == True:
                    # print(f"FinalScore: {self.state.score}")
                    scores.append(self.state.score)
                    train_progress_bar.set_postfix({"Epsilon": self.agent.epsilon, "Avg Score": sum(scores) / len(scores), "Score": self.state.score})
                    # score += self.state.sun_value + 
                    break

            self.agent.epsilon = max(self.agent.min_epsilon, self.agent.epsilon * self.agent.epsilon_decay)
            allScores.append(self.state.score)
            np.save("./pretrainedModels/PVZScoresEndless5000UpdFreq.npy", allScores)

            if (episode + 1) % 5 == 0:
                print(f"Episode {episode + 1}, Average Score: {(sum(scores) / len(scores)):.2f}, Score: {self.state.score:.2f}")
                self.agent.saveModel(episode + 1)

        print('game over')
    

def get_image(sheet, x, y, width, height, colorkey=c.BLACK, scale=1):
        image = pg.Surface([width, height])
        rect = image.get_rect()

        image.blit(sheet, (0, 0), (x, y, width, height))
        image.set_colorkey(colorkey)
        image = pg.transform.scale(image,
                                   (int(rect.width*scale),
                                    int(rect.height*scale)))
        return image

def load_image_frames(directory, image_name, colorkey, accept):
    frame_list = []
    tmp = {}
    # image_name is "Peashooter", pic name is 'Peashooter_1', get the index 1
    index_start = len(image_name) + 1 
    frame_num = 0
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            index = int(name[index_start:])
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            tmp[index]= img
            frame_num += 1

    for i in range(frame_num):
        frame_list.append(tmp[i])
    return frame_list

def load_all_gfx(directory, colorkey=c.WHITE, accept=('.png', '.jpg', '.bmp', '.gif')):
    graphics = {}
    for name1 in os.listdir(directory):
        # subfolders under the folder resources\graphics
        dir1 = os.path.join(directory, name1)
        if os.path.isdir(dir1):
            for name2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, name2)
                if os.path.isdir(dir2):
                # e.g. subfolders under the folder resources\graphics\Zombies
                    for name3 in os.listdir(dir2):
                        dir3 = os.path.join(dir2, name3)
                        # e.g. subfolders or pics under the folder resources\graphics\Zombies\ConeheadZombie
                        if os.path.isdir(dir3):
                            # e.g. it's the folder resources\graphics\Zombies\ConeheadZombie\ConeheadZombieAttack
                            image_name, _ = os.path.splitext(name3)
                            graphics[image_name] = load_image_frames(dir3, image_name, colorkey, accept)
                        else:
                            # e.g. pics under the folder resources\graphics\Plants\Peashooter
                            image_name, _ = os.path.splitext(name2)
                            graphics[image_name] = load_image_frames(dir2, image_name, colorkey, accept)
                            break
                else:
                # e.g. pics under the folder resources\graphics\Screen
                    name, ext = os.path.splitext(name2)
                    if ext.lower() in accept:
                        img = pg.image.load(dir2)
                        if img.get_alpha():
                            img = img.convert_alpha()
                        else:
                            img = img.convert()
                            img.set_colorkey(colorkey)
                        graphics[name] = img
    return graphics

def loadZombieImageRect():
    file_path = os.path.join('source', 'data', 'entity', 'zombie.json')
    f = open(file_path)
    data = json.load(f)
    f.close()
    return data[c.ZOMBIE_IMAGE_RECT]

def loadPlantImageRect():
    file_path = os.path.join('source', 'data', 'entity', 'plant.json')
    f = open(file_path)
    data = json.load(f)
    f.close()
    return data[c.PLANT_IMAGE_RECT]

pg.init()
pg.display.set_caption(c.ORIGINAL_CAPTION)
SCREEN = pg.display.set_mode(c.SCREEN_SIZE)

GFX = load_all_gfx(os.path.join("resources","graphics"))
ZOMBIE_RECT = loadZombieImageRect()
PLANT_RECT = loadPlantImageRect()
