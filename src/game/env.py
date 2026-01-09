#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贪吃蛇游戏环境模块

该模块提供了两个贪吃蛇游戏环境实现：
1. SnakeEnv - 基于matplotlib的简单可视化环境，用于训练
2. PyGameSnakeEnv - 基于pygame的完整可视化环境，用于测试和展示

主要功能包括：
- 游戏状态管理
- 奖励机制实现
- 状态特征提取
- 可视化渲染
- 游戏截图保存
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from collections import deque
import random
import pygame
from PIL import Image
from src.utils.config import Config , TestConfig
#from matplotlib import pyplot as plt

class SnakeEnv:
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.reset()
        
        if self.render_mode == 'human':
            #self.fig, self.ax = plt.subplots(figsize=(6, 4))
            self.ax.set_title("贪吃蛇AI训练可视化")
            self.ax.set_xlim(-0.5, Config.GRID_WIDTH-0.5)
            self.ax.set_ylim(-0.5, Config.GRID_HEIGHT-0.5)
            self.ax.set_xticks(range(Config.GRID_WIDTH))
            self.ax.set_yticks(range(Config.GRID_HEIGHT))
            self.ax.grid(True)
            self.snake_plot, = self.ax.plot([], [], 'bo', markersize=15)
            self.head_plot, = self.ax.plot([], [], 'ro', markersize=15)
            self.food_plot, = self.ax.plot([], [], 'go', markersize=15)
            self.text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, 
                                    verticalalignment='top', 
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    def reset(self):
        """重置游戏环境
        
        初始化游戏状态：
        - 蛇的位置（屏幕中央）
        - 食物位置（随机生成且不与蛇重叠）
        - 移动方向（初始向右）
        - 分数（0）
        - 步数计数器（0）
        - 累计奖励（0）
        
        返回:
            np.array: 当前状态的特征向量
        """
        self.snake = [(Config.GRID_WIDTH//2, Config.GRID_HEIGHT//2)]
        self.food = self._generate_food()
        self.direction = (1, 0)
        self.score = 0
        self.step_count = 0
        self.episode_reward = 0 
        return self._get_state()
    
    def _generate_food(self):
        """生成新的食物位置
        
        在网格范围内随机生成食物位置，确保不会与蛇身重叠。
        使用拒绝采样方法，直到找到有效位置为止。
        
        返回:
            tuple: (x, y)坐标的食物位置
        """
        while True:
            food = (random.randint(0, Config.GRID_WIDTH-1), random.randint(0, Config.GRID_HEIGHT-1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        """获取当前游戏状态的特征表示
        
        返回12维状态特征向量，包含：
        1. 食物相对位置的归一化坐标(2维)
        2. 当前移动方向的one-hot编码(4维)
        3. 边界和碰撞检测(4维)
        4. 蛇身长度归一化值(1维)
        5. 分数归一化值(1维)
        
        返回:
            np.array: 形状为(12,)的状态向量
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        state = [
            (food_x - head_x)/Config.GRID_WIDTH,
            (food_y - head_y)/Config.GRID_HEIGHT,
            1 if self.direction == (0, -1) else 0,
            1 if self.direction == (0, 1) else 0,
            1 if self.direction == (-1, 0) else 0,
            1 if self.direction == (1, 0) else 0,
            1 if head_y == 0 or (head_x, head_y-1) in self.snake else 0,
            1 if head_y == Config.GRID_HEIGHT-1 or (head_x, head_y+1) in self.snake else 0,
            1 if head_x == 0 or (head_x-1, head_y) in self.snake else 0,
            1 if head_x == Config.GRID_WIDTH-1 or (head_x+1, head_y) in self.snake else 0,
            len(self.snake)/50,
            self.score/100
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """执行一步游戏动作
        
        参数:
            action (int): 动作索引
                0: 上移
                1: 下移
                2: 左移
                3: 右移
                
        返回:
            tuple: (next_state, reward, done)
                next_state (np.array): 下一个状态的特征向量
                reward (float): 执行动作后的即时奖励
                done (bool): 游戏是否结束
        """
        self.step_count += 1
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = actions[action]
        
        # 防止180度转向(不能直接反向移动)
        if (new_dir[0] * -1, new_dir[1] * -1) == self.direction:
            new_dir = self.direction
        self.direction = new_dir
        
        head_x, head_y = self.snake[0]
        new_head = (head_x + new_dir[0], head_y + new_dir[1])
        done = False
        reward = 0
        
        # 碰撞检测与奖励设计
        collision = (new_head in self.snake or 
                    new_head[0] < 0 or new_head[0] >= Config.GRID_WIDTH or 
                    new_head[1] < 0 or new_head[1] >= Config.GRID_HEIGHT)
        if collision:
            done = True
            reward = -15  # 碰撞惩罚
        
        self.snake.insert(0, new_head)
        
        # 食物奖励机制
        if new_head == self.food:
            self.score += 1
            reward = 20  # 食物奖励
            self.food = self._generate_food()
            self.episode_reward += reward
        else:
            self.snake.pop() 
            distance_before = np.sqrt((head_x - self.food[0])**2 + (head_y - self.food[1])**2)
            distance_after = np.sqrt((new_head[0] - self.food[0])**2 + (new_head[1] - self.food[1])**2)
            distance_reward = 0.1 if distance_after < distance_before else -0.05
            reward = 0.2 + distance_reward  
            self.episode_reward += reward
        
        # 可视化
        if self.render_mode == 'human':
            self._render_frame()
            
        return self._get_state(), reward, done
    
    def _render_frame(self):
        snake_x = [x for x, y in self.snake[1:]]
        snake_y = [y for x, y in self.snake[1:]]
        self.snake_plot.set_data(snake_x, snake_y)
        
        head_x, head_y = self.snake[0]
        self.head_plot.set_data(head_x, head_y)
        self.food_plot.set_data(self.food[0], self.food[1])
        
        self.text.set_text(f"分数: {self.score} | 长度: {len(self.snake)} | 步数: {self.step_count}")
        #plt.pause(0.1)

class PyGameSnakeEnv:
    """基于pygame的贪吃蛇游戏环境
    
    提供更丰富的可视化效果和游戏功能，包括：
    - 网格绘制
    - 蛇身绘制（头部和身体不同颜色）
    - 食物绘制
    - 游戏信息显示
    - 游戏截图功能
    """
    def __init__(self, render_mode=None, screenshot_dir=None):
        """初始化PyGame贪吃蛇环境
        
        参数:
            render_mode (str): 渲染模式('human'显示窗口，None创建Surface)
            screenshot_dir (Path): 截图保存目录(可选)
        """
        self.render_mode = render_mode
        self.screenshot_dir = screenshot_dir
        self.frame_count = 0  # 帧计数器，用于截图命名
        
        pygame.init()  # 初始化pygame
        # 计算屏幕尺寸
        self.screen_width = TestConfig.GRID_WIDTH * TestConfig.GRID_SIZE
        self.screen_height = TestConfig.GRID_HEIGHT * TestConfig.GRID_SIZE + 100
        
        self.clock = pygame.time.Clock()  # 帧率控制
        
        # 初始化显示
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("贪吃蛇AI测试")
            # 尝试加载中文字体，失败则使用默认字体
            try:
                self.font = pygame.font.Font("simhei.ttf", 20)
            except:
                print("警告: 未找到simhei.ttf字体，将使用系统默认字体")
                self.font = pygame.font.SysFont('Arial', 20)
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.font = pygame.font.SysFont('Arial', 20)
        
        self.reset()
    
    def reset(self):
        """重置游戏环境
        
        初始化游戏状态：
        - 蛇的位置（屏幕中央）
        - 食物位置（随机生成且不与蛇重叠）
        - 移动方向（初始向右）
        - 分数（0）
        - 步数计数器（0）
        - 游戏结束标志（False）
        - 帧计数器（0）
        
        返回:
            np.array: 当前状态的特征向量
        """
        self.snake = deque([(TestConfig.GRID_WIDTH//2, TestConfig.GRID_HEIGHT//2)])
        self.food = self._generate_food()
        self.direction = (1, 0)
        self.score = 0
        self.steps = 0
        self.done = False
        self.frame_count = 0  
        return self._get_state()
    
    def _generate_food(self):
        """生成新的食物位置
        
        在网格范围内随机生成食物位置，确保不会与蛇身重叠。
        使用拒绝采样方法，直到找到有效位置为止。
        
        返回:
            tuple: (x, y)坐标的食物位置
        """
        while True:
            food = (random.randint(0, TestConfig.GRID_WIDTH-1), 
                    random.randint(0, TestConfig.GRID_HEIGHT-1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # 12维状态特征
        state = [
            (food_x - head_x)/TestConfig.GRID_WIDTH,
            (food_y - head_y)/TestConfig.GRID_HEIGHT,
            1 if self.direction == (0, -1) else 0,
            1 if self.direction == (0, 1) else 0,
            1 if self.direction == (-1, 0) else 0,
            1 if self.direction == (1, 0) else 0,
            1 if head_y == 0 or (head_x, head_y-1) in self.snake else 0,
            1 if head_y == TestConfig.GRID_HEIGHT-1 or (head_x, head_y+1) in self.snake else 0,
            1 if head_x == 0 or (head_x-1, head_y) in self.snake else 0,
            1 if head_x == TestConfig.GRID_WIDTH-1 or (head_x+1, head_y) in self.snake else 0,
            len(self.snake)/50,
            self.score/100
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = actions[action]
        
        # 防止180度转向
        if (new_dir[0] * -1, new_dir[1] * -1) == self.direction:
            new_dir = self.direction
        self.direction = new_dir
        
        head_x, head_y = self.snake[0]
        new_head = (head_x + new_dir[0], head_y + new_dir[1])
        
        # 碰撞检测
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= TestConfig.GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= TestConfig.GRID_HEIGHT):
            self.done = True

            reward = -15  # 碰撞惩罚

        else:
            self.snake.appendleft(new_head)
            
            # 吃食物
            if new_head == self.food:
                self.score += 1

                reward = 20  # 食物奖励

                self.food = self._generate_food()
            else:
                self.snake.pop()  
      
                distance_before = np.sqrt((head_x - self.food[0])**2 + (head_y - self.food[1])**2)
                distance_after = np.sqrt((new_head[0] - self.food[0])**2 + (new_head[1] - self.food[1])**2)
                

                distance_reward = 0.1 if distance_after < distance_before else -0.05
                reward = 0.2 + distance_reward 
        
        self.steps += 1
        if self.steps >= TestConfig.MAX_STEPS:
            self.done = True
        
        return self._get_state(), reward, self.done
    
    def render(self):
        """渲染游戏画面
        
        主要功能包括：
        1. 绘制游戏背景网格
        2. 绘制蛇身（头部和身体不同颜色）
        3. 绘制食物
        4. 显示游戏信息（分数、步数、长度）
        5. 保存游戏截图（如果启用）
        
        截图保存逻辑：
        - 仅在TestConfig.SAVE_GAMEPLAY_SCREEN为True且screenshot_dir不为None时保存
        - 截图文件名格式为：0001.png, 0002.png等
        - 截图频率：每帧都保存（由调用者控制）
        """
        self.screen.fill(TestConfig.BG_COLOR)
        
        # 绘制网格背景
        for x in range(TestConfig.GRID_WIDTH):
            for y in range(TestConfig.GRID_HEIGHT):
                rect = pygame.Rect(x * TestConfig.GRID_SIZE, y * TestConfig.GRID_SIZE, 
                                  TestConfig.GRID_SIZE, TestConfig.GRID_SIZE)
                pygame.draw.rect(self.screen, TestConfig.GRID_COLOR, rect, 1)
        
        # 绘制蛇身（头部和身体不同颜色）
        for i, (x, y) in enumerate(self.snake):
            # 蛇头使用HEAD_COLOR，身体使用SNAKE_COLOR
            color = TestConfig.HEAD_COLOR if i == 0 else TestConfig.SNAKE_COLOR
            rect = pygame.Rect(x * TestConfig.GRID_SIZE, y * TestConfig.GRID_SIZE, 
                              TestConfig.GRID_SIZE, TestConfig.GRID_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            # 绘制黑色边框
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        # 绘制食物
        food_rect = pygame.Rect(self.food[0] * TestConfig.GRID_SIZE, 
                                self.food[1] * TestConfig.GRID_SIZE, 
                                TestConfig.GRID_SIZE, TestConfig.GRID_SIZE)
        pygame.draw.rect(self.screen, TestConfig.FOOD_COLOR, food_rect)
        
        # 显示游戏信息
        info_y = TestConfig.GRID_HEIGHT * TestConfig.GRID_SIZE + 10
        # 分数显示
        score_text = self.font.render(f"Scores: {self.score}", True, TestConfig.TEXT_COLOR)
        # 步数显示
        steps_text = self.font.render(f"Steps: {self.steps}", True, TestConfig.TEXT_COLOR)
        # 蛇长度显示
        length_text = self.font.render(f"Length: {len(self.snake)}", True, TestConfig.TEXT_COLOR)
        
        # 将文本渲染到屏幕上
        self.screen.blit(score_text, (10, info_y))
        self.screen.blit(steps_text, (150, info_y))
        self.screen.blit(length_text, (300, info_y))
        
        # 更新显示（仅在human模式下）
        if self.render_mode == 'human':
            pygame.display.flip()
        
        # 保存游戏截图（如果需要）
        if TestConfig.SAVE_GAMEPLAY_SCREEN and self.screenshot_dir is not None:
            # 创建Surface对象用于保存截图
            screenshot_surface = pygame.Surface((self.screen_width, self.screen_height))
            screenshot_surface.blit(self.screen, (0, 0))
            
            # 生成截图文件名
            self.frame_count += 1
            screenshot_filename = f"{self.frame_count:04d}.png"
            screenshot_path = self.screenshot_dir / screenshot_filename
            
            # 保存截图
            pygame.image.save(screenshot_surface, screenshot_path)
        
        # 控制帧率
        self.clock.tick(TestConfig.FPS)
    
    def close(self):
        """关闭并清理Pygame环境
        
        主要功能：
        1. 终止Pygame运行
        2. 释放资源
        
        该方法应该在游戏结束时调用，确保所有Pygame资源被正确释放。
        """
        pygame.quit()


        
