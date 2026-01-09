"""
Q网络实现模块
该模块实现了DQN（深度Q网络）算法中的Q网络，包括主网络和目标网络。
主要功能包括：
- 构建神经网络模型
- 预测Q值（单样本/批量）
- 训练网络
- 同步主网络和目标网络权重
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

class QNetwork:
    """Q网络类
    实现了DQN算法中的Q网络，包括主网络和目标网络。
    """
    def __init__(self, state_size, action_size, learning_rate):
        """初始化Q网络
        参数:
            state_size (int): 状态特征的维度
            action_size (int): 动作空间的大小
            learning_rate (float): 学习率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_fn = Huber()
        
    def _build_model(self):
        """构建神经网络模型
        返回:
            Sequential: Keras序列模型
        """
        return Sequential([
            Dense(128, activation='relu', input_shape=(self.state_size,)),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(self.action_size, activation='linear')
        ])
        
    def update_target_network(self):
        """同步主网络和目标网络权重"""
        self.target_model.set_weights(self.model.get_weights())
        
    def predict_single(self, state):
        """预测单个状态的Q值（自动添加批次维度并禁用进度条）
        参数:
            state (np.array): 单个状态特征向量，形状为(state_size,)
        返回:
            np.array: 每个动作的Q值，形状为(action_size,)
        """
        state_with_batch = np.expand_dims(state, axis=0)
        return self.model.predict(state_with_batch, verbose=0)[0]
        
    def predict_batch(self, states):
        """预测批量状态的Q值（禁用进度条）
        参数:
            states (np.array): 状态批次，形状为(batch_size, state_size)
        返回:
            np.array: 每个状态-动作对的Q值，形状为(batch_size, action_size)
        """
        return self.model.predict(states, verbose=0)
        
    def predict(self, state, verbose=0):
        """兼容接口：根据输入形状自动选择单样本/批量预测
        参数:
            state (np.array): 状态特征向量或批次
        返回:
            np.array: Q值数组
        """
        if len(state.shape) == 1:  # 单样本
            return self.predict_single(state)
        else:  # 批量样本
            return self.predict_batch(state)
            
    def train(self, states, targets):
        """训练Q网络
        参数:
            states (np.array): 状态批次，形状为(batch_size, state_size)
            targets (np.array): 目标Q值批次，形状为(batch_size, action_size)
        返回:
            float: 训练损失值
        """
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = self.loss_fn(targets, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()
    def target_predict_single(self, state):
        """目标网络预测单个状态的Q值"""
        state_with_batch = np.expand_dims(state, axis=0)
        return self.target_model.predict(state_with_batch, verbose=0)[0]
        
    def target_predict_batch(self, states):
        """目标网络预测批量状态的Q值"""
        return self.target_model.predict(states, verbose=0)
    
