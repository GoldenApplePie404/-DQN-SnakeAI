
"""
重构计划：

原因：因项目整体架构调整，需要对此代码进行重构，以适应新的架构。

主要工作：保留并优化原先的功能框架，并优化代码结构，使其符合新的架构。

"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import pygame
import matplotlib.pyplot as plt

from collections import deque
from datetime import datetime
from pathlib import Path
from PIL import Image

from src.game.env import PyGameSnakeEnv
from src.utils.config import TestConfig


class ModelTester:
    def __init__(self):
        # 确保项目根目录路径正确
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # 更新配置中的路径为绝对路径
        TestConfig.MODEL_DIR = project_root / "saved_models"
        TestConfig.RESULT_DIR = project_root / "test_results"
        TestConfig.RESULT_IMG_DIR = TestConfig.RESULT_DIR / "images"
        TestConfig.RESULT_DATA_DIR = TestConfig.RESULT_DIR / "data"
        TestConfig.SCREENSHOT_DIR = project_root / "game_img"
        
        # 确保所有目录存在
        TestConfig.RESULT_DIR.mkdir(exist_ok=True, parents=True)
        TestConfig.RESULT_IMG_DIR.mkdir(exist_ok=True)
        TestConfig.RESULT_DATA_DIR.mkdir(exist_ok=True)
        
        # 创建游戏画面截图主目录
        if TestConfig.SAVE_GAMEPLAY_SCREEN:
            TestConfig.SCREENSHOT_DIR.mkdir(exist_ok=True, parents=True)
            print(f"游戏画面将保存至: {TestConfig.SCREENSHOT_DIR.absolute()}")
        
        # 加载模型
        self.selected_model_path = self._load_model()  # 保存选中模型路径
        self.model = tf.keras.models.load_model(self.selected_model_path)
        self.env = PyGameSnakeEnv()
        
        # 测试结果
        self.test_results = {
            'scores': [],
            'steps': [],
            'lengths': [],
            'episode_times': [],
            'avg_rewards': [],
            'exploration_rates': [],
            'performance_scores': [],
            'convergence_metrics': [],  # 收敛指标
            'exploration_efficiency': [],  # 探索效率
            'decision_quality': [],  # 决策质量
            'stability_metrics': [],  # 稳定性指标
            'q_value_differences': [],  # Q值差异
            'start_time': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        # 初始化近期Q值记录
        self.recent_q_values = deque(maxlen=1000)
        self.target_model = None
        self._load_target_model()  # 加载目标网络
        self._print_init_info()
    
    def _load_model(self):
        """加载模型，支持用户选择"""
        try:
            model_files = []
            for pattern in TestConfig.MODEL_PATTERNS:
                model_files.extend(TestConfig.MODEL_DIR.glob(pattern))
            
            if not model_files:
                raise FileNotFoundError(f"未在目录 {TestConfig.MODEL_DIR} 中找到任何模型文件")
            
            # 按修改时间排序
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # 显示模型选择菜单
            print("\n" + "="*50)
            print("  可用模型列表 (按修改时间排序)")
            print("\n  [模型类型说明]")
            print("  • 最终模型(首选): final_snake_model.keras")
            print("  • 常规模型(次选): snake_agent_[轮次].keras")
            print("  • 中断保存: interrupted_model_[轮次].keras")
            print("  • 错误保存: error_snake_model_[轮次].keras")
            print("  • TFLite模型(快速验证): snake_model.tflite")
            print("="*50)
            for i, model in enumerate(model_files):
                mod_time = datetime.fromtimestamp(os.path.getmtime(model))
                size_mb = os.path.getsize(model) / (1024 * 1024)
                print(f"  [{i+1}] {model.name} (修改时间: {mod_time}, 大小: {size_mb:.2f}MB)")
            print("  [0] 退出程序")
            print("="*50)
            
            # 获取用户选择
            while True:
                try:
                    choice = input("请选择要加载的模型编号: ")
                    if choice == "0":
                        print("退出程序...")
                        sys.exit(0)
                    
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(model_files):
                        selected_model = model_files[choice_idx]
                        break
                    else:
                        print(f"错误: 请输入0-{len(model_files)}之间的数字")
                except ValueError:
                    print("错误: 请输入有效的数字")
            
            print(f"\n加载模型: {selected_model.absolute()}")
            
            # 验证模型文件
            if not selected_model.exists():
                raise FileNotFoundError(f"模型文件不存在: {selected_model.absolute()}")
            if os.path.getsize(selected_model) == 0:
                raise ValueError(f"模型文件为空: {selected_model.absolute()}")
            
            print(f"模型加载成功: {selected_model.name}")
            return selected_model  # 返回路径
            
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def predict_action(self, state):
        """模型预测动作（与训练一致的处理方式）"""
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)
    
    def _load_target_model(self):
        """加载目标网络用于更准确的测试"""
        try:
            # 尝试加载与主模型对应的目标网络
            model_name = Path(self.selected_model_path).stem
            target_model_name = f"target_{model_name}"
            target_model_path = TestConfig.MODEL_DIR / f"{target_model_name}{TestConfig.MODEL_EXTENSION}"
            
            if target_model_path.exists():
                self.target_model = tf.keras.models.load_model(target_model_path)
                print(f"成功加载目标网络: {target_model_path.name}")
                return True
            else:
                print(f"未找到目标网络 {target_model_path.name}，使用主网络进行测试")
                self.target_model = None
                return False
        except Exception as e:
            print(f"加载目标网络失败: {str(e)}，使用主网络进行测试")
            self.target_model = None
            return False
    
    def predict_with_target(self, state):
        """使用目标网络预测动作（如果可用）"""
        if self.target_model is not None:
            q_values = self.target_model.predict(state[np.newaxis, :], verbose=0)[0]
            return np.argmax(q_values)
        else:
            return self.predict_action(state)
    
    def run_test_episode(self, episode_idx, render=True):
        """运行单轮测试"""
        state = self.env.reset()
        total_reward = 0
        start_time = time.time()
        states = []
        
        # 预分配动作数组
        actions = np.zeros(TestConfig.MAX_STEPS, dtype=np.int32)
        
        while not self.env.done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.env.close()
                        return
            
            # 批量收集状态
            states.append(state)
            
            # 每10步批量预测一次
            if len(states) >= 10 or self.env.done:
                states_batch = np.array(states)
                # 使用目标网络进行更准确的Q值预测（如果可用）
                if self.target_model is not None:
                    q_values = self.target_model.predict(states_batch, verbose=0)
                else:
                    q_values = self.model.predict(states_batch, verbose=0)
                for i, q in enumerate(q_values):
                    actions[i] = np.argmax(q)
                    # 记录Q值
                    self.recent_q_values.append(q)
                states = []
            
            if len(states) == 0:
                action = actions[len(self.test_results['steps'])]
            else:
                # 单步预测
                # 使用目标网络进行更准确的Q值预测（如果可用）
                if self.target_model is not None:
                    q_values = self.target_model.predict(state[np.newaxis, :], verbose=0)[0]
                else:
                    q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
                action = np.argmax(q_values)
                # 记录Q值
                self.recent_q_values.append(q_values)
            state, reward, done = self.env.step(action)
            total_reward += reward
            
            # 所有轮次都调用render()，但根据render_mode决定是否显示画面
            self.env.render()
        
        episode_time = time.time() - start_time
        self.test_results['scores'].append(self.env.score)
        self.test_results['steps'].append(self.env.steps)
        self.test_results['lengths'].append(len(self.env.snake))
        self.test_results['episode_times'].append(episode_time)
        
        # 计算综合性能指标
        avg_reward = total_reward / self.env.steps if self.env.steps > 0 else 0
        performance_score, convergence, exploration_efficiency, decision_quality, stability, q_diff = self._calculate_performance_metrics(episode_idx)
        
        # 更新测试结果
        self.test_results['avg_rewards'].append(avg_reward)
        self.test_results['exploration_rates'].append(TestConfig.EXPLORATION_RATE)
        self.test_results['performance_scores'].append(performance_score)
        self.test_results['convergence_metrics'].append(convergence)
        self.test_results['exploration_efficiency'].append(exploration_efficiency)
        self.test_results['decision_quality'].append(decision_quality)
        self.test_results['stability_metrics'].append(stability)
        self.test_results['q_value_differences'].append(q_diff)
        
        print(f"测试轮次 {episode_idx + 1}/{TestConfig.TEST_EPISODES}")
        print(f"  • 分数: {self.env.score} (阈值: {TestConfig.MIN_SCORE_THRESHOLD})")
        print(f"  • 长度: {len(self.env.snake)} (阈值: {TestConfig.MIN_LENGTH_THRESHOLD})")
        print(f"  • 步数: {self.env.steps}")
        print(f"  • 耗时: {episode_time:.2f}s")
        print(f"  • 平均奖励: {avg_reward:.4f}")
        print(f"  • 性能评分: {performance_score:.2f}")
        print(f"  • 收敛性: {convergence:.2f} (阈值: {TestConfig.CONVERGENCE_THRESHOLD})")
        print(f"  • 探索效率: {exploration_efficiency:.2f} (阈值: {TestConfig.EXPLORATION_EFFICIENCY_THRESHOLD})")
        print(f"  • 决策质量: {decision_quality:.2f} (阈值: {TestConfig.DECISION_QUALITY_THRESHOLD})")
        print(f"  • 稳定性: {stability:.2f}")
        print(f"  • Q值差异: {q_diff:.2f} (阈值: {TestConfig.MIN_Q_VALUE_DIFFERENCE})")
        print("-"*60)
    
    def run_full_test(self):
        """运行完整测试"""
        print("\n===== 开始模型测试 =====")
        start_time = time.time()
        
        try:
            # 为当前测试会话创建唯一GIF基础目录: test_results/game_gif/<模型名>_<时间>
            # 此目录将包含所有test_*子目录
            if TestConfig.SAVE_GAMEPLAY_GIF and TestConfig.SAVE_GAMEPLAY_SCREEN:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 修复时间戳格式
                model_name = Path(self.selected_model_path).stem
                self.gif_base_dir = TestConfig.RESULT_DIR / "game_gif" / f"{model_name}_{timestamp}"
                self.gif_base_dir.mkdir(exist_ok=True, parents=True)
                print(f"GIF将保存至基础目录: {self.gif_base_dir.absolute()}")
                
            for i in range(TestConfig.TEST_EPISODES):
                try:
                    render = i % 2 == 0  # 每隔一轮渲染一次，提高测试速度
                    
                    # 为当前轮次创建截图目录 - 每轮都创建目录
                    episode_dir = None
                    if TestConfig.SAVE_GAMEPLAY_SCREEN:
                        episode_dir_name = f"test_{i+1}"
                        episode_dir = TestConfig.SCREENSHOT_DIR / episode_dir_name
                        os.makedirs(episode_dir, exist_ok=True)
                        if render:
                            print(f"\n正在为第 {i+1} 轮创建截图目录并渲染: {episode_dir}")
                        else:
                            print(f"\n正在为第 {i+1} 轮创建截图目录: {episode_dir}")
                    
                    # 每轮都重新初始化环境，确保使用正确的截图目录
                    self.env.close()
                    self.env = PyGameSnakeEnv(
                        render_mode='human' if render else None,
                        screenshot_dir=episode_dir if TestConfig.SAVE_GAMEPLAY_SCREEN else None
                    )
                    
                    self.run_test_episode(i, render=render)
                    
                    # 生成GIF
                    if TestConfig.SAVE_GAMEPLAY_GIF and TestConfig.SAVE_GAMEPLAY_SCREEN:
                        # 使用预先创建的基础目录，为当前轮次创建子目录
                        if hasattr(self, 'gif_base_dir') and self.gif_base_dir is not None:
                            # 为当前轮次创建子目录
                            gif_episode_dir = self.gif_base_dir / f"test_{i+1}"
                            gif_episode_dir.mkdir(exist_ok=True)
                            
                            # 生成GIF
                            self._generate_gif(i, episode_dir, gif_episode_dir)
                except Exception as e:
                    print(f"测试轮次 {i+1} 发生错误: {str(e)}")
                    continue
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        finally:
            total_time = time.time() - start_time
            print(f"\n测试完成，总耗时: {total_time:.2f}s")
            if len(self.test_results['scores']) > 0:
                self._save_test_results()
                self._generate_summary_plots()
            else:
                print("警告: 无有效测试结果可保存")
            self.env.close()
            
            # 清理game_img目录内容
            if TestConfig.SAVE_GAMEPLAY_SCREEN:
                try:
                    import shutil
                    # 删除game_img目录下的所有内容
                    for item in TestConfig.SCREENSHOT_DIR.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    print(f"已清理game_img目录内容: {TestConfig.SCREENSHOT_DIR.absolute()}")
                except Exception as e:
                    print(f"清理game_img目录时出错: {str(e)}")
    
    def _calculate_performance_metrics(self, episode_idx):
        """计算综合性能指标"""
        window_size = min(TestConfig.PERFORMANCE_WINDOW, episode_idx + 1)
        if window_size == 0:
            return 0, 0, 0, 0, 0, 0
            
        start_idx = max(0, episode_idx - window_size + 1)
        recent_scores = self.test_results['scores'][start_idx:episode_idx + 1]
        recent_lengths = self.test_results['lengths'][start_idx:episode_idx + 1]
        recent_rewards = self.test_results['avg_rewards'][start_idx:episode_idx + 1]
        
        # 基础指标
        avg_score = np.mean(recent_scores) if recent_scores else 0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        # 收敛性指标 (分数标准差)
        convergence = 1 - (np.std(recent_scores) / max(1, avg_score)) if recent_scores else 0
        
        # 探索效率 (有效探索步数比例)
        recent_steps = self.test_results['steps'][start_idx:episode_idx + 1]
        exploration_efficiency = min(1, avg_score / max(1, np.mean(recent_steps))) if recent_steps else 0
        
        # 决策质量 (最优动作选择比例)
        decision_quality = self._calculate_decision_quality(episode_idx)
        
        # 稳定性指标 (连续窗口分数变化率)
        stability = self._calculate_stability_metric(episode_idx)
        
        # Q值差异 (最优动作与次优动作的Q值差异)
        q_diff = self._calculate_q_value_difference(episode_idx)
        
        # 综合性能评分
        performance_score = (
            avg_score * 0.3 + 
            avg_length * 0.2 + 
            avg_reward * 0.2 +
            convergence * 0.1 +
            exploration_efficiency * 0.1 +
            decision_quality * 0.1
        )
        
        return performance_score, convergence, exploration_efficiency, decision_quality, stability, q_diff
    
    def _calculate_decision_quality(self, episode_idx):
        """计算决策质量(最优动作选择比例) - 与训练一致的评估方法"""
        # 检查是否有历史Q值记录
        if not hasattr(self, 'recent_q_values') or len(self.recent_q_values) == 0:
            return 0.0
            
        # 计算最优动作选择比例
        correct_decisions = 0
        total_decisions = 0
        
        for q_values in self.recent_q_values:
            # 找到最优动作索引
            optimal_action = np.argmax(q_values)
            # 计算次优动作Q值（排除最优动作）
            q_values_without_optimal = q_values.copy()
            q_values_without_optimal[optimal_action] = -np.inf
            second_best_action = np.argmax(q_values_without_optimal)
            
            # 如果最优动作的Q值明显大于次优动作，则认为是正确决策
            if q_values[optimal_action] - q_values[second_best_action] > TestConfig.MIN_Q_VALUE_DIFFERENCE:
                correct_decisions += 1
            
            total_decisions += 1
        
        return correct_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def _calculate_stability_metric(self, episode_idx):
        """计算稳定性指标"""
        if episode_idx < TestConfig.STABILITY_WINDOW:
            return 0
            
        recent_scores = self.test_results['scores'][episode_idx - TestConfig.STABILITY_WINDOW:episode_idx + 1]
        if len(recent_scores) < 2:
            return 0
            
        score_changes = np.abs(np.diff(recent_scores))
        return 1 - (np.mean(score_changes) / max(1, np.mean(recent_scores)))
    
    def _calculate_q_value_difference(self, episode_idx):
        """计算Q值差异(最优动作与次优动作的Q值差异)"""
        # 检查是否有历史Q值记录
        if not hasattr(self, 'recent_q_values') or len(self.recent_q_values) == 0:
            return 0.0
            
        # 计算Q值差异的平均值
        total_diff = 0
        count = 0
        
        for q_values in self.recent_q_values:
            # 找到最优动作和次优动作
            optimal_action = np.argmax(q_values)
            q_values_without_optimal = q_values.copy()
            q_values_without_optimal[optimal_action] = -np.inf
            second_best_action = np.argmax(q_values_without_optimal)
            
            # 计算差异
            diff = q_values[optimal_action] - q_values[second_best_action]
            total_diff += diff
            count += 1
        
        return total_diff / count if count > 0 else 0.0

    def _save_test_results(self):
        """保存测试结果数据"""
        # 使用用户选择的模型名称
        if hasattr(self, 'selected_model_path') and self.selected_model_path is not None:
            model_name = Path(self.selected_model_path).stem
        else:
            model_name = "unknown_model"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用模型名称+测试数据+时间戳的格式命名
        data_filename = f"{model_name}_测试数据_{timestamp}.csv"
        data_path = TestConfig.RESULT_DATA_DIR / data_filename
        
        with open(data_path, 'w') as f:
            f.write("episode,score,steps,length,time,avg_reward,performance_score," 
                   "convergence,exploration_efficiency,decision_quality,stability,q_value_diff\n")
            for i in range(len(self.test_results['scores'])):
                f.write(f"{i + 1},{self.test_results['scores'][i]}," 
                        f"{self.test_results['steps'][i]},{self.test_results['lengths'][i]}," 
                        f"{self.test_results['episode_times'][i]}," 
                        f"{self.test_results['avg_rewards'][i]}," 
                        f"{self.test_results['performance_scores'][i]}," 
                        f"{self.test_results['convergence_metrics'][i]}," 
                        f"{self.test_results['exploration_efficiency'][i]}," 
                        f"{self.test_results['decision_quality'][i]}," 
                        f"{self.test_results['stability_metrics'][i]}," 
                        f"{self.test_results['q_value_differences'][i]}\n")
        
        print(f"测试结果已保存至: {data_path}")
    
    def _generate_gif(self, episode_idx, screenshot_dir, gif_episode_dir):
        """生成游戏画面GIF"""
        if not TestConfig.SAVE_GAMEPLAY_GIF:
            return
            
        try:
            # 使用传入的目录，不再创建新目录
            gif_episode_dir.mkdir(exist_ok=True, parents=True)
            
            # 获取所有截图文件
            screenshot_files = sorted(screenshot_dir.glob("*.png"))
            
            if not screenshot_files:
                print(f"警告: 第 {episode_idx + 1} 轮没有找到截图文件")
                return
                
            # 每隔N帧取一帧，减少GIF大小
            subsampled_files = screenshot_files[::TestConfig.GIF_SUBSAMPLE]
            
            # 使用Pillow生成GIF
            images = []
            for img_path in subsampled_files:
                try:
                    img = Image.open(img_path)
                    images.append(img)
                except Exception as e:
                    print(f"警告: 无法打开截图文件 {img_path}: {e}")
            
            if not images:
                print(f"警告: 第 {episode_idx + 1} 轮没有有效截图")
                return
                
            # 保存GIF
            gif_filename = f"{episode_idx + 1}.gif"
            gif_path = gif_episode_dir / gif_filename
            
            # 计算每帧之间的延迟
            duration = int(1000 / TestConfig.GIF_FPS)
            
            # 保存GIF
            gif_filename = f"{episode_idx + 1}.gif"
            gif_path = gif_episode_dir / gif_filename
            
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=TestConfig.GIF_LOOP,
                optimize=True,
                quality=TestConfig.GIF_QUALITY
            )
            
            print(f"已生成GIF: {gif_path}")
            
        except Exception as e:
            print(f"GIF生成失败: {str(e)}")
    
    def _print_init_info(self):
        """打印初始化信息"""
        print("\n" + "="*50)
        print("  贪吃蛇AI模型测试工具初始化")
        print("="*50)
        print(f"  • 测试轮次: {TestConfig.TEST_EPISODES}")
        print(f"  • 最大步数/轮: {TestConfig.MAX_STEPS}")
        print(f"  • 网格尺寸: {TestConfig.GRID_WIDTH}x{TestConfig.GRID_HEIGHT}")
        print(f"  • 最低分数阈值: {TestConfig.MIN_SCORE_THRESHOLD}")
        print(f"  • 最低长度阈值: {TestConfig.MIN_LENGTH_THRESHOLD}")
        print(f"  • 测试探索率: {TestConfig.EXPLORATION_RATE}")
        print(f"  • 性能评估窗口: {TestConfig.PERFORMANCE_WINDOW}")
        print(f"  • 模型目录: {TestConfig.MODEL_DIR.absolute()}")
        print(f"  • 结果目录: {TestConfig.RESULT_DIR.absolute()}")
        print("="*50 + "\n")

    def _generate_summary_plots(self):
        """生成测试结果图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(18, 12))
        
        # 使用用户选择的模型名称
        if hasattr(self, 'selected_model_path') and self.selected_model_path is not None:
            model_name = Path(self.selected_model_path).stem
        else:
            model_name = "unknown_model"
        
        # 模型名称和标题组合
        base_title = f"{model_name} 测试结果数据"
        
        # 分数趋势图
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax1.plot(self.test_results['scores'], 'b-')
        ax1.set_title(f'{base_title} - 分数趋势 (阈值: {TestConfig.MIN_SCORE_THRESHOLD})')
        ax1.set_xlabel('测试轮次')
        ax1.set_ylabel('分数')
        ax1.axhline(y=TestConfig.MIN_SCORE_THRESHOLD, color='r', linestyle='--')
        ax1.grid(True)
        
        # 蛇长度趋势图
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        ax2.plot(self.test_results['lengths'], 'g-')
        ax2.set_title(f'{base_title} - 蛇长度趋势 (阈值: {TestConfig.MIN_LENGTH_THRESHOLD})')
        ax2.set_xlabel('测试轮次')
        ax2.set_ylabel('长度')
        ax2.axhline(y=TestConfig.MIN_LENGTH_THRESHOLD, color='r', linestyle='--')
        ax2.grid(True)
        
        # 综合指标趋势
        ax3 = plt.subplot2grid((3, 3), (1, 0))
        ax3.plot(self.test_results['performance_scores'], 'm-', label='性能评分')
        ax3.plot(self.test_results['convergence_metrics'], 'y-', label='收敛性')
        ax3.plot(self.test_results['exploration_efficiency'], 'c-', label='探索效率')
        ax3.plot(self.test_results['decision_quality'], 'g-', label='决策质量')
        ax3.set_title(f'{base_title} - 综合指标趋势 (窗口大小: {TestConfig.PERFORMANCE_WINDOW})')
        ax3.set_xlabel('测试轮次')
        ax3.set_ylabel('指标值')
        ax3.legend()
        ax3.grid(True)
        
        # 添加参考线
        ax3.axhline(y=TestConfig.CONVERGENCE_THRESHOLD, color='y', linestyle='--', alpha=0.5)
        ax3.axhline(y=TestConfig.EXPLORATION_EFFICIENCY_THRESHOLD, color='c', linestyle='--', alpha=0.5)
        ax3.axhline(y=TestConfig.DECISION_QUALITY_THRESHOLD, color='g', linestyle='--', alpha=0.5)
        
        # 平均奖励趋势
        ax4 = plt.subplot2grid((3, 3), (1, 1))
        ax4.plot(self.test_results['avg_rewards'], 'c-')
        ax4.set_title(f'{base_title} - 平均奖励趋势')
        ax4.set_xlabel('测试轮次')
        ax4.set_ylabel('平均奖励')
        ax4.grid(True)
        
        # 步数趋势图
        ax5 = plt.subplot2grid((3, 3), (1, 2))
        ax5.plot(self.test_results['steps'], 'r-')
        ax5.set_title(f'{base_title} - 步数趋势')
        ax5.set_xlabel('测试轮次')
        ax5.set_ylabel('步数')
        ax5.grid(True)
        
        # 耗时分布图
        ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        ax6.hist(self.test_results['episode_times'], bins=10, color='purple')
        ax6.set_title(f'{base_title} - 单轮耗时分布')
        ax6.set_xlabel('时间(s)')
        ax6.set_ylabel('频次')
        ax6.grid(True)
        
        plt.tight_layout()
        
        # 确保 images 目录存在
        TestConfig.RESULT_IMG_DIR.mkdir(exist_ok=True)
        
        # 保存图表到 images 目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用模型名称+测试数据+时间戳的格式命名
        plot_filename = f"{model_name}_测试数据_{timestamp}.png"
        plot_path = TestConfig.RESULT_IMG_DIR / plot_filename
        plt.savefig(plot_path)
        plt.close()
        
        print(f"测试总结图表已保存至: {plot_path}")

# 添加主运行逻辑
if __name__ == "__main__":
    try:
        tester = ModelTester()
        tester.run_full_test()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()