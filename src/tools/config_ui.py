import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config, TestConfig, config_loader

# 中英文映射字典
CONFIG_TRANSLATIONS = {
    "game": {
        "GRID_WIDTH": "游戏网格宽度",
        "GRID_HEIGHT": "游戏网格高度",
        "STATE_SIZE": "状态向量维度",
        "ACTION_SIZE": "动作空间大小"
    },
    "training": {
        "EPISODES": "总训练轮次",
        "BATCH_SIZE": "经验回放采样批量大小",
        "GAMMA": "折扣因子",
        "LEARNING_RATE": "神经网络学习率",
        "EPSILON_INIT": "ε-贪婪策略初始探索率",
        "EPSILON_MIN": "最小探索率",
        "EPSILON_DECAY": "探索率衰减系数",
        "REPLAY_BUFFER_SIZE": "经验回放缓冲区容量",
        "TARGET_UPDATE_FREQ": "目标网络更新频率"
    },
    "model": {
        "SAVE_INTERVAL": "模型保存间隔",
        "MODEL_DIR": "模型保存目录",
        "LOG_DIR": "训练日志目录",
        "CHECKPOINT_PREFIX": "模型文件名前缀",
        "MODEL_EXTENSION": "模型文件扩展名",
        "TENSORBOARD_LOG_DIR": "TensorBoard日志目录"
    },
    "test": {
        "GRID_SIZE": "游戏网格像素大小",
        "TEST_EPISODES": "测试轮次",
        "MAX_STEPS": "每轮最大步数",
        "FPS": "游戏帧率",
        "EXPLORATION_RATE": "测试时的探索率",
        "MIN_SCORE_THRESHOLD": "最低分数阈值",
        "MIN_LENGTH_THRESHOLD": "最低长度阈值",
        "PERFORMANCE_WINDOW": "性能评估窗口大小",
        "MIN_AVG_REWARD": "最低平均奖励阈值",
        "MAX_TIME_PER_EPISODE": "单轮最大允许时间",
        "SNAKE_GROWTH_RATE_THRESHOLD": "蛇长度增长率阈值",
        "CONVERGENCE_THRESHOLD": "收敛阈值",
        "EXPLORATION_EFFICIENCY_THRESHOLD": "探索效率阈值",
        "DECISION_QUALITY_THRESHOLD": "决策质量阈值",
        "STABILITY_WINDOW": "稳定性评估窗口大小",
        "MIN_Q_VALUE_DIFFERENCE": "最小Q值差异阈值",
        "MODEL_DIR": "模型保存目录",
        "MODEL_EXTENSION": "模型文件扩展名",
        "RESULT_DIR": "测试结果目录",
        "RESULT_IMG_DIR": "测试图像结果目录",
        "RESULT_DATA_DIR": "测试数据结果目录",
        "SAVE_GAMEPLAY_SCREEN": "是否保存游戏画面",
        "SCREENSHOT_DIR": "游戏画面保存目录",
        "SCREENSHOT_QUALITY": "截图质量",
        "SAVE_GAMEPLAY_GIF": "是否保存游戏画面为GIF",
        "GIF_FPS": "GIF帧率",
        "GIF_LOOP": "GIF循环次数",
        "GIF_QUALITY": "GIF质量",
        "GIF_SUBSAMPLE": "GIF子采样"
    }
}

class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = Path(config_file)
        # 初始默认配置
        self.initial_default_config = {
            "game": {
                "GRID_WIDTH": 16,
                "GRID_HEIGHT": 8,
                "STATE_SIZE": 12,
                "ACTION_SIZE": 4
            },
            "training": {
                "EPISODES": 10000,
                "BATCH_SIZE": 64,
                "GAMMA": 0.90,
                "LEARNING_RATE": 0.0005,
                "EPSILON_INIT": 1.0,
                "EPSILON_MIN": 0.05,
                "EPSILON_DECAY": 0.995,
                "REPLAY_BUFFER_SIZE": 20000,
                "TARGET_UPDATE_FREQ": 300
            },
            "model": {
                "SAVE_INTERVAL": 500,
                "MODEL_DIR": "saved_models",
                "LOG_DIR": "logs",
                "CHECKPOINT_PREFIX": "snake_agent_",
                "MODEL_EXTENSION": ".keras",
                "TENSORBOARD_LOG_DIR": "logs/tensorboard"
            },
            "test": {
                "GRID_SIZE": 40,
                "TEST_EPISODES": 50,
                "MAX_STEPS": 1000,
                "FPS": 10,
                "EXPLORATION_RATE": 0.1,
                "MIN_SCORE_THRESHOLD": 5,
                "MIN_LENGTH_THRESHOLD": 5,
                "PERFORMANCE_WINDOW": 5,
                "MIN_AVG_REWARD": 0.5,
                "MAX_TIME_PER_EPISODE": 60,
                "SNAKE_GROWTH_RATE_THRESHOLD": 0.3,
                "CONVERGENCE_THRESHOLD": 0.1,
                "EXPLORATION_EFFICIENCY_THRESHOLD": 0.7,
                "DECISION_QUALITY_THRESHOLD": 0.8,
                "STABILITY_WINDOW": 3,
                "MIN_Q_VALUE_DIFFERENCE": 0.5,
                "MODEL_DIR": "saved_models",
                "MODEL_EXTENSION": ".keras",
                "RESULT_DIR": "test_results",
                "RESULT_IMG_DIR": "test_results/images",
                "RESULT_DATA_DIR": "test_results/data",
                "SAVE_GAMEPLAY_SCREEN": True,
                "SCREENSHOT_DIR": "game_img",
                "SCREENSHOT_QUALITY": 95,
                "SAVE_GAMEPLAY_GIF": True,
                "GIF_FPS": 10,
                "GIF_LOOP": 0,
                "GIF_QUALITY": 85,
                "GIF_SUBSAMPLE": 1
            }
        }
        self.load_config()
    
    def load_config(self):
        """从JSON文件加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.initial_default_config
            self.save_config()
    
    def save_config(self):
        """将配置保存到JSON文件"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_config_value(self, section, key):
        """获取配置值"""
        return self.config[section][key]
    
    def set_config_value(self, section, key, value):
        """设置配置值"""
        self.config[section][key] = value

class ConfigEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake Game Configuration Editor")
        self.root.geometry("550x500")
        self.config_manager = ConfigManager()
        self.create_widgets()
        self.load_config_values()
    
    def create_widgets(self):
        # 创建Notebook控件用于分页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建各个配置页
        self.create_game_frame()
        self.create_training_frame()
        self.create_model_frame()
        self.create_test_frame()
        
        # 创建按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.save_button = tk.Button(
            button_frame, 
            text="保存配置", 
            command=self.save_config
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(
            button_frame, 
            text="恢复默认", 
            command=self.reset_config
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.close_button = tk.Button(
            button_frame, 
            text="关闭", 
            command=self.root.destroy
        )
        self.close_button.pack(side=tk.RIGHT, padx=5)
    
    def create_game_frame(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="游戏参数")
        
        # 创建滚动区域
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 游戏参数输入字段
        game_params = self.config_manager.initial_default_config["game"]
        self.game_entries = {}
        row = 0
        for key, value in game_params.items():
            # 使用中文翻译
            translated_key = CONFIG_TRANSLATIONS["game"].get(key, key)
            label = ttk.Label(scrollable_frame, text=translated_key + ":")
            label.grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
            
            entry = ttk.Entry(scrollable_frame, width=20)
            entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
            self.game_entries[key] = entry
            row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_training_frame(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="训练参数")
        
        # 创建滚动区域
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 训练参数输入字段
        training_params = self.config_manager.initial_default_config["training"]
        self.training_entries = {}
        row = 0
        for key, value in training_params.items():
            # 使用中文翻译
            translated_key = CONFIG_TRANSLATIONS["training"].get(key, key)
            label = ttk.Label(scrollable_frame, text=translated_key + ":")
            label.grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
            
            entry = ttk.Entry(scrollable_frame, width=20)
            entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
            self.training_entries[key] = entry
            row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_model_frame(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="模型参数")
        
        # 创建滚动区域
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 模型参数输入字段
        model_params = self.config_manager.initial_default_config["model"]
        self.model_entries = {}
        row = 0
        for key, value in model_params.items():
            # 使用中文翻译
            translated_key = CONFIG_TRANSLATIONS["model"].get(key, key)
            label = ttk.Label(scrollable_frame, text=translated_key + ":")
            label.grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
            
            entry = ttk.Entry(scrollable_frame, width=20)
            entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
            self.model_entries[key] = entry
            row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_test_frame(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="测试参数")
        
        # 创建滚动区域
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 测试参数输入字段
        test_params = self.config_manager.initial_default_config["test"]
        self.test_entries = {}
        row = 0
        for key, value in test_params.items():
            # 使用中文翻译
            translated_key = CONFIG_TRANSLATIONS["test"].get(key, key)
            label = ttk.Label(scrollable_frame, text=translated_key + ":")
            label.grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)
            
            entry = ttk.Entry(scrollable_frame, width=20)
            entry.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
            self.test_entries[key] = entry
            row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def load_config_values(self):
        """从配置文件加载值到输入框"""
        # 加载游戏参数
        for key, entry in self.game_entries.items():
            value = self.config_manager.get_config_value("game", key)
            entry.delete(0, tk.END)
            entry.insert(0, str(value))
        
        # 加载训练参数
        for key, entry in self.training_entries.items():
            value = self.config_manager.get_config_value("training", key)
            entry.delete(0, tk.END)
            entry.insert(0, str(value))
        
        # 加载模型参数
        for key, entry in self.model_entries.items():
            value = self.config_manager.get_config_value("model", key)
            entry.delete(0, tk.END)
            entry.insert(0, str(value))
        
        # 加载测试参数
        for key, entry in self.test_entries.items():
            value = self.config_manager.get_config_value("test", key)
            entry.delete(0, tk.END)
            entry.insert(0, str(value))
    
    def save_config(self):
        """保存配置到文件，进行类型转换"""
        try:
            # 获取类型定义
            type_definitions = config_loader.type_definitions
            
            # 保存游戏参数
            for key, entry in self.game_entries.items():
                value_str = entry.get()
                target_type = type_definitions["game"][key]
                value = self.convert_value(value_str, target_type)
                self.config_manager.set_config_value("game", key, value)
            
            # 保存训练参数
            for key, entry in self.training_entries.items():
                value_str = entry.get()
                target_type = type_definitions["training"][key]
                value = self.convert_value(value_str, target_type)
                self.config_manager.set_config_value("training", key, value)
            
            # 保存模型参数
            for key, entry in self.model_entries.items():
                value_str = entry.get()
                target_type = type_definitions["model"][key]
                value = self.convert_value(value_str, target_type)
                self.config_manager.set_config_value("model", key, value)
            
            # 保存测试参数
            for key, entry in self.test_entries.items():
                value_str = entry.get()
                target_type = type_definitions["test"][key]
                value = self.convert_value(value_str, target_type)
                self.config_manager.set_config_value("test", key, value)
            
            self.config_manager.save_config()
            messagebox.showinfo("成功", "配置已保存成功！")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置时发生错误：{str(e)}")
    
    def convert_value(self, value_str, target_type):
        """将字符串值转换为目标类型"""
        if target_type == int:
            return int(value_str)
        elif target_type == float:
            return float(value_str)
        elif target_type == bool:
            return value_str.lower() == 'true'
        elif target_type == str:
            return value_str
        else:
            return value_str
    
    def reset_config(self):
        """恢复到初始默认配置 - 使用硬编码的初始默认值"""
        if messagebox.askyesno("确认", "确定要恢复到初始默认配置吗？"):
            # 使用硬编码的初始默认配置
            self.config_manager.config = self.config_manager.initial_default_config
            # 保存到JSON文件
            self.config_manager.save_config()
            # 更新输入框中的值
            self.load_config_values()
            messagebox.showinfo("成功", "已恢复到初始默认配置！")

def main():
    root = tk.Tk()
    app = ConfigEditorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()