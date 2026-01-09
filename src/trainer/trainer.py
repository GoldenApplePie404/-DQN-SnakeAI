import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import tensorflow as tf

from src.model.q_network import QNetwork
from src.utils.logger import ColorLogger
from src.utils.config import Config
from src.utils.device import get_training_device
from src.utils.agent_trainer import AgentTrainer
from src.utils.model_manager import ModelManager
from src.utils.replay_buffer import ReplayBuffer
from src.utils.train_log import TrainingLogger
from src.utils.env_handler import EnvironmentHandler



def main(render_mode=None, load_prev_model=True):
    """主程序入口，整合各模块执行训练流程"""
    # 初始化设备
    device = get_training_device()
    ColorLogger.info(f"训练设备: {device}")
    
    with tf.device(device):
        # 初始化核心组件
        env_handler = EnvironmentHandler(render_mode=render_mode)
        agent = QNetwork(Config.STATE_SIZE, Config.ACTION_SIZE, Config.LEARNING_RATE)
        replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)
        
        # 初始化辅助模块
        model_manager = ModelManager(agent)
        logger = TrainingLogger()
        
        # 加载模型并获取起始轮次
        start_episode = model_manager.load_latest_model(load_prev_model)
        
        # 初始化训练器
        trainer = AgentTrainer(
            agent=agent,
            env_handler=env_handler,
            replay_buffer=replay_buffer,
            model_manager=model_manager,
            logger=logger
        )
        
        # 开始训练
        score_history, loss_history, episodes_x = trainer.train(start_episode)
        
        # 清理环境
        env_handler.close()
        return score_history, loss_history, episodes_x

if __name__ == "__main__":
    config = Config()  
    main()  
