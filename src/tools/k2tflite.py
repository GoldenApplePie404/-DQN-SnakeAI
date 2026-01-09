import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import tensorflow as tf
from src.utils.logger import ColorLogger

def scan_keras_models(model_dir):
    """扫描指定目录下的所有.keras模型文件
    
    参数:
        model_dir (Path): 模型目录路径
        
    返回:
        list: 包含(文件编号, 文件路径)元组的列表
    """
    if not model_dir.exists():
        ColorLogger.error(f"模型目录不存在: {model_dir}")
        return []
    
    keras_files = list(model_dir.glob("*.keras"))
    if not keras_files:
        ColorLogger.warning("未找到任何.keras模型文件")
        return []
    
    # 按修改时间排序（最新的在前）
    keras_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # 为每个文件分配编号
    indexed_files = [(i+1, file) for i, file in enumerate(keras_files)]
    return indexed_files

def select_model(indexed_files):
    """让用户选择要转换的模型
    
    参数:
        indexed_files (list): 包含(文件编号, 文件路径)元组的列表
        
    返回:
        Path: 选定的模型文件路径，如果用户取消则返回None
    """
    if not indexed_files:
        return None
    
    # 显示可选模型列表
    print("\n可用模型列表:")
    for idx, file in indexed_files:
        mod_time = os.path.getmtime(file)
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"{idx}. {file.name} (修改时间: {mod_time}, 大小: {file_size:.2f}MB)")
    
    # 获取用户输入
    while True:
        try:
            choice = input("\n请输入要转换的模型编号(1-{}), 或输入q取消: ".format(len(indexed_files)))
            if choice.lower() == 'q':
                return None
                
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(indexed_files):
                return indexed_files[choice_idx][1]
            else:
                print(f"无效输入，请输入1-{len(indexed_files)}之间的数字或q取消")
        except ValueError:
            print("无效输入，请输入数字或q取消")

def convert_to_tflite(keras_path, output_dir="tflite_models"):
    """将.keras模型转换为TFLite格式
    
    参数:
        keras_path (Path): 输入的.keras模型路径
        output_dir (str): 输出目录名，默认为"tflite_models"
        
    返回:
        Path: 生成的TFLite模型路径
    """
    # 准备输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 构造输出路径
    output_path = output_dir / (keras_path.stem + ".tflite")
    
    try:
        # 加载Keras模型
        model = tf.keras.models.load_model(keras_path)
        
        # 创建TFLite转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 转换模型
        tflite_model = converter.convert()
        
        # 保存TFLite模型
        with open(output_path, "wb") as f:
            f.write(tflite_model)
            
        ColorLogger.success(f"模型转换成功，已保存至: {output_path}")
        return output_path
        
    except Exception as e:
        ColorLogger.error(f"模型转换失败: {str(e)}")
        return None

def main():
    """主函数"""
    # 设置项目根目录
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "saved_models"
    
    # 初始化日志
    ColorLogger.info("===== 模型转换工具 =====")
    
    # 扫描.keras模型
    indexed_files = scan_keras_models(model_dir)
    if not indexed_files:
        ColorLogger.error("没有找到可用的.keras模型，程序退出")
        return
    
    # 用户选择模型
    selected_model = select_model(indexed_files)
    if selected_model is None:
        ColorLogger.info("用户取消了模型选择，程序退出")
        return
    
    # 转换模型
    convert_to_tflite(selected_model)

if __name__ == "__main__":
    main()
