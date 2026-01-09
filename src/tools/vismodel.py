import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def simple_visualize_model(model_path):
    """可视化模型结构"""
    model = load_model(model_path)
    print("=" * 50)
    print(f"模型结构摘要 - {os.path.basename(model_path)}")
    print("=" * 50)
    model.summary()

def get_available_models(folder_path):
    """扫描文件夹内的.keras文件，排除final_snake_model.keras"""
    keras_files = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.keras') and file != 'final_snake_model.keras':
                keras_files.append(os.path.join(folder_path, file))
    return keras_files

if __name__ == "__main__":
    # 获取用户输入的文件夹路径
    folder_path = input("请输入包含.keras模型的文件夹路径：")
    
    # 扫描文件夹内的模型文件
    models = get_available_models(folder_path)
    
    if not models:
        print("未找到可用的.keras模型文件（已排除final_snake_model.keras）")
    else:
        # 显示模型列表
        print("\n可用模型文件：")
        for idx, model_path in enumerate(models, 1):
            print(f"{idx}. {os.path.basename(model_path)}")
        
        # 获取用户选择
        while True:
            try:
                choice = int(input("\n请输入要查看的模型编号："))
                if 1 <= choice <= len(models):
                    selected_model = models[choice-1]
                    break
                else:
                    print(f"请输入1到{len(models)}之间的有效编号")
            except ValueError:
                print("请输入有效的数字编号")
        
        # 可视化选中的模型
        print(f"\n正在可视化模型：{os.path.basename(selected_model)}")
        simple_visualize_model(selected_model)