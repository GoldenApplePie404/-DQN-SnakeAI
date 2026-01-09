import sys

def tflite_to_c_array(input_file, output_file, array_name="g_snake_model"):
    """
    将TFLite模型转换为C语言数组
    input_file: 输入.tflite文件路径
    output_file: 输出.h文件路径
    array_name: 生成的数组名称
    """
    try:
        # 读取二进制文件
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # 生成C数组内容
        c_array = []
        c_array.append(f"const unsigned char {array_name}[] = {{")
        
        # 每16个字节一行，格式化输出
        for i, byte in enumerate(data):
            if i % 16 == 0:
                c_array.append("    ")
            c_array.append(f"0x{byte:02x}, ")
            if (i + 1) % 16 == 0 and i != len(data) - 1:
                c_array.append("\n")
        
        # 移除最后一个逗号
        if c_array[-1].endswith(", "):
            c_array[-1] = c_array[-1][:-2]
        
        c_array.append("\n};\n")
        c_array.append(f"const unsigned int {array_name}_len = {len(data)};\n")
        
        # 写入输出文件
        with open(output_file, 'w') as f:
            f.write(''.join(c_array))
        
        print(f"成功生成C数组文件: {output_file}")
        print(f"数组名称: {array_name}")
        print(f"模型大小: {len(data)} 字节")
        return True
        
    except FileNotFoundError:
        print(f"错误: 未找到输入文件 {input_file}")
        return False
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("TFLite模型转C数组工具")
    print("=" * 50)
    
    # 获取用户输入，支持默认值
    input_path = input("请输入TFLite模型文件路径（默认：snake_model.tflite）: ").strip()
    if not input_path:
        input_path = "snake_model.tflite"
    
    output_path = input("请输入输出C头文件路径（默认：model_array.h）: ").strip()
    if not output_path:
        output_path = "model_array.h"
    
    array_name = input("请输入转换后的模型数组名称（默认：g_snake_model）: ").strip()
    if not array_name:
        array_name = "g_snake_model"
    
    print("\n转换参数:")
    print(f"输入模型: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"数组名称: {array_name}")
    print("=" * 50)
    
    # 执行转换
    tflite_to_c_array(input_path, output_path, array_name)