import subprocess
import sys
import os

def is_package_installed(package_name):
    """检查指定的包是否已安装"""
    try:
        # 使用pip show命令检查包是否存在
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"检查包 {package_name} 安装状态时发生错误: {e}")
        return False
def install_requirements():
    """安装requirements.txt中列出的所有包（仅在未安装时）"""
    try:
        # 检查requirements.txt是否存在
        if not os.path.exists('requirements.txt'):
            print("错误: requirements.txt文件不存在")
            return False
            
        print("正在读取requirements.txt...")
        
        # 读取requirements.txt文件
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
        
        # 过滤掉空行和注释行
        packages = []
        for line in requirements:
            line = line.strip()
            # 跳过空行和注释
            if line and not line.startswith('#'):
                packages.append(line)
        
        if not packages:
            print("requirements.txt中没有找到有效的包")
            return True
            
        print(f"找到 {len(packages)} 个包需要检查安装状态:")
        for pkg in packages:
            print(f"  - {pkg}")
            
        # 检查每个包的安装状态并安装未安装的包
        installed_count = 0
        not_installed_count = 0
        
        for package in packages:
            # 提取包名（去除版本号等信息）
            package_name = package.split('>=')[0].split('==')[0].split('<=')[0]
            
            if is_package_installed(package_name):
                print(f"✓ 已安装: {package}")
                installed_count += 1
            else:
                print(f"⚠️ 未安装: {package}")
                try:
                    # 安装包
                    print(f"正在安装: {package}")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"✓ 成功安装: {package}")
                    not_installed_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"✗ 安装失败: {package}")
                    print(f"  错误详情: {e}")
                    return False
                
        print(f"\n安装完成: {installed_count} 个已安装, {not_installed_count} 个新安装")
        if not_installed_count > 0:
            print("所有需要的包都已成功安装!")
        else:
            print("所有依赖包都已经安装完毕!")
        return True
        
    except Exception as e:
        print(f"安装过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = install_requirements()
    if not success:
        print("安装过程出现错误")
