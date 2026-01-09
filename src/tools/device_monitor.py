import psutil
import time
from datetime import datetime
import csv
import os

# 检查 Matplotlib 是否存在
try:
    import matplotlib.pyplot as plt 
    import matplotlib.dates as mdates
    MATPLOTLIB_EXISTS = True
except ImportError:
    MATPLOTLIB_EXISTS = False

# 替代 GPUtil 的 GPU 监控
def get_gpu_info():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # 默认 GPU 0
        utilization = nvmlDeviceGetUtilizationRates(handle)
        memory = nvmlDeviceGetMemoryInfo(handle)
        gpu_info = {
            "gpu_load_percent": utilization.gpu,
            "gpu_mem_total_gb": round(memory.total / (1024 ** 3), 2),
            "gpu_mem_used_gb": round(memory.used / (1024 ** 3), 2),
            "gpu_mem_percent": round((memory.used / memory.total) * 100, 2),
        }
        nvmlShutdown()
        return gpu_info
    except:
        return None  # 无 GPU 或驱动未安装

def get_system_info(interval=1.0, output_dir="log/device_log"):
    # 创建存储目录
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"system_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # 初始化 CSV 文件
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "timestamp",
            "cpu_percent",
            "mem_total_gb",
            "mem_used_gb",
            "mem_percent",
            "disk_total_gb",
            "disk_used_gb",
            "disk_percent",
            "gpu_load_percent",
            "gpu_mem_total_gb",
            "gpu_mem_used_gb",
            "gpu_mem_percent",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 存储数据用于绘图
        data = {
            "timestamp": [],
            "cpu_percent": [],
            "mem_used_gb": [],
            "mem_percent": [],
            "disk_used_gb": [],
            "disk_percent": [],
            "gpu_load_percent": [],
            "gpu_mem_used_gb": [],
            "gpu_mem_percent": [],
        }

        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # CPU
                cpu_percent = psutil.cpu_percent(interval=interval)
                
                # 内存
                mem = psutil.virtual_memory()
                mem_total = round(mem.total / (1024 ** 3), 2)
                mem_used = round(mem.used / (1024 ** 3), 2)
                mem_percent = mem.percent
                
                # 磁盘
                disk = psutil.disk_usage('/')
                disk_total = round(disk.total / (1024 ** 3), 2)
                disk_used = round(disk.used / (1024 ** 3), 2)
                disk_percent = disk.percent
                
                # GPU（可选）
                gpu_info = get_gpu_info() if get_gpu_info() else {
                    "gpu_load_percent": 0,
                    "gpu_mem_total_gb": 0,
                    "gpu_mem_used_gb": 0,
                    "gpu_mem_percent": 0,
                }
                
                # 合并数据
                record = {
                    "timestamp": timestamp,
                    "cpu_percent": cpu_percent,
                    "mem_total_gb": mem_total,
                    "mem_used_gb": mem_used,
                    "mem_percent": mem_percent,
                    "disk_total_gb": disk_total,
                    "disk_used_gb": disk_used,
                    "disk_percent": disk_percent,
                    **gpu_info
                }
                
                # 写入 CSV
                writer.writerow(record)
                
                # 存储数据用于绘图
                for key in data:
                    if key in record:
                        if key == "timestamp":
                            data[key].append(record[key])
                        else:
                            data[key].append(float(record[key]))
                
                # 打印格式化数据（终端友好）
                print("\n\n\n\n\n" + "="*50)
                print(f"时间: {timestamp}")
                print("-"*50)
                print(f"CPU 使用率: {cpu_percent}%")
                print("-"*50)
                print(f"内存使用: {mem_used} GB/{mem_total} GB ({mem_percent}%)")
                print("-"*50)    
                print(f"磁盘使用: {disk_used} GB/{disk_total} GB ({disk_percent}%)")
                print("-"*50)
                if gpu_info["gpu_load_percent"] > 0:
                    print(f"GPU 使用率: {gpu_info['gpu_load_percent']}%")
                    print(f"GPU 显存: {gpu_info['gpu_mem_used_gb']}/{gpu_info['gpu_mem_total_gb']} GB ({gpu_info['gpu_mem_percent']}%)")
                else:
                    print("未检测到 GPU 或 GPU 监控不可用")
                print("="*50)
                
        except KeyboardInterrupt:
            print("\n监控已停止，数据已保存至:", csv_file)
            
            # 如果安装了 Matplotlib，则生成图表
            if MATPLOTLIB_EXISTS:
                plot_system_metrics(data, output_dir)
            else:
                print("Matplotlib 未安装，跳过图表生成。")

def plot_system_metrics(data, output_dir):
    plt.figure(figsize=(12, 8))
    full_timestamps = data["timestamp"]
    
    # 转换为datetime对象
    times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in full_timestamps]
    dates = [ts.date() for ts in times]
    
    # 通用格式化函数
    def format_time_axis(ax, start_time, end_time):
        duration = (end_time - start_time).total_seconds()
        
        
        if duration <= 60:  # 1分钟内
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, int(duration/5))))
        elif duration <= 3600:  # 1小时内
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int(duration/300))))
        else:  # 更长时间
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(duration/3600))))
        
        plt.xticks(rotation=45)
    
    # CPU子图
    plt.subplot(3, 1, 1)
    plt.plot(times, data["cpu_percent"], label="CPU Usage (%)", color="red")
    plt.title(f"System Resource Monitoring (Date: {dates[0]})")
    plt.ylabel("CPU (%)")
    plt.grid(True)
    ax = plt.gca()
    format_time_axis(ax, times[0], times[-1])
    
    # 内存使用
    plt.subplot(3, 1, 2)
    plt.plot(times, data["mem_used_gb"], label="Memory Used (GB)", color="blue")
    plt.plot(times, data["mem_percent"], label="Memory Usage (%)", color="green", linestyle="--")
    plt.ylabel("Memory")
    plt.grid(True)
    ax = plt.gca()
    format_time_axis(ax, times[0], times[-1])
    
    # 磁盘使用
    plt.subplot(3, 1, 3)
    plt.plot(times, data["disk_used_gb"], label="Disk Used (GB)", color="purple")
    plt.plot(times, data["disk_percent"], label="Disk Usage (%)", color="orange", linestyle="--")
    plt.ylabel("Disk")
    plt.grid(True)
    ax = plt.gca()
    format_time_axis(ax, times[0], times[-1])
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"system_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"监控图表已保存至: {plot_file}")
    
    # 如果有 GPU 数据，单独生成 GPU 图表
    if any(data["gpu_load_percent"]):
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.plot(times, data["gpu_load_percent"], label="GPU Usage (%)", color="red")
        plt.title(f"GPU Monitoring (Date: {dates[0]})")  # 标题中显示日期
        plt.ylabel("GPU (%)")
        ax = plt.gca()
        format_time_axis(ax, times[0], times[-1])
        
        plt.subplot(2, 1, 2)
        plt.plot(times, data["gpu_mem_used_gb"], label="GPU Memory Used (GB)", color="blue")
        plt.plot(times, data["gpu_mem_percent"], label="GPU Memory Usage (%)", color="green", linestyle="--")
        plt.ylabel("GPU Memory")
        plt.grid(True)
        ax = plt.gca()
        format_time_axis(ax, times[0], times[-1])
        
        plt.tight_layout()
        gpu_plot_file = os.path.join(output_dir, f"gpu_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(gpu_plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"GPU 监控图表已保存至: {gpu_plot_file}")

if __name__ == "__main__":
    get_system_info(interval=1.0)