import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置随机种子以确保结果可重复
np.random.seed(36)

# 生成模拟UV光谱数据
def generate_uv_spectrum():
    # 波长范围 (200nm to 375nm)
    wavelengths = list(range(200, 376))
    
    # 生成多个样本数据
    num_samples = 50  # 生成50个样本
    
    # 创建数据字典
    data = {}
    
    # 添加波长列
    for wl in wavelengths:
        data[f'{wl}nm'] = []
    
    # 添加其他列
    data['样品ID'] = []
    data['COD标签(mg/L)'] = []
    data['pH'] = []
    data['浊度(NTU)'] = []
    
    # 生成每个样本的数据
    for i in range(num_samples):
        # 生成COD值 (范围: 20-500 mg/L)
        cod_value = np.random.uniform(20, 500)
        
        # 生成基础光谱曲线
        # 主峰值在225nm左右
        spectrum = []
        for wl in wavelengths:
            # 高斯分布模拟主峰
            main_peak = np.exp(-0.5 * ((wl - 225) / 10) ** 2)
            
            # 次峰在270nm左右
            secondary_peak = 0.2 * np.exp(-0.5 * ((wl - 270) / 15) ** 2)
            
            # 基线偏移
            baseline = 0.05 + 0.0001 * (wl - 200)
            
            # 根据波长调整噪声水平，高波段噪声更小
            if wl > 300:
                noise = np.random.normal(0, 0.005)
            else:
                noise = np.random.normal(0, 0.01)
            
            # 总吸光度，与COD值相关，按比例缩小一半
            absorbance = (main_peak + secondary_peak + baseline + noise) * (cod_value / 200)
            
            # 确保吸光度为正值
            absorbance = max(0, absorbance)
            
            spectrum.append(absorbance)
        
        # 移动平均滤波，使曲线更平滑
        def moving_average(data, window_size=5):
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')
        
        # 对整个光谱应用基本平滑
        spectrum = moving_average(spectrum)
        
        # 对高波段（>300nm）应用额外的平滑处理
        high_wave_spectrum = []
        for idx, (wl, abs_val) in enumerate(zip(wavelengths, spectrum)):
            if wl > 300:
                # 对高波段使用更大的窗口进行平滑
                if idx >= 2 and idx < len(spectrum) - 2:
                    # 7点移动平均
                    avg_val = np.mean(spectrum[max(0, idx-3):min(len(spectrum), idx+4)])
                    high_wave_spectrum.append(avg_val)
                else:
                    high_wave_spectrum.append(abs_val)
            else:
                high_wave_spectrum.append(abs_val)
        
        spectrum = high_wave_spectrum
        
        # 添加光谱数据
        for wl, abs_val in zip(wavelengths, spectrum):
            data[f'{wl}nm'].append(abs_val)
        
        # 添加其他信息
        data['样品ID'].append(f'S{i+1}')
        data['COD标签(mg/L)'].append(round(cod_value, 2))
        data['pH'].append(round(np.random.uniform(6, 8), 2))
        data['浊度(NTU)'].append(round(np.random.uniform(5, 50), 2))
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV文件
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'UV_test_Dataset.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"生成了 {num_samples} 个样本的UV光谱数据")
    print(f"数据已保存到 {csv_path}")
    
    return df

# 绘制生成的光谱数据
def plot_spectra(data):
    # 按COD值排序后，均匀选择20个样本，确保COD分布均匀
    sorted_data = data.sort_values('COD标签(mg/L)')
    # 选择均匀分布的20个样本
    indices = np.linspace(0, len(sorted_data)-1, 20, dtype=int)
    selected_samples = sorted_data.iloc[indices]
    
    print(f"选择了 {len(selected_samples)} 个COD分布均匀的样本进行绘图")
    
    # 创建波长列表 (200nm to 375nm)
    wavelengths = list(range(200, 376))
    
    # 绘制光谱图
    plt.figure(figsize=(12, 8))
    
    for idx, (row_idx, sample) in enumerate(selected_samples.iterrows()):
        # 提取该样本的UV光谱数据
        uv_spectrum = [sample[str(wl)+'nm'] for wl in wavelengths]
        
        # 获取COD含量
        cod_value = sample['COD标签(mg/L)']
        
        plt.plot(wavelengths, uv_spectrum, linewidth=2)
    
    plt.title('UV-Vis 光谱图 (不同COD含量的样本)', fontsize=14)
    plt.xlabel('波长 (nm)', fontsize=12)
    plt.ylabel('吸光度', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(200, 375)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成数据
    df = generate_uv_spectrum()
    
    # 绘制光谱图
    plot_spectra(df)
    
    # 打印前5行数据
    print("\n前5行数据:")
    print(df.head())