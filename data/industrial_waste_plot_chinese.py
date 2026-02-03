import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 获取当前脚本所在目录
script_dir = Path(__file__).parent
# 读取数据 - 使用相对于脚本位置的路径
data = pd.read_csv(script_dir / 'UV_COD_Correct_Dataset.csv')

# 按COD值排序后，均匀选择20个样本，确保COD分布均匀
sorted_data = data.sort_values('COD标签(mg/L)')
# 选择均匀分布的20个样本
indices = np.linspace(0, len(sorted_data)-1, 20, dtype=int)
selected_samples = sorted_data.iloc[indices]

print(f"选择了 {len(selected_samples)} 个COD分布均匀的样本进行绘图")

# 创建波长列表 (200nm to 375nm)
wavelengths = [i for i in range(200, 376)]

# 绘制光谱图
plt.figure(figsize=(12, 8))

for idx, (row_idx, sample) in enumerate(selected_samples.iterrows()):
    # 提取该样本的UV光谱数据 (190nm to 400nm)
    uv_spectrum = [sample[str(wl)+'nm'] for wl in wavelengths]
    
    # 获取COD含量和样本ID
    cod_value = sample['COD标签(mg/L)']
    sample_id = sample['样品ID']
    
    plt.plot(wavelengths, uv_spectrum, label=f'样本 {sample_id}, COD: {cod_value} mg/L', linewidth=2)

plt.title('COD均匀分布样本 UV-Vis 光谱图 (20个不同COD含量的样本)', fontsize=14)
plt.xlabel('波长 (nm)', fontsize=12)
plt.ylabel('吸光度', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(200, 375)
plt.tight_layout()
plt.show()

# 打印所选样本的信息
print("\n所选样本信息:")
for idx, (row_idx, sample) in enumerate(selected_samples.iterrows()):
    print(f"样本ID: {sample['样品ID']}, COD: {sample['COD标签(mg/L)']} mg/L, "
          f"pH: {sample['pH']}, 浊度: {sample['浊度(NTU)']} NTU")