import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def load_trained_model():
    """加载训练好的模型和标准化器"""
    script_dir = Path(__file__).parent
    model_path = script_dir / 'models' / '1d_cnn' / 'cod_model.keras'
    scaler_path = script_dir / 'models' / '1d_cnn' / 'scaler.pkl'
    
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    
    # 加载scaler
    import joblib
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def test_model_on_dataset():
    """在测试数据集上测试模型"""
    print("加载测试数据...")
    # 加载测试数据
    test_data_path = Path(__file__).parent / 'data' / 'UV_test_Dataset.csv'
    test_data = pd.read_csv(test_data_path)
    
    # 提取特征和真实值
    wavelength_cols = [f'{i}nm' for i in range(200, 376)]  # 200nm to 375nm
    X_test = test_data[wavelength_cols].values  # UV spectrum data
    y_true = test_data['COD标签(mg/L)'].values  # True COD values
    
    print(f"测试数据集形状: {X_test.shape}")
    print(f"真实COD值范围: {y_true.min():.2f} - {y_true.max():.2f}")
    
    # 加载模型和标准化器
    print("加载训练好的模型...")
    model, scaler = load_trained_model()
    
    # 预处理测试数据
    print("预处理测试数据...")
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # 进行预测
    print("进行预测...")
    y_pred = model.predict(X_test_scaled, verbose=1)
    
    # 将预测值展平
    y_pred = y_pred.flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n测试集评估结果:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # 创建结果DataFrame
    results_df = test_data[['样品ID', 'COD标签(mg/L)']].copy()
    results_df['预测COD值'] = y_pred
    results_df['绝对误差'] = abs(y_true - y_pred)
    
    # 保存预测结果
    results_path = Path(__file__).parent / 'test_results' / 'test_predictions.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {results_path}")
    
    # 打印前10个样本的预测结果
    print(f"\n前10个样本的预测结果:")
    print(results_df.head(10).to_string())
    
    # 绘制预测结果图
    plot_predictions(y_true, y_pred, r2, Path(__file__).parent / 'test_results' / 'test_predictions.png')
    
    # 绘制残差图
    plot_residuals(y_true, y_pred, Path(__file__).parent / 'test_results' / 'test_residuals.png')
    
    # 保存评估指标到文本文件
    save_metrics_to_file(mse, mae, rmse, r2)
    
    return results_df

def plot_predictions(y_true, y_pred, r2, save_path):
    """绘制预测结果图表"""
    plt.figure(figsize=(10, 8))
    
    # 实际值 vs 预测值
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'测试集实际COD vs 预测COD\nR² = {r2:.4f}')
    plt.xlabel('实际COD (mg/L)')
    plt.ylabel('预测COD (mg/L)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测结果图已保存到: {save_path}")

def plot_residuals(y_true, y_pred, save_path):
    """绘制残差图表"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('测试集残差图')
    plt.xlabel('预测COD (mg/L)')
    plt.ylabel('残差 (实际 - 预测)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"残差图已保存到: {save_path}")

def save_metrics_to_file(mse, mae, rmse, r2):
    """保存评估指标到文本文件"""
    import datetime
    metrics_path = Path(__file__).parent / 'test_results' / 'test_metrics.txt'
    
    with open(metrics_path, 'a', encoding='utf-8') as f:
        f.write(f"\n测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("测试集评估结果:\n")
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write("-" * 30 + "\n")
    
    print(f"评估指标已保存到: {metrics_path}")

if __name__ == "__main__":
    test_results = test_model_on_dataset()
    print("\n测试完成！")