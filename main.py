import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置随机种子以确保结果可重复
np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    """加载数据"""
    script_dir = Path(__file__).parent
    data_path = script_dir / 'data' / 'UV_COD_Correct_Dataset.csv'
    data = pd.read_csv(data_path)
    wavelength_cols = [f'{i}nm' for i in range(200, 376)]  # 200nm to 375nm
    X = data[wavelength_cols].values  # UV spectrum data
    y = data['COD标签(mg/L)'].values  # COD values to predict
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """预处理数据"""
    # 确保数据分割的一致性
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 重塑为适合CNN的形状
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_model(input_shape):
    """构建CNN模型"""
    model = Sequential([
        Conv1D(32, 5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(32, 5, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),  # 增加dropout率
        
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.4),  # 增加dropout率
        
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        Conv1D(128, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.5),  # 增加dropout率
        
        Dense(256, activation='relu'),
        Dropout(0.6),  # 增加dropout率
        Dense(128, activation='relu'),
        Dropout(0.5),  # 增加dropout率
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def plot_training_history(history, save_path):
    """绘制训练历史图表"""
    import os
    from pathlib import Path
    
    # 确保保存目录存在
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='训练MAE')
    plt.plot(history.history['val_mae'], label='验证MAE')
    plt.title('训练和验证MAE')
    plt.xlabel('轮数')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true, y_pred, save_path):
    """绘制预测结果图表"""
    import os
    from pathlib import Path
    
    # 确保y_pred是1维数组
    y_pred = y_pred.flatten()
    
    # 确保保存目录存在
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # 实际值 vs 预测值
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('实际COD vs 预测COD')
    plt.xlabel('实际COD (mg/L)')
    plt.ylabel('预测COD (mg/L)')
    plt.grid(True, alpha=0.3)
    
    # 计算R²并显示
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, save_path):
    """绘制残差图表"""
    import os
    from pathlib import Path
    
    # 确保y_pred是1维数组
    y_pred = y_pred.flatten()
    residuals = y_true - y_pred
    
    # 确保保存目录存在
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差图')
    plt.xlabel('预测COD (mg/L)')
    plt.ylabel('残差 (实际 - 预测)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_model():
    """训练模型"""
    print("加载数据...")
    X, y = load_data()
    print(f"数据集形状: {X.shape}")
    
    print("使用5折交叉验证评估模型...")
    
    # 初始化K折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 用于存储每折的结果
    cv_scores = []
    histories = []
    
    fold = 1
    for train_index, test_index in kfold.split(X):
        print(f"\n第 {fold} 折训练...")
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        # 预处理数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # 重塑为适合CNN的形状
        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        # 构建模型
        model = build_model((X_train_scaled.shape[1], 1))
        
        # 训练模型
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.0001)
        ]
        
        history = model.fit(
            X_train_scaled, y_train_fold,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0  # 关闭详细输出，因为会有很多折叠
        )
        
        # 预测和评估
        y_pred_fold = model.predict(X_test_scaled, verbose=0)
        mse = mean_squared_error(y_test_fold, y_pred_fold)
        mae = mean_absolute_error(y_test_fold, y_pred_fold)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_fold, y_pred_fold)
        
        print(f"第 {fold} 折 - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        cv_scores.append({
            'fold': fold,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model': model,
            'scaler': scaler,
            'history': history,
            'y_true': y_test_fold,
            'y_pred': y_pred_fold.flatten()  # 展平预测结果
        })
        
        histories.append(history)
        fold += 1
    
    # 计算平均性能指标
    avg_mse = np.mean([score['mse'] for score in cv_scores])
    avg_mae = np.mean([score['mae'] for score in cv_scores])
    avg_rmse = np.mean([score['rmse'] for score in cv_scores])
    avg_r2 = np.mean([score['r2'] for score in cv_scores])
    
    std_mse = np.std([score['mse'] for score in cv_scores])
    std_mae = np.std([score['mae'] for score in cv_scores])
    std_rmse = np.std([score['rmse'] for score in cv_scores])
    std_r2 = np.std([score['r2'] for score in cv_scores])
    
    print(f"\n5折交叉验证平均结果:")
    print(f"MSE: {avg_mse:.2f}, 标准差: {std_mse:.2f}")
    print(f"MAE: {avg_mae:.2f}, 标准差: {std_mae:.2f}")
    print(f"RMSE: {avg_rmse:.2f}, 标准差: {std_rmse:.2f}")
    print(f"R²: {avg_r2:.4f}, 标准差: {std_r2:.4f}")
    
    # 选择性能最好的模型作为最终模型
    best_fold_idx = np.argmax([score['r2'] for score in cv_scores])  # R²越大越好
    best_model = cv_scores[best_fold_idx]['model']
    best_scaler = cv_scores[best_fold_idx]['scaler']
    
    print(f"\n选择第 {best_fold_idx+1} 折的模型作为最终模型")
    
    # 使用最佳模型在对应的测试集上评估
    best_y_true = cv_scores[best_fold_idx]['y_true']
    best_y_pred = cv_scores[best_fold_idx]['y_pred']
    
    best_mse = mean_squared_error(best_y_true, best_y_pred)
    best_mae = mean_absolute_error(best_y_true, best_y_pred)
    best_rmse = np.sqrt(best_mse)
    best_r2 = r2_score(best_y_true, best_y_pred)
    
    print(f"最佳模型性能:")
    print(f"MSE: {best_mse:.2f}")
    print(f"MAE: {best_mae:.2f}")
    print(f"RMSE: {best_rmse:.2f}")
    print(f"R²: {best_r2:.4f}")
    
    print("保存最佳模型...")
    script_dir = Path(__file__).parent
    # 创建新的模型子目录
    cnn_model_dir = script_dir / 'models' / '1d_cnn'
    cnn_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = cnn_model_dir / 'cod_model.keras'
    
    # 保存最佳模型
    best_model.save(model_path)
    
    # 同时保存最佳的scaler
    import joblib
    scaler_path = cnn_model_dir / 'scaler.pkl'
    joblib.dump(best_scaler, scaler_path)
    
    # 使用最佳模型进行预测用于可视化
    y_pred_final = best_y_pred
    y_test = best_y_true
    
    print("保存训练结果...")
    import datetime
    import os
    
    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 使用相对于脚本位置的路径
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results目录路径: {results_dir}")
    
    # 构造完整的结果文件路径
    summary_file_path = results_dir / 'training_summary.txt'
    print(f"结果文件路径: {summary_file_path}")
    
    try:
        # 保存为文本文件，更易读
        with open(summary_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n训练时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("5折交叉验证结果:\n")
            f.write(f"MSE: {avg_mse:.2f}, 标准差: {std_mse:.2f}\n")
            f.write(f"MAE: {avg_mae:.2f}, 标准差: {std_mae:.2f}\n")
            f.write(f"RMSE: {avg_rmse:.2f}, 标准差: {std_rmse:.2f}\n")
            f.write(f"R²: {avg_r2:.4f}, 标准差: {std_r2:.4f}\n")
            f.write("最佳模型性能:\n")
            f.write(f"MSE: {best_mse:.2f}\n")
            f.write(f"MAE: {best_mae:.2f}\n")
            f.write(f"RMSE: {best_rmse:.2f}\n")
            f.write(f"R²: {best_r2:.4f}\n")
            f.write("-" * 50 + "\n")  # 分隔线，便于区分每次训练
        print("训练结果已成功保存到文件")
    except Exception as e:
        print(f"保存训练结果时出错: {str(e)}")
        print(f"请检查是否有足够的权限写入该目录")
        import traceback
        traceback.print_exc()
    
    print("生成并保存可视化结果...")
    # 生成并保存可视化图表 - 使用最终模型的历史和预测
    # 使用相对于脚本位置的路径
    script_dir = Path(__file__).parent
    plot_training_history(history, script_dir / 'results' / 'training_history.png')
    plot_predictions(y_test, y_pred_final, script_dir / 'results' / 'predictions.png')
    plot_residuals(y_test, y_pred_final, script_dir / 'results' / 'residuals.png')
    
    print("训练结果和可视化图表已保存到 results/ 目录")
    
    return best_model, best_scaler

def predict_cod(spectrum, model, scaler):
    """预测单个光谱的COD值"""
    spectrum = np.array(spectrum).reshape(1, -1)
    scaled = scaler.transform(spectrum)
    scaled = scaled.reshape(1, -1, 1)
    prediction = model.predict(scaled, verbose=0)
    return prediction[0][0]

def load_model_and_scaler():
    """加载模型和创建标准化器"""
    script_dir = Path(__file__).parent
    model_path = script_dir / 'models' / '1d_cnn' / 'cod_model.keras'
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print(f"无法加载模型文件: {model_path}")
        return None, None
    import joblib
    scaler_path = script_dir / 'models' / '1d_cnn' / 'scaler.pkl'
    try:
        scaler = joblib.load(scaler_path)
    except:
        print(f"无法加载scaler文件: {scaler_path}")
        return None, None
    return model, scaler

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        # 预测模式
        print("加载模型...")
        model, scaler = load_model_and_scaler()
        
        # 预测前5个样本
        print("预测样本...")
        X, y = load_data()
        for i in range(5):
            pred = predict_cod(X[i], model, scaler)
            print(f"样本 {i+1}: 实际COD = {y[i]:.2f}, 预测COD = {pred:.2f}")
    else:
        # 训练模式
        train_model()

if __name__ == "__main__":
    main()