import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 設定資料夾路徑
current_directory = os.getcwd()

# 使用列表存放讀取的資料
dataframes = []

# 迴圈讀取每一個檔案
for i in range(1, 18):  # 1 到 17
    file_name = f"L{i}_Train.csv"
    file_path = os.path.join(current_directory, file_name)  # 合併路徑與檔名
    df = pd.read_csv(file_path)  # 讀取 CSV 檔案
    dataframes.append(df)  # 加入列表

# 繪圖和SVR訓練
for i, df in enumerate(dataframes):
    if 'Sunlight(Lux)' in df.columns and 'Power(mW)' in df.columns:
        print(f"DataFrame L{i+1} 包含 Sunlight(Lux) 和 Power(mW) 欄位，正在進行訓練...")
        
        # 過濾條件
        filtered_df = df[ 
            (df['Sunlight(Lux)'] != 117758.2) & 
            (df['Sunlight(Lux)'] != 54612.5)
        ]
        
        # 檢查是否有可用數據
        if len(filtered_df) == 0:
            print(f"DataFrame L{i+1} 沒有可用數據，跳過...")
            continue

        # 特徵和標籤
        X = filtered_df[['Sunlight(Lux)']]
        y = filtered_df['Power(mW)']

        # 切分數據集為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 特徵縮放
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

        # 建立 SVR 模型
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train, y_train)

        # 預測
        y_pred = svr_model.predict(X_test)

        # 繪製預測結果
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, scaler_y.inverse_transform(y_test.reshape(-1, 1)), color='b', label='Actual Power (mW)')
        plt.scatter(X_test, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='r', label='Predicted Power (mW)')
        
        # 添加標籤和標題
        plt.title(f'SVR Prediction for L{i+1}: Sunlight(Lux) vs Power(mW)')
        plt.xlabel('Sunlight (Lux)')
        plt.ylabel('Power (mW)')
        plt.legend()
        
        # 顯示圖表
        plt.show()
    else:
        print(f"DataFrame L{i+1} 缺少 Sunlight(Lux) 或 Power(mW) 欄位")
