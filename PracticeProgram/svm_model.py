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

# 處理的Sunlight(Lux)異常值
anomalous_values = [117758.2, 54612.5]
# 繪圖和SVR訓練
for i, df in enumerate(dataframes):
    if 'Sunlight(Lux)' in df.columns and 'Power(mW)' in df.columns:
        print(f"DataFrame L{i+1} 包含 Sunlight(Lux) 和 Power(mW) 欄位，正在進行訓練...")
        
       # 過濾掉異常的數據行來進行模型訓練
        filtered_df = df[~df['Sunlight(Lux)'].isin(anomalous_values)]
        
        # 檢查是否有可用數據
        if len(filtered_df) == 0:
            print(f"DataFrame L{i+1} 沒有可用數據，跳過...")
            continue

        # 特徵和標籤
        X = filtered_df[['Power(mW)']]
        y = filtered_df['Sunlight(Lux)']

        # 切分數據集為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 保留原本未縮放的 X_test 和 y_test，方便繪圖
        X_test_original = X_test.copy()
        y_test_original = y_test.copy()

        # 特徵縮放
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

        # 建立 SVR 模型
        svr_model = SVR(kernel='linear')
        #svr_model = SVR(kernel='poly', degree=2, C=1.0, epsilon=0.1)

        svr_model.fit(X_train, y_train)

        # 預測
        y_pred = svr_model.predict(X_test)

        # 找出異常值的數據行
        anomalous_df = df[df['Sunlight(Lux)'].isin(anomalous_values)]

        # 對異常數據行的 Power(mW) 進行特徵縮放並預測其對應的 Sunlight(Lux)
        if len(anomalous_df) > 0:
            X_anomalous = scaler_X.transform(anomalous_df[['Power(mW)']])
            y_anomalous_pred = svr_model.predict(X_anomalous)

            # 將預測的異常值還原到原始尺度
            y_anomalous_pred_rescaled = scaler_y.inverse_transform(y_anomalous_pred.reshape(-1, 1))
            print(f"DataFrame L{i+1} 的異常值預測結果為:\n {y_anomalous_pred_rescaled.flatten()}")

            # 更新原始 DataFrame 中的異常值
            df.loc[df['Sunlight(Lux)'].isin(anomalous_values), 'Sunlight(Lux)'] = y_anomalous_pred_rescaled.flatten()
        output_file_path = os.path.join(current_directory, f"L{i+1}_Processed.csv")
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"處理後的 DataFrame L{i+1} 已儲存為 {output_file_path}")
        # 繪製預測結果
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test_original, y_test_original, color='b', label='Actual Sunlight (Lux)')
        plt.scatter(X_test_original, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='r', label='Predicted Power (mW)')
        #plt.scatter(df['Power(mW)'], df['Sunlight(Lux)'], color='b', label='Actual Sunlight (Lux)')

        # 添加標籤和標題
        plt.title(f'SVR Prediction for L{i+1}: Sunlight(Lux) vs Power(mW)')
        plt.xlabel('Power (mW)')
        plt.ylabel('Sunlight (Lux)')
        plt.legend()
        
        # 顯示圖表
        plt.show()
    else:
        print(f"DataFrame L{i+1} 缺少 Sunlight(Lux) 或 Power(mW) 欄位")
