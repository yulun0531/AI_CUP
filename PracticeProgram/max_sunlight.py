import pandas as pd
import numpy as np
import os

# 準備資料
dataframes = []
current_directory = os.getcwd()
anomalous_value = 117758.2

# 讀取所有檔案
for i in range(1, 18):  # 假設檔案從 L1_Train.csv 到 L17_Train.csv
    file_name = f"L{i}_Train.csv"
    file_path = os.path.join(current_directory, file_name)
    df = pd.read_csv(file_path)
    
    # 將 DateTime 轉換為 pandas datetime 格式（假設 DateTime 欄位名稱為 'DateTime'）
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 新增一個欄位，標記是否 Sunlight(Lux) 為 117758.2
    df['IsAnomalous'] = (df['Sunlight(Lux)'] == anomalous_value)
    
    # 新增保持時間欄位，初始為 0
    df['HoldDuration'] = 0
    df['SinHoldDuration'] = 0  # 用來存放轉換後的值
    
    hold_start = None  # 初始化持續的起始時間
    max_hold_duration = 0  # 每次異常區間的最大持續時間，初始化為 0
    
    # 計算每一筆資料的持續時間並更新最大持續時間
    for j in range(len(df)):
        if df.loc[j, 'IsAnomalous']:  # 當值為異常值
            if hold_start is None:  # 開始計算
                hold_start = df.loc[j, 'DateTime']
                max_hold_duration = 0  # 異常區間開始，重置最大持續時間

            # 計算當前行的持續時間
            current_time = df.loc[j, 'DateTime']
            hold_duration = (current_time - hold_start).total_seconds()
            df.loc[j, 'HoldDuration'] = hold_duration

            # 更新最大持續時間（對於這個異常區間）
            max_hold_duration = max(max_hold_duration, hold_duration)
        else:  # 當異常狀態結束
            if hold_start is not None:  # 結束異常區間時
                # 異常區間結束後，根據當前區間最大持續時間計算 omega
                omega = 2 * np.pi / max_hold_duration if max_hold_duration > 0 else 0

                # 使用正弦轉換持續時間，並將結果保存在 SinHoldDuration 欄位
                df.loc[(df['IsAnomalous']) & (df['HoldDuration'] <= max_hold_duration) & (df['DateTime'] >= hold_start), 'SinHoldDuration'] = np.sin(omega * df['HoldDuration'])

            hold_start = None  # 重置起始時間
            max_hold_duration = 0  # 重置最大持續時間

    # 將處理後的 DataFrame 加回列表
    dataframes.append(df)

# 將結果輸出回檔案或檢查
for i, df in enumerate(dataframes):
    output_file_name = f"L{i+1}_Train_Processed.csv"
    output_path = os.path.join(current_directory, output_file_name)
    df.to_csv(output_path, index=False)
    print(f"檔案 {output_file_name} 已處理完成")
