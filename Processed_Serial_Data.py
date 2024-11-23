import os
import pandas as pd
from datetime import datetime

# 設定資料夾路徑
folder_path = os.getcwd() + '\\36_TrainingData_Additional_V2'
output_folder = "processed_csv"
os.makedirs(output_folder, exist_ok=True)

# 遍歷資料夾中的所有 CSV 檔案
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv") and file_name.startswith("L"):  # 篩選符合條件的檔案
        file_path = os.path.join(folder_path, file_name)

        # 讀取 CSV 檔案
        df = pd.read_csv(file_path)

        # 清理多餘空格
        df['DateTime'] = df['DateTime'].str.strip()

        # 確認必要欄位存在
        if "LocationCode" in df.columns and "DateTime" in df.columns:
            # 移除 "RoundedTime" 欄位
            if "RoundedTime" in df.columns:
                df = df.drop(columns=["RoundedTime"])

            def process_data(group):
                """針對每個 10 分鐘區間計算平均值，並生成 Serial"""
                try:
                    # 計算平均值，忽略非數值型欄位
                    avg_data = group.iloc[:, 2:].mean()  # 計算數值欄位的平均值，忽略前兩欄（DateTime, LocationCode）

                    # 取得第一筆資料的時間與位置代碼
                    original_time = group['DateTime'].iloc[0]
                    location_code = f"{int(group['LocationCode'].iloc[0]):02}"  # 確保是兩位數
                    
                    # 組合 Serial (以原始時間的年月日時分+LocationCode)
                    # 這裡直接使用 Timestamp 的 year, month, day, hour, minute 屬性來組合
                    serial = f"{original_time.year:04}{original_time.month:02}{original_time.day:02}{original_time.hour:02}{original_time.minute:02}{location_code}"

                    # 返回 Serial 和平均值
                    avg_data['Serial'] = serial
                    return avg_data
                except Exception as e:
                    print(f"處理錯誤: {e}")
                    return None

            # 將 DateTime 欄位轉為 pandas 的 Timestamp 類型
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')

            # 按照 10 分鐘區間進行分組
            df['RoundedTime'] = df['DateTime'].dt.floor('10T')  # 按 10 分鐘向下取整

            # 按照 RoundedTime 分組並處理
            result = df.groupby('RoundedTime').apply(process_data)

            # 移除 "RoundedTime" 欄位後儲存處理結果
            result = result.drop(columns=["RoundedTime"], errors='ignore')

            # 確保 Serial 欄位放在第一欄
            columns_order = ['Serial'] + [col for col in result.columns if col != 'Serial']
            result = result[columns_order]

            # 儲存處理後的檔案
            output_path = os.path.join(output_folder, file_name)
            result.to_csv(output_path, index=False, encoding='utf-8')
            print(f"處理完成並儲存: {output_path}")
        else:
            print(f"檔案 {file_name} 缺少必要欄位，跳過處理。")
