import numpy as np
import pandas as pd
import os
from datetime import date, timedelta

def convert_datetime_to_serial(row):
    """將 DateTime 欄位轉換為序號（serial）格式"""
    try:
        dt = pd.to_datetime(row['DateTime'])
        location = str(row['LocationCode']).zfill(2)  # 將地點編碼補零至兩位數
        return f"{dt.year}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}{location}"
    except Exception as e:
        print(f"轉換 DateTime 時發生錯誤: {str(e)}")
        return None

from datetime import date, timedelta
import numpy as np

from datetime import date, timedelta
import numpy as np

def match_previous_day_power(upload_df, dataframes, max_search_days=7):
    result_df = upload_df.copy()
    
    # 確保 '序號' 是字串，並處理可能的小數點
    result_df['序號'] = result_df['序號'].apply(lambda x: str(x).split('.')[0])
    
    # 初始化 '答案' 欄位
    if '答案' not in result_df.columns:
        result_df['答案'] = np.nan
    
    for index, row in result_df.iterrows():
        try:
            original_serial = row['序號']
            
            # 確保序號格式長度正確
            if len(original_serial) != 14:
                print(f"警告: 無效的序號格式 {original_serial}")
                continue
                
            # 從序號中提取年份、月份、日期、時間、地點代碼
            year = int(original_serial[:4])
            month = int(original_serial[4:6])
            day = int(original_serial[6:8])
            time = original_serial[8:14]
            location = int(original_serial[12:])  # 最後兩位為地點代碼
            
            # 找到對應的資料框
            if 0 <= location-1 < len(dataframes):
                target_df = dataframes[location-1]
                
                if target_df.empty:
                    print(f"地點 {location} 無可用數據")
                    continue
                
                # 確保 Serial 是字串型別
                target_df['Serial'] = target_df['Serial'].astype(str)
                
                # 搜尋策略
                found_previous = False
                found_next = False
                previous_power = None
                next_power = None

                # 定義搜尋順序：先找前面的最近一天，後找後面的最近一天
                previous_order = list(range(-1, -max_search_days - 1, -1))  # 前面天數
                next_order = list(range(1, max_search_days + 1))           # 後面天數

                # 搜尋前面的最近一天
                for days_offset in previous_order:
                    check_date = date(year, month, day) + timedelta(days=days_offset)
                    check_serial = f"{check_date.year:04d}{check_date.month:02d}{check_date.day:02d}{time}"
                    
                    matching_row = target_df[target_df['Serial'] == check_serial]
                    
                    if not matching_row.empty:
                        previous_power = round(matching_row['Power(mW)'].values[0], 2)
                        found_previous = True
                        break  # 找到最近一天的數據，停止搜尋

                # 搜尋後面的最近一天
                for days_offset in next_order:
                    check_date = date(year, month, day) + timedelta(days=days_offset)
                    check_serial = f"{check_date.year:04d}{check_date.month:02d}{check_date.day:02d}{time}"
                    
                    matching_row = target_df[target_df['Serial'] == check_serial]
                    
                    if not matching_row.empty:
                        next_power = round(matching_row['Power(mW)'].values[0], 2)
                        found_next = True
                        break  # 找到最近一天的數據，停止搜尋

                # 計算答案
                if found_previous and found_next:
                    # 如果同時找到前後的最近數據，取平均值
                    result_df.at[index, '答案'] = round((previous_power + next_power) / 2, 2)
                elif found_previous:
                    # 如果只找到前面的數據
                    result_df.at[index, '答案'] = previous_power
                elif found_next:
                    # 如果只找到後面的數據
                    result_df.at[index, '答案'] = next_power
                else:
                    # 如果前後都找不到匹配數據
                    print(f"找不到 {max_search_days} 天內的匹配數據: {original_serial}")

            else:
                print(f"警告: 無效的地點編碼 {location}")
                
        except Exception as e:
            print(f"處理第 {index} 行資料時發生錯誤: {str(e)}")
            print(f"原始序號: {row['序號']}")
            continue
    
    return result_df

# 資料載入部分
dataframes = []
current_directory = os.getcwd()
upload_df = pd.read_csv('36_TestSet_SubmissionTemplate/upload(no answer).csv')

# 載入 1 至 17 位置的原始資料
for loc in range(1, 17+1):  # 1 到 17
    file_name = f"ExampleTrainData(AVG)/AvgDATA_{loc:02d}.csv"
    file_path = os.path.join(current_directory, file_name)
    df = pd.read_csv(file_path)

    # 只保留 Serial 和 Power(mW) 欄位
    df = df[['Serial', 'Power(mW)']]
    
    dataframes.append(df)

# 載入額外的訓練資料
additional_files = {
    2: "L2_Train_2.csv",
    4: "L4_Train_2.csv",
    7: "L7_Train_2.csv",
    8: "L8_Train_2.csv",
    9: "L9_Train_2.csv",
    10: "L10_Train_2.csv",
    12: "L12_Train_2.csv"
}

# 處理額外的訓練資料
additional_folder = "36_TrainingData_Additional_V2"
for location, filename in additional_files.items():
    file_path = os.path.join(current_directory, additional_folder, filename)
    if os.path.exists(file_path):
        try:
            # 讀取額外的訓練資料
            additional_df = pd.read_csv(file_path)
            
            # 將 DateTime 轉換為序號格式
            additional_df['Serial'] = additional_df.apply(convert_datetime_to_serial, axis=1)
            
            # 確保 Serial 和 Power(mW) 欄位的數據類型正確
            additional_df['Serial'] = additional_df['Serial'].astype(str)
            additional_df['Power(mW)'] = pd.to_numeric(additional_df['Power(mW)'], errors='coerce')
            
            # 僅保留 Serial 和 Power(mW) 欄位
            additional_df = additional_df[['Serial', 'Power(mW)']]
            
            # 與現有的資料表合併
            if not dataframes[location-1].empty:
                # 確保原始資料框的欄位類型正確
                dataframes[location-1]['Serial'] = dataframes[location-1]['Serial'].astype(str)
                dataframes[location-1]['Power(mW)'] = pd.to_numeric(dataframes[location-1]['Power(mW)'], errors='coerce')
                
                # 合併資料並移除重複的序號
                dataframes[location-1] = pd.concat([dataframes[location-1], additional_df], ignore_index=True)
                dataframes[location-1] = dataframes[location-1].drop_duplicates(subset=['Serial'], keep='first')
                
                # 根據 Serial 排序
                dataframes[location-1] = dataframes[location-1].sort_values('Serial').reset_index(drop=True)
            else:
                # 如果原始資料不存在，直接使用額外資料
                dataframes[location-1] = additional_df
            
            print(f"成功載入地點 {location} 的額外資料")
            print(f"資料筆數: {len(dataframes[location-1])}")
            
        except Exception as e:
            print(f"載入地點 {location} 的額外資料時發生錯誤: {str(e)}")

# 匹配並更新答案欄位
result_df = match_previous_day_power(upload_df, dataframes)
result_df.to_csv('upload.csv', index=False)