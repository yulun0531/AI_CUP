#%% Load packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import os
current_directory = os.getcwd()
exclude_columns = ['Serial','Power(mW)','LocationCode']  # 這裡列出你不需要的欄位

#載入測試資料
question_path = os.getcwd()+r'/36_TestSet_SubmissionTemplate/upload(no answer).csv'
#question_path = os.getcwd()+r'/ExampleTestData/upload.csv'
Upload = pd.read_csv(question_path, encoding='utf-8')
target = ['序號']
EXquestion = Upload[target].values

inputs = [] #存放參考資料
PredictOutput = [] #存放預測值(天氣參數)
PredictPower = [] #存放預測值(發電量)
# FeatureLSTM minMax欄位順序
#feature_column = ['Pressure(hpa)','Temperature(°C)','Humidity(%)','Sunlight(Lux)', 'WindSpeed(m/s)']

df2 = pd.DataFrame(columns=['序號', '答案'])

count = 0
while(count < len(EXquestion)):
  # print('count : ',count)
  LocationCode = int(EXquestion[count])
  strLocationCode = str(LocationCode)[-2:]
  strLocationCode_extra = strLocationCode.lstrip('0')
  if LocationCode < 10 :
    strLocationCode = '0'+LocationCode
  
  loc_index = int(strLocationCode) - 1
  
  DataName = os.getcwd()+'/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_'+ strLocationCode +'.csv'
  SourceData = pd.read_csv(DataName, encoding='utf-8')
  feature_column = [col for col in SourceData.columns if col not in exclude_columns]

  # 嘗試讀取第二個檔案
  print(strLocationCode)
  try:
      NewSourceData = pd.read_csv(os.getcwd()+'/processed_csv/L'+ strLocationCode_extra +'_Train_2'+'.csv', encoding='utf-8')
      # 如果檔案成功讀取，可以進行合併
      SourceData = pd.concat([SourceData, NewSourceData], axis=0, ignore_index=True)
      print("兩個檔案已經成功合併")
  except FileNotFoundError:
      # 如果第二個檔案不存在，則只使用第一個檔案
      SourceData = SourceData
      print("第二個檔案未找到，僅使用第一個檔案")
  ReferTitle = SourceData[['Serial']].values
  ReferData = SourceData[[col for col in SourceData.columns if col not in exclude_columns]].values

  for DaysCount in range(len(ReferTitle)):
    if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count] ))[:8] =="20240927"):
      # 將數據轉換為 DataFrame，並加入正確的欄位名稱
      # Fix: UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names warnings.warn(
        TempData = pd.DataFrame(ReferData[DaysCount].reshape(1,-1), 
                            columns=feature_column)
      
        inputs.append(TempData)
        print(TempData)
  count+=48
