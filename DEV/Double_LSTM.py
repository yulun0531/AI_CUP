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

#%% Config setting & Load Training Data
#設定LSTM往前看的筆數和預測筆數
LookBackNum = 12 #LSTM往前看的筆數
ForecastNum = 48 #預測筆數
LocationCount = 17
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.6f}'.format

current_directory = os.getcwd()
dataframes = []

for loc in range(1, LocationCount+1):  # 1 到 17
    file_name = f"ExampleTrainData(AVG)/AvgDATA_{loc:02d}.csv"
    file_path = os.path.join(current_directory, file_name)  # 合併路徑與檔名
    df = pd.read_csv(file_path)  # 讀取 CSV 檔案
    try:
      additional_data = pd.read_csv(os.getcwd()+'/processed_csv/L'+ str(loc) +'_Train_2'+'.csv', encoding='utf-8')
      # 如果檔案成功讀取，可以進行合併
      df = pd.concat([df, additional_data], axis=0, ignore_index=True)
      print("兩個檔案已經成功合併")
    except FileNotFoundError:
      # 如果第二個檔案不存在，則只使用第一個檔案
      df = df
      print("第二個檔案未找到，僅使用第一個檔案")
    dataframes.append(df)  # 加入列表

# 創建模型存放檔案
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_folder = os.path.join(current_directory, "models", f"{timestamp}")
os.makedirs(model_folder)
#%% Model Training location by location
for loc, df in enumerate(dataframes):
  # drop useless column
  # df = df.drop(columns=['WindSpeed(m/s)'])
  
  # =================================資料預處理Begin=================================
  '''1. 時間格式轉換'''

  df['Serial'] = df['Serial'].astype(str)

  # 提取 DateTime 相關資訊
  df['year'] = df['Serial'].str[:4].astype(int)
  df['month'] = df['Serial'].str[4:6].astype(int)
  df['day'] = df['Serial'].str[6:8].astype(int)
  df['hour'] = df['Serial'].str[8:10].astype(int)
  df['minute'] = df['Serial'].str[10:12].astype(int)
  df['LocationCode'] = df['Serial'].str[12:].astype(int)  # 裝置 ID

  # 建立 DateTime 欄位
  df['DateTime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
  df['days_in_month'] = df['DateTime'].dt.days_in_month
  date_time_values = df['DateTime'].copy()  #先記下完整的時間，plot圖的時候用

  # 拆分 DateTime 成多個特徵
  df['month'] = df['DateTime'].dt.month
  df['day'] = df['DateTime'].dt.day
  df['hour'] = df['DateTime'].dt.hour


  # 進行正弦和餘弦轉換
  df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
  df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
  df['day_sin'] = np.sin(2 * np.pi * df['day'] / df['days_in_month'])
  df['day_cos'] = np.cos(2 * np.pi * df['day'] / df['days_in_month'])
  df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
  df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

  df.loc[:, 'is_continuous'] = df['DateTime'].diff().dt.total_seconds().fillna(60) <= 700

  df = df.drop(columns=['year','month','minute','day','hour','days_in_month'])
  df_datatime=df.pop('DateTime')
  # print("Raw df attributes: ")
  # print(df.columns)

  # =================================資料預處理End=================================

  #============================建置&訓練Feature「LSTM模型」============================
  # 進行訓練集與測試集的劃分 (前 80% 為訓練集，後 20% 為測試集)
  # TODO：抽的方式要另外做，感覺要隨機抽一天，而非拿最後幾天 
  #     ：因為最後幾天會是連續的日期，前面並沒有那個Date區間的資料

  # 自動抓取所有欄位並排除不需要的欄位
# 自動抓取所有欄位並排除不需要的欄位
  exclude_columns = ['Serial','Power(mW)','LocationCode']  # 這裡列出你不需要的欄位
  feature_column = [col for col in df.columns if col not in exclude_columns]
  #feature_column = ['Pressure(hpa)','Temperature(°C)','Humidity(%)','Sunlight(Lux)', 'WindSpeed(m/s)']
  FeatLSTM_df = df[feature_column]

  # Train:Test = 8:2
  train_size = int(len(FeatLSTM_df) * 0.8)
  train_data = FeatLSTM_df[:train_size]
  is_continuous_train = train_data.pop('is_continuous')

  test_data = FeatLSTM_df[train_size:]
  is_continuous_test = test_data.pop('is_continuous')


  # 正規化(MinMaxScaler 縮放)
  Feature_MinMaxModel = MinMaxScaler(feature_range=(0, 1))
  train_scaled = Feature_MinMaxModel.fit_transform(train_data)
  test_scaled = Feature_MinMaxModel.transform(test_data)

  # LSTM 模型需要 3D 輸入數據
  X_train = []
  y_train = []
  
  # Input: 5 cols -> Output: 5 cols
  for i in range(LookBackNum, len(train_scaled)):
      if is_continuous_train.iloc[i-LookBackNum:i].all():
        X_train.append(train_scaled[i-LookBackNum:i, :])
        y_train.append(train_scaled[i, :]) 
  X_train, y_train = np.array(X_train), np.array(y_train)

  # 調整輸入數據格式為 LSTM 所需的 3D 格式 (sample數, 時間步, 特徵數)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
  n_feature = X_train.shape[2]  # 特徵數(Input)
  n_prediction = y_train.shape[1] # 預測數(Output)

  #建置LSTM模型
  regressor = Sequential ()

  regressor.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

  regressor.add(LSTM(units = 64))

  regressor.add(Dropout(0.2))

  regressor.add(Dense(units = n_prediction))

  regressor.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])
  regressor.fit(X_train, y_train, epochs = 100, batch_size = 20)

  #保存模型
  p = os.path.join(model_folder, f'FeatureLSTM_{loc+1:02d}.h5')
  regressor.save(p)
  print(f'FeatureLSTM_{loc+1:02d} saved')

  # 保存 Feature MinMaxScaler
  scaler_path = os.path.join(model_folder, f'Feature_MinMaxScaler_{loc+1:02d}.pkl')
  joblib.dump(Feature_MinMaxModel, scaler_path)
  print(f'Feature MinMaxScaler_{loc+1:02d} saved')
  '''
  
  # 準備輸入數據 (同樣需要按 LookBackNum 組合為 3D 格式)
  X_test = []
  for i in range(LookBackNum, len(test_scaled)):
      if is_continuous_test.iloc[i-n_step:i].all():
        X_test.append(test_scaled[i-LookBackNum:i, :])
  X_test = np.array(X_test)

  # 進行預測
  predictions_scaled = regressor.predict(X_test)

  # 將預測結果反向縮放回原始數據範圍
  predictions = Feature_MinMaxModel.inverse_transform(predictions_scaled)

  # 輸出預測結果與原始數據的對比 (只顯示第一筆)
  print("First Prediction:")
  print(predictions[:10])  # 預測結果的第一筆

  print("First Original Data (Ground Truth):")
  print(test_data.iloc[LookBackNum:LookBackNum+10])  # 原始測試數據對應的第一筆
  '''
  # 訓練Power_LSTM
  # TODO: 前面12步的Power值若要當作特徵要考量縮放的問題(skip)
  # 預測欄位

  Power_df = df[['Power(mW)']+feature_column]


  Power_df.loc[:, 'is_continuous'] = df_datatime.diff().dt.total_seconds().fillna(60) <= 700
  Power_index = Power_df.columns.get_loc('Power(mW)')
  # Train:Test = 8:2
  train_size = int(len(df) * 0.8)
  train_data = Power_df[:train_size]
  is_continuous_train = train_data.pop('is_continuous')
  test_data = Power_df[train_size:]
  is_continuous_test = test_data.pop('is_continuous')

  
  # 正規化(MinMaxScaler 縮放)
  Power_MinMaxModel = MinMaxScaler(feature_range=(0, 1))
  train_scaled = Power_MinMaxModel.fit_transform(train_data)
  test_scaled = Power_MinMaxModel.transform(test_data)

  # LSTM 模型需要 3D 輸入數據
  X_train = []
  y_train = []
  for i in range(LookBackNum, len(train_scaled)):
      if is_continuous_train.iloc[i-LookBackNum:i].all():
        X_train.append(train_scaled[i-LookBackNum:i, Power_index:])
        y_train.append(train_scaled[i, Power_index])
  X_train, y_train = np.array(X_train), np.array(y_train)
  # 調整輸入數據格式為 LSTM 所需的 3D 格式
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
  n_feature = X_train.shape[2]  # 這是特徵數
  

  # 建構並優化 LSTM 模型
  model = Sequential()

  # 調整 LSTM 層單元數及 Dropout
  model.add(LSTM(units = 64, activation='relu', return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

  model.add(LSTM(units = 32))
  model.add(Dropout(0.2))

  model.add(Dense(units=1))

  model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])
  # 進行模型訓練
  early_stopping = EarlyStopping(monitor='val_loss', patience=20)

  model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    
  # 對測試數據進行處理
  X_test = []
  y_test=[]
  original_y_test=[]
  consolidation_test_scaled=[]
  date_time=[]

  for i in range(LookBackNum, len(test_scaled)):
    if is_continuous_test.iloc[i-LookBackNum:i].all():
      X_test.append(test_scaled[i-LookBackNum:i, Power_index:])  # 將過去LookBackNum筆的所有特徵添加到 X_test
      y_test.append(test_scaled[i, Power_index])  # 目標是Power
      original_y_test.append(test_data.iloc[i]['Power(mW)'])
      part1 = test_scaled[i, :Power_index]  # 不加擴展維度，保持一維
      part2 = test_scaled[i, Power_index + 1:]  # 同樣保持一維

      consolidation_test_scaled.append(np.concatenate([part1, part2], axis=0))  # 按列合併
      date_time.append(date_time_values[train_size+i])

  X_test,y_test,consolidation_test_scaled = np.array(X_test), np.array(y_test) ,np.array(consolidation_test_scaled)

  # 調整輸入數據格式為 LSTM 所需的 3D 格式
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

  # 進行預測
  y_pred = model.predict(X_test)

  #保存模型
  p = os.path.join(model_folder, f'PowerLSTM_{loc+1:02d}.h5')
  model.save(p)
  print(f'PowerLSTM_{loc:02d} saved')
  
  # 保存 Power MinMaxScaler
  scaler_path = os.path.join(model_folder, f'Power_MinMaxScaler_{loc+1:02d}.pkl')
  joblib.dump(Power_MinMaxModel, scaler_path)
  print(f'Power MinMaxScaler_{loc+1:02d} saved')
  
  # 反縮放
  y_pred_rescaled = Power_MinMaxModel.inverse_transform(np.concatenate((y_pred, consolidation_test_scaled[:, :]), axis=1))[:, 0]
  # y_pred_rescaled = scaler.inverse_transform(np.concatenate((y_pred, test_scaled[LookBackNum:, :Power_index], test_scaled[LookBackNum:, Power_index+1:]), axis=1))[:, 0]
  # print(test_data[LookBackNum:]['Power(mW)'])

  # 計算測試集的均方誤差 (MSE)
  total_score = np.mean(np.abs(original_y_test - y_pred_rescaled))
  
  print(f"測試集的均方誤差: {total_score}")
  # 繪製預測結果
  plt.figure(figsize=(14, 5))
  plt.plot(date_time[:],original_y_test, label='actually', color='blue')
  plt.plot(date_time[:],y_pred_rescaled, label='predict', color='red')
  plt.title('actually vs predict')
  plt.xlabel('time')
  plt.ylabel('power')
  plt.legend()
  plt.show()
  

#%%
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 顯示所有訊息（默認值）
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 僅顯示警告和錯誤
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 僅顯示錯誤訊息
#============================預測數據============================
#載入模型

fetureModels, powerModels = [], []
feature_scalers, power_scalers = [], []
model_folder = os.path.join(current_directory, "models", f"{timestamp}")
for loc in range(1, LocationCount+1):
  fetureModels.append(load_model(os.path.join(model_folder, f'FeatureLSTM_{loc:02d}.h5')))
  powerModels.append(load_model(os.path.join(model_folder,f'PowerLSTM_{loc:02d}.h5')))
  
  # 載入對應的 MinMaxScaler
  feature_scalers.append(joblib.load(os.path.join(model_folder, f'Feature_MinMaxScaler_{loc:02d}.pkl')))
  power_scalers.append(joblib.load(os.path.join(model_folder, f'Power_MinMaxScaler_{loc:02d}.pkl')))

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
feature_column = [col for col in df.columns if col not in exclude_columns]

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
  fetureLSTM = fetureModels[loc_index]
  powerLSTM = powerModels[loc_index]
  Feature_MinMaxModel = feature_scalers[loc_index]
  Power_MinMaxModel = power_scalers[loc_index]  
  
  DataName = os.getcwd()+'/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_'+ strLocationCode +'.csv'
  SourceData = pd.read_csv(DataName, encoding='utf-8')
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
  SourceData['Serial'] = SourceData['Serial'].astype(str)

  # 提取 DateTime 相關資訊
  SourceData['year'] = SourceData['Serial'].str[:4].astype(int)
  SourceData['month'] = SourceData['Serial'].str[4:6].astype(int)
  SourceData['day'] = SourceData['Serial'].str[6:8].astype(int)
  SourceData['hour'] = SourceData['Serial'].str[8:10].astype(int)
  SourceData['minute'] = SourceData['Serial'].str[10:12].astype(int)
  SourceData['LocationCode'] = SourceData['Serial'].str[12:].astype(int)  # 裝置 ID

  # 建立 DateTime 欄位
  SourceData['DateTime'] = pd.to_datetime(SourceData[['year', 'month', 'day', 'hour', 'minute']])
  SourceData['days_in_month'] = SourceData['DateTime'].dt.days_in_month



  # 進行正弦和餘弦轉換
  SourceData['month_sin'] = np.sin(2 * np.pi * SourceData['month'] / 12)
  SourceData['month_cos'] = np.cos(2 * np.pi * SourceData['month'] / 12)
  SourceData['day_sin'] = np.sin(2 * np.pi * SourceData['day'] / SourceData['days_in_month'])
  SourceData['day_cos'] = np.cos(2 * np.pi * SourceData['day'] / SourceData['days_in_month'])
  SourceData['hour_sin'] = np.sin(2 * np.pi * SourceData['hour'] / 24)
  SourceData['hour_cos'] = np.cos(2 * np.pi * SourceData['hour'] / 24)

  SourceData = SourceData.drop(columns=['year','month','minute','day','hour','days_in_month','DateTime'])
  ReferTitle = SourceData[['Serial']].values
  ReferData = SourceData[[col for col in SourceData.columns if col not in exclude_columns]].values
  inputs = []#重置存放參考資料
  inputs_for_power = []#重置存放參考資料

  #找到相同的一天，把12個資料都加進inputs
  for DaysCount in range(len(ReferTitle)):
    if(str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]):
      # 將數據轉換為 DataFrame，並加入正確的欄位名稱
      # Fix: UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names warnings.warn(
      TempData = pd.DataFrame(ReferData[DaysCount].reshape(1,-1), 
                            columns=feature_column)
      TempData_for_power= pd.DataFrame(ReferData[DaysCount].reshape(1,-1), 
                            columns=["Power(mW)"]+feature_column)
      TempDatas = Feature_MinMaxModel.transform(TempData)
      TempDatas_for_power = Power_MinMaxModel.transform(TempData)
      inputs.append(TempDatas)
      inputs_for_power.append(TempDatas_for_power)
  print(EXquestion[count])
  # Error handling
  if len(inputs) != 12:
    print("當日資料不足12筆：", EXquestion[count])

  #用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
  for i in range(ForecastNum) :
    #將新的預測值加入參考資料(用自己的預測值往前看)
    if i > 0 :
      inputs.append(PredictOutput[i-1].reshape(1,11))
      inputs_for_power.append(np.concatenate(PredictOutput[i-1], PredictOutput[i-1]).reshape(1,12))

    #切出新的參考資料12筆(往前看12筆)
    X_test = []
    X_test.append(inputs[-LookBackNum+i:])
    X_test_for_power = []
    X_test_for_power.append(inputs_for_power[-LookBackNum+i:])

    #Reshaping
    NewTest,NewTest_for_power = np.array(X_test),np.array(X_test_for_power)

    NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], NewTest.shape[3]))
    NewTest_for_power = np.reshape(NewTest_for_power, (NewTest_for_power.shape[0], NewTest_for_power.shape[1], NewTest_for_power.shape[3]))

    predicted_feature = fetureLSTM.predict(NewTest, verbose=0) #吃前12筆預測第13筆特徵
    PredictOutput.append(predicted_feature) # 預測下一筆的特徵

    predicted_power = powerLSTM.predict(NewTest_for_power, verbose=0) #吃前12筆預測第13筆Power

    predicted_power = Power_MinMaxModel.inverse_transform(
        np.concatenate((predicted_power, predicted_feature), axis=1)
    )[:, Power_index]
    print(predicted_power)

    predicted_power = np.round(predicted_power,2).flatten()[0]
    
    PredictPower.append(predicted_power)
    df2.loc[len(df2)] = {
      '序號': EXquestion[count+i][0],
      '答案': predicted_power
    }
  #每次預測都要預測48個，因此加48個會切到下一天
  #0~47,48~95,96~143...
  count += 48

#寫預測結果寫成新的CSV檔案
# 將 DataFrame 寫入 CSV 檔案
df2.to_csv('output.csv', index=False) 
print('Output CSV File Saved')

# %%
# 問題1. MinMaxScalar要儲存起來在預測的時候load，不然不同地區在反縮放時會有問題
# 已解決

