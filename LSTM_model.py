import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 設定資料夾路徑
current_directory = os.getcwd()

# 使用列表存放讀取的資料
dataframes = []
n_step = 5
# 迴圈讀取每一個檔案
for i in range(1, 18):  # 1 到 17
    file_name = f"L{i}_Processed.csv"
    file_path = os.path.join(current_directory, file_name)  # 合併路徑與檔名
    df = pd.read_csv(file_path)  # 讀取 CSV 檔案
    dataframes.append(df)  # 加入列表
for i, df in enumerate(dataframes):
    # 明確指定要進行編碼的類別
    df = df.drop(columns=['WindSpeed(m/s)'])
    # 將 DateTime 列轉換為 datetime 格式
    df['DateTime'] = pd.to_datetime(df['DateTime'])  # 讓 Pandas 自動推斷格式
    df['days_in_month'] = df['DateTime'].dt.days_in_month

    # 拆分 DateTime 成多個特徵
    df['year'] = df['DateTime'].dt.year
    df['month'] = df['DateTime'].dt.month
    df['day'] = df['DateTime'].dt.day
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute
    df['second'] = df['DateTime'].dt.second
    df['weekday'] = df['DateTime'].dt.weekday  # 0=Monday, 6=Sunday

    # 進行正弦和餘弦轉換
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / df['days_in_month'])
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / df['days_in_month'])
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['second_sin'] = np.sin(2 * np.pi * df['second'] / 60)
    df['second_cos'] = np.cos(2 * np.pi * df['second'] / 60)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    date_time_values = df['DateTime'].copy()  # 使用 copy() 確保不會影響原始 DataFrame
    df['is_continuous'] = df['DateTime'].diff().dt.total_seconds().fillna(60) <= 120

    df = df.drop(columns=['month','day', 'hour','minute','second','weekday','days_in_month','DateTime'])
    # 將 'Power(mW)' 列刪除並保存到變數
    power_column = df.pop('Power(mW)')

    # 在第一列插入 'Power(mW)' 列
    df.insert(0, 'Power(mW)', power_column)
    Power_index = df.columns.get_loc('Power(mW)')
    categories = [[str(i) for i in range(1, 18)]]  # 生成字串 '1' 到 '17'
    # 初始化編碼器，並設置 `handle_unknown='ignore'`
    encoder = OneHotEncoder(categories=categories, handle_unknown='ignore', sparse=False)

    # 進行編碼
    encoded_features = encoder.fit_transform(df[['LocationCode']])

    # 將編碼後的特徵轉換為 DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['LocationCode']))

    # 合併編碼後的特徵到原始數據集
    df_encoded = pd.concat([df.drop(columns=['LocationCode']), encoded_df], axis=1)
    
     # 進行訓練集與測試集的劃分 (前 80% 為訓練集，後 20% 為測試集)
    train_size = int(len(df_encoded) * 0.8)
    
    # 前 80% 資料用作訓練集
    train_data = df_encoded[:train_size]
    is_continuous_train = train_data.pop('is_continuous')

    # 後 20% 資料用作測試集
    test_data = df_encoded[train_size:]
    is_continuous_test = test_data.pop('is_continuous')

    scaler = MinMaxScaler(feature_range=(0, 1))
    # 對訓練集和測試集進行 MinMaxScaler 縮放
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # LSTM 模型需要 3D 輸入數據
    X_train = []
    y_train = []
    print(is_continuous_train)
    for i in range(n_step, len(train_scaled)):
        if is_continuous_train.iloc[i-n_step:i].all():
            X_train.append(train_scaled[i-n_step:i, :])  # 將過去5天的所有特徵添加到 X_train
            y_train.append(train_scaled[i, Power_index])  # 目標是Power
    X_train, y_train = np.array(X_train), np.array(y_train)

    # 調整輸入數據格式為 LSTM 所需的 3D 格式
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    n_feature = X_train.shape[2]  # 這是特徵數
    
    # 建構並優化 LSTM 模型
    model = Sequential()

    # 調整 LSTM 層單元數及 Dropout
    model.add(LSTM(units=50, activation='relu',return_sequences=False, input_shape=(n_step, n_feature)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])

    # 加载之前保存的模型權重
    try:
        model.load_weights("new_model.h5")
        print("成功加载模型权重。")
    except:
        print("未找到保存的权重文件，开始从头训练模型。")

    # 進行模型訓練
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.2, callbacks=[early_stopping])
    model.save("new_model.h5")
    # 對測試數據進行處理
    X_test = []
    y_test=[]
    original_y_test=[]
    consolidation_test_scaled=[]
    date_time=[]
    for i in range(n_step, len(test_scaled)):
        if is_continuous_test.iloc[i-n_step:i].all():
            X_test.append(test_scaled[i-n_step:i, :])  # 將過去5天的所有特徵添加到 X_test
            y_test.append(train_scaled[i, Power_index])  # 目標是Power
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

    # 反縮放預測結果
    
    y_pred_rescaled = scaler.inverse_transform(np.concatenate((y_pred, consolidation_test_scaled[:, :]), axis=1))[:, 0]
    # 計算總分數
    total_score = np.mean(np.abs(original_y_test - y_pred_rescaled))

    print("總分數:", total_score)

    # 繪製預測結果
    plt.figure(figsize=(14, 5))
    plt.plot(date_time[:],original_y_test, label='actually', color='blue')
    plt.plot(date_time[:],y_pred_rescaled, label='predict', color='red')
    plt.title('actually vs predict')
    plt.xlabel('time')
    plt.ylabel('power')
    plt.legend()
    plt.show()
    