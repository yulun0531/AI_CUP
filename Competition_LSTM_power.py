import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class PowerPredictor:
    def __init__(self, look_back_num):
        self.look_back_num = look_back_num
        
    def prepare_power_data(self, df, feature_column, df_datetime):
        """準備電力預測數據"""
        # 合併電力數據和特徵
        power_df = df[['Power(mW)'] + feature_column]
        
        # 添加時間連續性檢查
        power_df.loc[:, 'is_continuous'] = df_datetime.diff().dt.total_seconds().fillna(60) <= 700
        power_index = power_df.columns.get_loc('Power(mW)')
        
        return power_df, power_index
    
    def split_scale_data(self, power_df, train_size):
        """分割和縮放數據"""
        # 分割訓練集和測試集
        train_data = power_df[:train_size]
        is_continuous_train = train_data.pop('is_continuous')
        test_data = power_df[train_size:]
        is_continuous_test = test_data.pop('is_continuous')
        
        # 數據縮放
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        return train_scaled, test_scaled, test_data, is_continuous_train, is_continuous_test, scaler
    
    def create_power_sequences(self, scaled_data, is_continuous, power_index):
        """創建用於電力預測的序列數據"""
        X, y = [], []
        for i in range(self.look_back_num, len(scaled_data)):
            if is_continuous.iloc[i-self.look_back_num:i].all():
                X.append(scaled_data[i-self.look_back_num:i, power_index:])
                y.append(scaled_data[i, power_index])
        return np.array(X), np.array(y)
    
    def build_power_model(self, input_shape):
        """建立電力預測模型"""
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])
        return model
    
    def prepare_test_data(self, test_scaled, test_data, is_continuous_test, power_index, date_time_values, train_size):
        """準備測試數據"""
        X_test = []
        y_test = []
        original_y_test = []
        consolidation_test_scaled = []
        date_time = []
        
        for i in range(self.look_back_num, len(test_scaled)):
            if is_continuous_test.iloc[i-self.look_back_num:i].all():
                X_test.append(test_scaled[i-self.look_back_num:i, power_index:])
                y_test.append(test_scaled[i, power_index])
                original_y_test.append(test_data.iloc[i]['Power(mW)'])
                
                # 合併其他特徵
                part1 = test_scaled[i, :power_index]
                part2 = test_scaled[i, power_index + 1:]
                consolidation_test_scaled.append(np.concatenate([part1, part2]))
                
                date_time.append(date_time_values[train_size + i])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        consolidation_test_scaled = np.array(consolidation_test_scaled)
        
        return X_test, y_test, original_y_test, consolidation_test_scaled, date_time
    
    def evaluate_predictions(self, y_pred, original_y_test, date_time):
        """評估預測結果"""
        # 計算平均絕對誤差
        mae = np.mean(np.abs(original_y_test - y_pred))
        print(f"平均絕對誤差 (MAE): {mae:.2f}")
        
        # 繪製預測結果
        plt.figure(figsize=(14, 5))
        plt.plot(date_time, original_y_test, label='Actual', color='blue')
        plt.plot(date_time, y_pred, label='Predicted', color='red')
        plt.title('Actual vs Predicted Power Values')
        plt.xlabel('Time')
        plt.ylabel('Power (mW)')
        plt.legend()
        plt.show()
        
        return mae

def train_power_model(df, feature_column, df_datetime, model_folder, location, look_back_num):
    """訓練電力預測模型的主函數"""
    predictor = PowerPredictor(look_back_num)
    
    # 準備數據
    power_df, power_index = predictor.prepare_power_data(df, feature_column, df_datetime)
    train_size = int(len(df) * 0.8)
    
    # 分割和縮放數據
    train_scaled, test_scaled, test_data, is_continuous_train, is_continuous_test, scaler = \
        predictor.split_scale_data(power_df, train_size)
    
    # 創建序列數據
    X_train, y_train = predictor.create_power_sequences(train_scaled, is_continuous_train, power_index)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    
    # 建立和訓練模型
    model = predictor.build_power_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit(X_train, y_train, 
             epochs=100, 
             batch_size=32, 
             validation_split=0.2, 
             callbacks=[early_stopping])
    
    # 準備測試數據
    X_test, y_test, original_y_test, consolidation_test_scaled, date_time = \
        predictor.prepare_test_data(test_scaled, test_data, is_continuous_test, 
                                  power_index, df_datetime, train_size)
    
    # 進行預測
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    y_pred = model.predict(X_test)
    
    # 反縮放預測結果
    y_pred_rescaled = scaler.inverse_transform(
        np.concatenate((y_pred, consolidation_test_scaled), axis=1))[:, 0]
    
    # 評估預測結果
    mae = predictor.evaluate_predictions(y_pred_rescaled, original_y_test, date_time)
    
    # 保存模型和縮放器
    model.save(os.path.join(model_folder, f'PowerLSTM_{location:02d}.h5'))
    joblib.dump(scaler, os.path.join(model_folder, f'Power_MinMaxScaler_{location:02d}.pkl'))
    
    print(f'Location {location}: Power prediction model and scaler saved successfully')
    
    return mae