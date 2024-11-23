import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

class ConfigSettings:
    def __init__(self):
        self.LOOK_BACK_NUM = 12  # LSTM往前看的筆數
        self.FORECAST_NUM = 48   # 預測筆數
        self.LOCATION_COUNT = 17
        self.TRAIN_TEST_SPLIT = 0.8
        
        # 設定數值顯示格式
        np.set_printoptions(suppress=True)
        pd.options.display.float_format = '{:.6f}'.format

class DataPreprocessor:
    @staticmethod
    def load_and_merge_data(current_directory, location):
        """載入並合併训练数据"""
        # 讀取主要訓練資料
        main_file = f"ExampleTrainData(AVG)/AvgDATA_{location:02d}.csv"
        main_df = pd.read_csv(os.path.join(current_directory, main_file))
        
        try:
            # 嘗試讀取額外訓練資料
            additional_file = f'processed_csv/L{location}_Train_2.csv'
            additional_df = pd.read_csv(os.path.join(current_directory, additional_file), encoding='utf-8')
            return pd.concat([main_df, additional_df], axis=0, ignore_index=True)
        except FileNotFoundError:
            print(f"Location {location}: Additional file not found, using main file only")
            return main_df
    
    @staticmethod
    def process_datetime(df):
        """處理時間相關特徵"""
        df['Serial'] = df['Serial'].astype(str)
        
        # 提取時間信息
        df['year'] = df['Serial'].str[:4].astype(int)
        df['month'] = df['Serial'].str[4:6].astype(int)
        df['day'] = df['Serial'].str[6:8].astype(int)
        df['hour'] = df['Serial'].str[8:10].astype(int)
        df['minute'] = df['Serial'].str[10:12].astype(int)
        df['LocationCode'] = df['Serial'].str[12:].astype(int)
        
        # 創建DateTime欄位
        df['DateTime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        df['days_in_month'] = df['DateTime'].dt.days_in_month
        
        # 循環特徵轉換
        df['month_sin'] = np.sin(2 * np.pi * df['DateTime'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['DateTime'].dt.month / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['DateTime'].dt.day / df['days_in_month'])
        df['day_cos'] = np.cos(2 * np.pi * df['DateTime'].dt.day / df['days_in_month'])
        df['hour_sin'] = np.sin(2 * np.pi * df['DateTime'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['DateTime'].dt.hour / 24)
        
        # 檢查時間連續性
        df['is_continuous'] = df['DateTime'].diff().dt.total_seconds().fillna(60) <= 700
        
        # 清理不需要的列
        df = df.drop(columns=['year', 'month', 'minute', 'day', 'hour', 'days_in_month'])
        df_datetime = df.pop('DateTime')
        
        return df, df_datetime

class LSTMModelBuilder:
    def __init__(self, look_back_num):
        self.look_back_num = look_back_num
        
    def prepare_data(self, df, train_size):
        """準備LSTM訓練數據"""
        # 排除不需要的特徵
        exclude_columns = ['Serial', 'Power(mW)', 'LocationCode']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        feature_df = df[feature_columns]
        
        # 分割訓練集和測試集
        train_data = feature_df[:train_size]
        is_continuous_train = train_data.pop('is_continuous')
        test_data = feature_df[train_size:]
        is_continuous_test = test_data.pop('is_continuous')
        
        return train_data, test_data, is_continuous_train, is_continuous_test
    
    def create_sequences(self, scaled_data, is_continuous):
        """創建LSTM序列數據"""
        X, y = [], []
        for i in range(self.look_back_num, len(scaled_data)):
            if is_continuous.iloc[i-self.look_back_num:i].all():
                X.append(scaled_data[i-self.look_back_num:i, :])
                y.append(scaled_data[i, :])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, output_shape):
        """建立LSTM模型"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            LSTM(64),
            Dropout(0.2),
            Dense(output_shape)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])
        return model

def main():
    # 初始化配置
    config = ConfigSettings()
    current_directory = os.getcwd()
    
    # 創建模型存放目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(current_directory, "models", timestamp)
    os.makedirs(model_folder)
    
    # 對每個位置進行訓練
    for loc in range(1, config.LOCATION_COUNT + 1):
        print(f"\nProcessing Location {loc}")
        
        # 載入數據
        df = DataPreprocessor.load_and_merge_data(current_directory, loc)
        
        # 處理時間特徵
        df, df_datetime = DataPreprocessor.process_datetime(df)
        
        # 準備LSTM模型
        model_builder = LSTMModelBuilder(config.LOOK_BACK_NUM)
        train_size = int(len(df) * config.TRAIN_TEST_SPLIT)
        
        # 準備訓練數據
        train_data, test_data, is_continuous_train, is_continuous_test = model_builder.prepare_data(df, train_size)
        
        # 特徵縮放
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # 創建序列數據
        X_train, y_train = model_builder.create_sequences(train_scaled, is_continuous_train)
        
        # 建立並訓練模型
        model = model_builder.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=y_train.shape[1]
        )
        
        model.fit(X_train, y_train, epochs=100, batch_size=20)
        
        # 保存模型和縮放器
        model.save(os.path.join(model_folder, f'FeatureLSTM_{loc:02d}.h5'))
        joblib.dump(scaler, os.path.join(model_folder, f'Feature_MinMaxScaler_{loc:02d}.pkl'))
        
        print(f'Location {loc}: Model and scaler saved successfully')

if __name__ == "__main__":
    main()