import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

class PowerPredictionModel:
    def __init__(self, timestamp, location_count, current_directory):
        """
        初始化預測模型
        
        Args:
            timestamp: 模型時間戳記
            location_count: 位置數量
            current_directory: 當前工作目錄
        """
        self.location_count = location_count
        self.model_folder = os.path.join(current_directory, "models", f"{timestamp}")
        self.look_back_num = 12  # 往前看的資料筆數
        self.forecast_num = 48   # 預測未來小時數
        
        # 初始化模型和縮放器
        self._load_models_and_scalers()
        
        # 設定特徵欄位
        self.feature_columns = [
            'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)',
            'Sunlight(Lux)', 'WindSpeed(m/s)',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'hour_sin', 'hour_cos'
        ]
        
    def _load_models_and_scalers(self):
        """載入所有模型和縮放器"""
        self.feature_models = []
        self.power_models = []
        self.feature_scalers = []
        self.power_scalers = []
        
        for loc in range(1, self.location_count + 1):
            # 載入LSTM模型
            self.feature_models.append(
                load_model(os.path.join(self.model_folder, f'FeatureLSTM_{loc:02d}.h5'))
            )
            self.power_models.append(
                load_model(os.path.join(self.model_folder, f'PowerLSTM_{loc:02d}.h5'))
            )
            
            # 載入MinMaxScaler
            self.feature_scalers.append(
                joblib.load(os.path.join(self.model_folder, f'Feature_MinMaxScaler_{loc:02d}.pkl'))
            )
            self.power_scalers.append(
                joblib.load(os.path.join(self.model_folder, f'Power_MinMaxScaler_{loc:02d}.pkl'))
            )
    
    def _process_source_data(self, source_data):
        """處理源數據,添加時間特徵"""
        df = source_data.copy()
        
        # 提取時間資訊
        df['year'] = df['Serial'].str[:4].astype(int)
        df['month'] = df['Serial'].str[4:6].astype(int)
        df['day'] = df['Serial'].str[6:8].astype(int)
        df['hour'] = df['Serial'].str[8:10].astype(int)
        df['minute'] = df['Serial'].str[10:12].astype(int)
        
        # 建立DateTime欄位並計算每月天數
        df['DateTime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
        df['days_in_month'] = df['DateTime'].dt.days_in_month
        
        # 計算循環特徵
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / df['days_in_month'])
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / df['days_in_month'])
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 移除不需要的欄位
        df = df.drop(columns=['year', 'month', 'minute', 'day', 'hour', 
                            'days_in_month', 'DateTime'])
        
        return df
    
    def _prepare_prediction_data(self, source_data, question_id, loc_index):
        """準備預測所需的輸入數據"""
        refer_title = source_data[['Serial']].values
        refer_data = source_data[self.feature_columns].values
        
        inputs = []
        inputs_for_power = []
        
        # 找出同一天的資料
        for i in range(len(refer_title)):
            if str(int(refer_title[i]))[:8] == str(int(question_id))[:8]:
                temp_data = pd.DataFrame(
                    refer_data[i].reshape(1, -1),
                    columns=self.feature_columns
                )
                temp_data_power = pd.DataFrame(
                    refer_data[i].reshape(1, -1),
                    columns=['Power(mW)'] + self.feature_columns
                )
                
                scaled_data = self.feature_scalers[loc_index].transform(temp_data)
                scaled_data_power = self.power_scalers[loc_index].transform(temp_data_power)
                
                inputs.append(scaled_data)
                inputs_for_power.append(scaled_data_power)
        
        return inputs, inputs_for_power
    
    def predict(self, question_path):
        """執行預測流程"""
        # 讀取測試資料
        upload_df = pd.read_csv(question_path, encoding='utf-8')
        questions = upload_df['序號'].values
        
        # 建立輸出DataFrame
        results_df = pd.DataFrame(columns=['序號', '答案'])
        
        count = 0
        while count < len(questions):
            # 取得位置代碼
            location_code = int(questions[count])
            str_location_code = f"{location_code:02d}"
            loc_index = int(str_location_code) - 1
            
            # 讀取和處理源數據
            source_data = self._load_and_merge_source_data(str_location_code)
            processed_data = self._process_source_data(source_data)
            
            # 準備預測資料
            inputs, inputs_power = self._prepare_prediction_data(
                processed_data, questions[count], loc_index
            )
            
            # 進行預測
            for i in range(self.forecast_num):
                predicted_power = self._make_single_prediction(
                    inputs, inputs_power, i, loc_index
                )
                
                # 儲存預測結果
                results_df.loc[len(results_df)] = {
                    '序號': questions[count + i][0],
                    '答案': predicted_power
                }
            
            count += self.forecast_num
        
        return results_df
    
    def _load_and_merge_source_data(self, location_code):
        """載入並合併源數據"""
        base_path = os.getcwd()
        primary_data = pd.read_csv(
            f'{base_path}/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_{location_code}.csv',
            encoding='utf-8'
        )
        
        try:
            secondary_data = pd.read_csv(
                f'{base_path}/processed_csv/L{location_code.lstrip("0")}_Train_2.csv',
                encoding='utf-8'
            )
            return pd.concat([primary_data, secondary_data], axis=0, ignore_index=True)
        except FileNotFoundError:
            return primary_data
    
    def _make_single_prediction(self, inputs, inputs_power, step, loc_index):
        """進行單次預測"""
        if step > 0:
            inputs.append(predict_output[step-1].reshape(1, 11))
            inputs_power.append(
                np.concatenate((predict_output[step-1], predict_output[step-1])).reshape(1, 12)
            )
        
        # 準備測試資料
        x_test = [inputs[-self.look_back_num + step:]]
        x_test_power = [inputs_power[-self.look_back_num + step:]]
        
        # 重塑資料
        x_test = np.reshape(np.array(x_test), 
                          (1, self.look_back_num, len(self.feature_columns)))
        x_test_power = np.reshape(np.array(x_test_power),
                                (1, self.look_back_num, len(self.feature_columns) + 1))
        
        # 預測特徵和發電量
        predicted_feature = self.feature_models[loc_index].predict(x_test, verbose=0)
        predicted_power = self.power_models[loc_index].predict(x_test_power, verbose=0)
        
        # 反轉縮放並取得最終預測值
        final_power = self.power_scalers[loc_index].inverse_transform(
            np.concatenate((predicted_power, predicted_feature), axis=1)
        )[:, 0]
        
        return np.round(final_power.flatten()[0], 2)

# 使用範例
if __name__ == "__main__":
    # 初始化模型
    model = PowerPredictionModel(
        timestamp="YOUR_TIMESTAMP",
        location_count=36,
        current_directory=os.getcwd()
    )
    
    # 執行預測
    question_path = os.path.join(os.getcwd(), 
                               '36_TestSet_SubmissionTemplate/upload(no answer).csv')
    results = model.predict(question_path)
    
    # 儲存結果
    results.to_csv('output.csv', index=False)
    print('Output CSV File Saved')