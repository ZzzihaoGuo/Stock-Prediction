import os
import sys
import importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd

from configs import config
from Model_process import load_model
# from models.PatchTST import PatchTST
from datasets.data_preprocessing import create_inout_sequences


args = config.Args().get_parser()

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def get_backtesting_data(data_path, start_index, model_info, pickle_filename):

    test_file_name = data_path[-1]

    backtesting_data = pd.read_csv(test_file_name, low_memory=False)

    stock_tag_list = [1000]
    backtesting_data = backtesting_data[backtesting_data['stock_tag'].isin(stock_tag_list)]

    # 删除 'price' 为 0 的行
    backtesting_data = backtesting_data[backtesting_data['price'] != 0]
   
    # 删除 'y15price' 列中包含 NaN 的行
    backtesting_data = backtesting_data.dropna(subset=['y15price'])

    if 'wind_code' in backtesting_data.columns and 'time' in backtesting_data.columns:
        df_cleaned = backtesting_data.drop_duplicates(subset=['wind_code', 'time'])
  
    # 动态文件名基于第一个和最后一个文件的日期
    first_file_date = data_path[0].split('_')[-1].split('.')[0]
    last_file_date = data_path[-1].split('_')[-1].split('.')[0]

    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)
    test_data = data['test']

    if len(df_cleaned) != len(test_data)*len(test_data[0]):
        raise ValueError("模型测试集数据和回测数据不符！")
    
    # 生成验证序列和标签
    tw = args.input_window
    test_sequences_labels = [create_inout_sequences(wind_code, tw) for wind_code in test_data]
    test_sequences = torch.cat([i[0] for i in test_sequences_labels])
    test_labels = torch.cat([i[1] for i in test_sequences_labels])

    feature_size = test_sequences.shape[2]
    df_cleaned = df_cleaned.drop(df_cleaned.index[:tw-1])

    print("length of test data: ", test_sequences.shape[0])

    test_dataset = Data.TensorDataset(test_sequences.to(device), test_labels.to(device))

    test_dataloader = Data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    result = []

    def keep_last_n_rows(group, n=222-tw+1):
        return group.iloc[-n:]

    for module_name, class_name in model_info:

        df_cleaned_temp = df_cleaned.copy(deep=True)
        df_cleaned_temp = df_cleaned_temp.groupby('wind_code').apply(keep_last_n_rows).reset_index(drop=True)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        outputs = []
        outputs_prob = []
        softmax = nn.Softmax(dim=1)
        model = load_model(model_class, feature_size, class_name, start_index).to(device)
        model.eval()

        for x, y in test_dataloader:
            # x = x.to(device)
            # with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = model(x)
                _, predicted = torch.max(output, 1)
                probs = softmax(output)
                outputs.append(predicted)
                outputs_prob.append(probs)
                torch.cuda.empty_cache()

        outputs_tensor = torch.cat(outputs, dim=0)
        outputs_prob_tensor = torch.cat(outputs_prob, dim=0)

        outputs_numpy = outputs_tensor.cpu().detach().numpy()
        outputs_prob_numpy = outputs_prob_tensor.cpu().detach().numpy()

        df_cleaned_temp['Prediction'] = outputs_numpy
        df_cleaned_temp['Probability'] = outputs_prob_numpy.tolist()
        df_cleaned_temp['label'] = list(test_labels)

        backtesting_data_filename = f'data/backtesting_{first_file_date}_to_{last_file_date}_{class_name}.csv'
        df_cleaned_temp.to_csv(backtesting_data_filename, index=False)

        result.append(df_cleaned_temp)
    return result
