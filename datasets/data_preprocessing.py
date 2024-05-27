import os

import torch
import concurrent.futures
import pandas as pd
import numpy as np
from functools import partial
from sklearn.preprocessing import MinMaxScaler

from configs import config
from logger import logger

args = config.Args().get_parser()

def label_cal(data):
    # 定义一个新的列来存放变化
    data['change'] = data['y1price'] - data['price']    
    # 根据变化定义标签
    data['label'] = 1  # 默认设置为1，表示价格保持不变
    data.loc[data['change'] > 0, 'label'] = 2  # 如果y1price > price，标签设置为2，表示上涨
    data.loc[data['change'] < 0, 'label'] = 0  # 如果y1price < price，标签设置为0，表示下跌    
    # 计算标签的比例
    count_0 = (data['label'] == 0).sum()
    count_1 = (data['label'] == 1).sum()
    count_2 = (data['label'] == 2).sum()
    total = len(data['label'])
    ratio_0 = count_0 / total
    ratio_1 = count_1 / total
    ratio_2 = count_2 / total
    logger.info('Label ratio (2: rise, 1: stationary, 0: fall): rise ratio: {:.2f}, stationary ratio: {:.2f}, fall ratio: {:.2f}'.format(ratio_2, ratio_1, ratio_0))  

    # 删除'change'列
    if 'change' in data.columns:
        data.drop('change', axis=1, inplace=True)   
    return data

def process_file(file, global_mapping_columns):
    data = pd.read_csv(file, low_memory=False)
    stock_tag_list = [1000]
    data = data[data['stock_tag'].isin(stock_tag_list)]
    unique_values = {col: set(data[col].dropna().astype(str).unique()) for col in global_mapping_columns}
    data = data.drop(columns=['stock_tag', 'time', 'price', 'y1price', 'y2price', 'y3price', 
                              'y4price', 'y5price', 'y6price', 'y7price', 'y8price', 'y9price', 
                              'y10price', 'y11price', 'y12price', 'y13price', 'y14price', 'y15price', 
                              'wind_code', 'date'])
    data = data.dropna().reset_index(drop=True)
    return data, unique_values


def create_inout_sequences(input_data, tw):
    sequences = []
    labels = []
    L = len(input_data)
    for i in range(L - tw + 1):
        # 提取tw个时间步的所有特征，不包括标签
        input_seq = input_data[i:i + tw, :-1]   
        # 标签取tw个时间步中最后一个时间步的最后一列    
        input_label = input_data[i + tw - 1, -1]
        sequences.append(input_seq)
        labels.append(input_label)
    # 将列表转换为单个 NumPy 数组
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    sequences = torch.FloatTensor(sequences)
    labels = torch.LongTensor(labels)
    return sequences, labels


def encode_columns(data, mapping_dicts):
    for column, mapping_dict in mapping_dicts.items():
        # 确保列中的值为字符串类型
        data[column] = data[column].astype(str)
        # 应用映射
        data[column + '_embedding'] = data[column].map(mapping_dict)
        # 删除原始列
        data = data.drop(columns=[column])
    return data

def mapping_and_identify(data_path, global_mapping_columns, columns_not_remove, threshold=0.3):
    processed_data_list = []
    all_unique_values = {col: set() for col in global_mapping_columns}

    # 使用 functools.partial 来创建一个可序列化的函数调用
    process_func = partial(process_file, global_mapping_columns=global_mapping_columns)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_func, data_path)

    for data, unique_values in results:
        processed_data_list.append(data)
        for col in unique_values:
            all_unique_values[col].update(unique_values[col])

    mapping_dicts = {col: {value: idx for idx, value in enumerate(sorted(values))} for col, values in all_unique_values.items()}
    combined_data = pd.concat(processed_data_list, ignore_index=True)
    columns_to_remove = []

    for column in combined_data.columns:
        if column not in columns_not_remove:
            max_duplication_ratio = combined_data[column].value_counts().max() / len(combined_data)
            if max_duplication_ratio > threshold:
                columns_to_remove.append(column)

    return mapping_dicts, columns_to_remove

# IRQ
def replace_outliers(group, columns, threshold_factor=3):
    for col in columns:
        Q1 = group[col].quantile(0.25)
        Q3 = group[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold_factor * IQR
        upper_bound = Q3 + threshold_factor * IQR

        outlier_indices = group[(group[col] < lower_bound) | (group[col] > upper_bound)].index
        
        for i in outlier_indices:
            col_index = group.columns.get_loc(col)
            if i > 0 and i < len(group) - 1:
                # 使用前一个和后一个元素的平均值
                group.iloc[i, col_index] = (group.iloc[i - 1, col_index] + group.iloc[i + 1, col_index]) / 2
            elif i == 0 and len(group) > 1:
                # 第一个元素，使用后一个元素的值
                group.iloc[i, col_index] = group.iloc[i + 1, col_index]
            elif i == len(group) - 1 and len(group) > 1:
                # 最后一个元素，使用前一个元素的值
                group.iloc[i, col_index] = group.iloc[i - 1, col_index]
    
    return group

# 并行处理文件
def process_group(group, columns_to_normalize, scaler, threshold=3.0):
    group = replace_outliers(group, columns_to_normalize, threshold)
    group[columns_to_normalize] = scaler.fit_transform(group[columns_to_normalize])
    return group

def process_and_normalize_file(file, columns_to_remove, columns_to_ignore, global_mappings):
    # 读取单个文件
    data = pd.read_csv(file, low_memory=False)

    # wind_code_list = [key for key in global_mappings['wind_code']][:500]
    # data = data[data['wind_code'].isin(wind_code_list)]
    stock_tag_list = [1000]
    data = data[data['stock_tag'].isin(stock_tag_list)]
    # 删除特定的列
    data = data.drop(columns=['date', 'y2price', 'y3price', 'y4price', 'y5price', 'y6price', 
                              'y7price', 'y8price', 'y9price', 'y10price', 'y11price', 'y12price', 
                              'y13price', 'y14price', 'y15price']) 
    # 删除含空值的行
    data = data.dropna().reset_index(drop=True)
    # 特征嵌入
    data = encode_columns(data, global_mappings)
    # 标签计算
    data = label_cal(data)
    # 删除 'y1price' 列
    data= data.drop(columns=['y1price'])
    # 标记数据来源文件
    filename = os.path.basename(file)
    data['file_source'] = filename
    # 删除重复度高的列
    data = data.drop(columns=columns_to_remove).reset_index(drop=True)
    # 选择所有数值类型列
    cols_to_convert = data.columns.difference(columns_to_ignore)
    data[cols_to_convert] = data[cols_to_convert].astype(float)

    # 分组归一化处理
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data['index'] = data.index
    # 根据 'wind_code_embedding' 分组
    grouped = data.groupby('wind_code_embedding')  
    # 初始化一个空的DataFrame来存储处理后的数据
    data_normalized = pd.DataFrame()
    # 并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
        futures = {executor.submit(process_group, group.copy(),
                                    [col for col in group.columns if col not in columns_to_ignore],
                                    scaler, threshold=3.0):name for name, group in grouped}
    
        for future in concurrent.futures.as_completed(futures):
            result_group = future.result()
            data_normalized = pd.concat([data_normalized, result_group], ignore_index=True)

    return data_normalized

def combine_processed_data(data_path, columns_to_remove, columns_to_ignore, global_mappings):
    processed_data_list = [process_and_normalize_file(file, columns_to_remove, columns_to_ignore, global_mappings) for file in data_path]
    combined_data = pd.concat(processed_data_list, ignore_index=True)
    return combined_data

def load_data(train_filename, test_filename):
    train_data = pd.read_csv(train_filename)
    val_data = pd.read_csv(test_filename)

    train_data = train_data.drop(columns=['index'])
    val_data = val_data.drop(columns=['index'])

    tw = args.input_window
    train_sequences, train_labels = create_inout_sequences(train_data.values, tw)
    test_sequences, test_labels = create_inout_sequences(val_data.values, tw)

    return train_sequences, train_labels, test_sequences, test_labels

def make_data(folder_path, start_index, num_files):

    # file names
    data_path = [] 
    files = os.listdir(folder_path)
    files.sort()
    end_index = start_index + num_files
    if end_index > len(files):
        raise ValueError("索引超出范围，没有足够的文件可选")
    selected_files = files[start_index:end_index]

    # 获取首尾文件的日期部分作为文件名
    first_date = selected_files[0].split('_')[-1].split('.')[0]
    last_date = selected_files[-1].split('_')[-1].split('.')[0]

    for file in selected_files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            data_path.append(file_path)

    # 要编码的列名列表
    global_mapping_columns = ['wind_code', 'time', 'stock_tag']  
    # 列表中不需要归一化的列
    columns_to_ignore = ['stock_tag_embedding', 'time_embedding', 'wind_code_embedding', 'label','file_source','index']
    # 创建全局映射，并返回需要删除的列
    global_mappings, columns_to_remove = mapping_and_identify(data_path, global_mapping_columns, columns_to_ignore, threshold=0.3)
    # 调用combine_data函数处理文件并合并
    data_combined = combine_processed_data(data_path, columns_to_remove, columns_to_ignore, global_mappings)

    val_file = os.path.basename(data_path[-2])
    test_file = os.path.basename(data_path[-1])
    data_combined = data_combined.reset_index(drop=True)
    train_data = data_combined[(data_combined['file_source'] != test_file) & (data_combined['file_source'] != val_file)]
    val_data = data_combined[data_combined['file_source'] == val_file]
    test_data = data_combined[data_combined['file_source'] == test_file]
    train_data = train_data.drop(columns=['file_source'])
    val_data = val_data.drop(columns=['file_source'])
    test_data = test_data.drop(columns=['file_source'])

    # 生成含日期的文件名
    train_filename = f'data/train_data_{first_date}_to_{last_date}.csv'
    val_filename = f'data/val_data_{first_date}_to_{last_date}.csv'
    test_filename = f'data/test_data_{first_date}_to_{last_date}.csv'

    # 保存处理后的数据到文件中
    train_data.to_csv(train_filename, index=False)
    val_data.to_csv(val_filename, index=False)
    test_data.to_csv(test_filename, index=False)
    logger.info('----------------Saved preprocessed data----------------')

    train_data = train_data.drop(columns=['index'])
    val_data = val_data.drop(columns=['index'])

    tw = args.input_window
    train_sequences, train_labels = create_inout_sequences(train_data.values, tw)
    val_sequences, val_labels = create_inout_sequences(val_data.values, tw)

    return train_sequences, train_labels, val_sequences, val_labels


