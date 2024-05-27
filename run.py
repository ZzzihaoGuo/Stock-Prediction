import os
import time
import logging
import importlib
import random

import torch
import torch.distributed
import torch.multiprocessing as mp
import pandas as pd
from torch.distributed import destroy_process_group

from configs import config
from backtesting.make_back_test_data import get_backtesting_data
from datasets.data_preprocessing import load_data, make_data
from utils.utils import setup_seed, ddp_setup
from Model_process import Model_precessor
from logger import logger
from datasets.data_loader import Data_load_process
from models.Vanilla import Vanilla
from models.iTransformer import iTransformer
from backtesting.back_test_func import Backtest_v0,data_process_v0


args = config.Args().get_parser()

setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"


def main(rank, world_size, train_sequences, train_labels, val_sequences, val_labels, model_info):
    ddp_setup(rank, world_size)

    # preprocess and load data
    Data_load_class = Data_load_process(args)
    train_loader, test_loader, train_sampler, val_sampler = \
    Data_load_class.data_load_process(train_sequences, train_labels, val_sequences, val_labels)

    # training model
    for module_name, class_name in model_info:
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 从模块中获取类对象
        model_class = getattr(module, class_name)
        logger.info('--------------------train model {}--------------------'.format(class_name))
        feature_size = train_sequences.shape[2]
        model = model_class(feature_size)
        model_precessor = Model_precessor(args, train_loader, test_loader, train_sampler, val_sampler, model, rank, class_name)
        model_precessor.train_model()
    destroy_process_group()

def get_random_start_index(folder_path, num_files):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # 确保有足够的文件可供选择
    max_files = len(files)
    if max_files < num_files:
        raise ValueError(f"Not enough files in directory. Required: {num_files}, Available: {max_files}")
    max_start_index = max_files - num_files
    return random.randint(0, max_start_index)



if __name__ == '__main__':
    data_folder_path = args.data_folder_dir
    num_files = args.num_files
    
    logger.info(args)
    world_size = torch.cuda.device_count()
    logger.info('Number of GPUs: {}'.format(world_size))
    
    model_names = args.model_names.split(',')
    model_info = [('models.' + model_name, model_name)for model_name in model_names]

    # 预定义起始索引
    predefined_indices = [4, 16, 29]

    for start_index in predefined_indices:   
        # 根据动态文件名生成逻辑
        files = os.listdir(data_folder_path)
        files.sort()
        selected_files = files[start_index:start_index + num_files]
        logger.info(selected_files)
        first_file_date = selected_files[0].split('_')[-1].split('.')[0]
        last_file_date = selected_files[-1].split('_')[-1].split('.')[0]  
        train_filename = f'data/train_data_{first_file_date}_to_{last_file_date}.csv'
        test_filename = f'data/test_data_{first_file_date}_to_{last_file_date}.csv'
        backtesting_data_filename = f'data/backtesting_{first_file_date}_to_{last_file_date}.csv'

        # Check if backtesting data already exists
        if os.path.exists(backtesting_data_filename):
            logger.info(f"Loading existing backtesting data from {backtesting_data_filename}")
            data = pd.read_csv(backtesting_data_filename)
            should_train_model = False
        else:
            should_train_model = True
            logger.info(f"No existing backtesting data found for {first_file_date} to {last_file_date}")
  
        if args.should_train_model:
            # Check if model data already exists
            if os.path.isfile(train_filename) and os.path.isfile(test_filename):
                logger.info('----------Load saved data----------')
                train_sequences, train_labels, val_sequences, val_labels = load_data(train_filename, test_filename)
            else:
                logger.info('-----------Make new data-----------')
                train_sequences, train_labels, val_sequences, val_labels = make_data(data_folder_path, start_index, num_files)

            
            mp.spawn(main, args=(world_size, train_sequences, train_labels, val_sequences, val_labels, model_info), nprocs=world_size)
        
        if args.should_generating_data:
            # Generating new backtesting data
            logger.info(f"Generating new backtesting data for {first_file_date} to {last_file_date}")
            data_list = get_backtesting_data(data_folder_path, start_index, num_files, model_info)
            # data.to_csv(backtesting_data_filename, index=False)
            logger.info(f"Backtesting data saved to {backtesting_data_filename}")
        else:
            data_list = []
            for each_model in model_names:
                data_list.append(pd.read_csv(f'data/backtesting_{first_file_date}_to_{last_file_date}_{each_model}.csv'))
            logger.info(f"Load backtesting data {backtesting_data_filename}")

        if args.should_backtest:
            for i, data in enumerate(data_list):
                model_name = model_names[i]
                data_all, data_consider, data_label = data_process_v0(data)
                file_name = backtesting_data_filename.split('.')[0] + f'_{model_name}.png'
                
                backtest_v0 =  Backtest_v0(data_all, data_consider, data_label, time_col='time',stock_col='wind_code',profit_col='profit_rate',
                                sort_col='predict_prob',file_name=file_name)
                
                data_consider_after_n_max = backtest_v0.get_trading_data(data_consider, count_adj_portion = 1,n_max=None)
                data_label_after_n_max = backtest_v0.get_trading_data(data_label, count_adj_portion = 1,n_max=None)

                logger.info(len(data_consider_after_n_max[data_consider_after_n_max['profit_rate']>0])/len(data_consider_after_n_max))
                logger.info(data_consider_after_n_max[data_consider_after_n_max['profit_rate']>0]['profit_rate'].mean())
                logger.info(data_consider_after_n_max[data_consider_after_n_max['profit_rate']<0]['profit_rate'].mean())


                back_result = backtest_v0.get_result(data_consider_after_n_max, data_label_after_n_max, split_x=10)
                # print(back_result)

                # 保存每次循环的结果到单独的文件
                result_filename = f'data/backtesting_results_{first_file_date}_to_{last_file_date}_{model_name}.csv'
                back_result.to_csv(result_filename, index=False)
                logger.info(f'Results saved to {result_filename}')
