import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()

        return parser
    
    @staticmethod
    def initialize(parser):
        # args for training, generating data, backtesting switches
        parser.add_argument('--should_train_model', default=False, type=bool, help='whether we wanna train models')
        parser.add_argument('--should_generating_data', default=True, type=bool, help='whether we wanna make backtesting data')
        parser.add_argument('--should_backtest', default=True, type=bool, help='whether we wanna make backtesting result')

        # args for path
        parser.add_argument('--output_dir', default='./checkpoints', help='output dir  of model checkpoints')
        parser.add_argument('--data_folder_dir', default='/data/share/new_factor_data', help='data folder dir')
        parser.add_argument('--log_dir', default='./logs/', help='logs dir of training and testing process')

        # args for training and testing
        parser.add_argument('--train_epochs', default=11, type=int, help='setting training epoch')
        parser.add_argument('--warmup_epochs', default=5, type=int, help='setting warmup epoch')
        parser.add_argument('--early_stop_patience', default=1000, type=int, help='setting early stop patience')
        parser.add_argument('--dropout_prob', default=0.1, type=float, help='drop out probability')
        parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
        parser.add_argument('--min_lr', default=3e-5, type=float, help='learning rate')
        parser.add_argument('--weight_decay', default=0.00, type=float)
        parser.add_argument('--batch_size', default=2048, type=int)

        # other args
        parser.add_argument('--log_name', type=str, default='Vanilla,iTransformer', help='names of prediction logs')
        parser.add_argument('--model_names', type=str, default='Vanilla', help='names of AI models (Vanilla, iTransformer)')
        parser.add_argument('--train_ratio', type=int, default=0.8, help='ratio of separating train and validation')
        parser.add_argument('--num_files', type=int, default=10, help='number of files')
        parser.add_argument('--start_index',type=int, default=3, help='index of the start file')
        parser.add_argument('--input_window', type=int, default=60, help='number of time steps we use to predict')
        parser.add_argument('--output_window', type=int, default=1, help='number of time steps we want to predict')
        parser.add_argument('--seed', type=int, default=40, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default='4,5,6,7', help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
        parser.add_argument("--local_rank", type=int, default=0,)

        # args for iTransformer
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')


        return parser
    
    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)

        return parser.parse_args()
