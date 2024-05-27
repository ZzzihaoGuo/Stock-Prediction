import torch.utils.data as Data
from torch.utils.data.distributed import DistributedSampler

from configs import config

args = config.Args().get_parser()

class Data_load_process:
    def __init__(self, args):
        self.args = args
        # self.device = device

    def data_load_process(self, train_sequences, train_labels, val_sequences, val_labels):

        train_dataset = Data.TensorDataset(train_sequences, train_labels)
        val_dataset = Data.TensorDataset(val_sequences, val_labels)
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        train_dataloader = Data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
        val_dataloader = Data.DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=DistributedSampler(val_dataset))

        return train_dataloader, val_dataloader, train_sampler, val_sampler
