import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast


from logger import logger
from configs import config


args = config.Args().get_parser()

class Model_precessor:
    def __init__(self, args, train_loader, val_loader, train_sampler, val_sampler, model, gpu_id, class_name):
        self.gpu_id = gpu_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.best_up_val_acc = float('-inf')
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.model_name = class_name
        model = model
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Initialize warmup scheduler and main scheduler
        warmup_steps = len(self.train_loader) * self.args.warmup_epochs
        lr_lambda = lambda step: min(step / warmup_steps, 1.0)  # Linear warmup
        self.scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # Cosine Annealing after warmup
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(self.args.train_epochs - self.args.warmup_epochs) * len(self.train_loader), eta_min=0.00003)

        # Use SequentialLR to connect warmup and cosine annealing schedulers
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.scheduler_warmup, self.scheduler_cosine], milestones=[warmup_steps])


    
    # def lr_lambda(self, epoch):
    #     if epoch < self.args.warmup_epochs:
    #         # 预热阶段，学习率逐渐增加
    #         return (epoch) / self.args.warmup_epochs
    #     else:
    #         # 余弦退火学习率衰减
    #         rr = epoch / len(self.train_loader)/self.args.train_epochs
    #         rr = (1 - rr) ** 1.5
    #         rr = (rr*(self.args.lr-self.args.min_lr) + self.args.min_lr)/self.args.lr

    #         return rr

    def train_model(self):
        # self.current_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
        for  epoch in range(self.args.train_epochs):
            # Print current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{self.args.train_epochs}, Current Learning Rate: {current_lr}')

            total_correct = 0
            total_up = 0
            total_stay = 0
            total_down = 0
            total_correct_up = 0
            total_correct_stay = 0
            total_correct_down = 0
            total_loss = 0
            total_size = 0
            self.model.train()
            self.train_sampler.set_epoch(epoch)
            self.val_sampler.set_epoch(epoch)

            scaler = torch.cuda.amp.GradScaler()

            for i, (X, y) in enumerate(tqdm(self.train_loader)):

                X = X.to(self.gpu_id)
                y = y.to(self.gpu_id)

                with torch.cuda.amp.autocast():
                    predictions = self.model(X)
                    loss = self.criterion(predictions, y)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                # loss.backward()
                # self.optimizer.step()
                scaler.step(self.optimizer)
                scaler.update()

                self.scheduler.step()  # Update the learning rate at each step

                _, predicted = torch.max(predictions, 1)
                total_correct += (predicted == y).sum().item()
                
                for k, each_sample in enumerate(y):
                    if each_sample == 2:
                        total_up += 1
                        if predicted[k] == 2:
                            total_correct_up += 1
                    if each_sample == 1:
                        total_stay += 1
                        if predicted[k] == 1:
                            total_correct_stay += 1
                    if each_sample == 0:
                        total_down += 1
                        if predicted[k] == 0:
                            total_correct_down += 1

                total_loss += loss.item() * X.shape[0]
                total_size += X.shape[0]

            epoch_loss = total_loss / total_size
            train_accuracy = total_correct / total_size

            up_acc = total_correct_up/total_up
            stay_acc = total_correct_stay/total_stay
            down_acc = total_correct_down/total_down

            logger.info('[Train] epoch: {} | loss: {:.4f} | accuracy: {:.4f} | up_acc: {:.4f} | stay_acc: {:.4f} | down_acc: {:.4f}'.format(epoch, epoch_loss, train_accuracy, up_acc, stay_acc, down_acc))


            if epoch % 5 == 0 and self.gpu_id == 1:
                self.model.eval()
                total_val_size = 0
                total_val_loss = 0
                total_val_correct = 0
                total_val_up = 0
                total_val_stay = 0
                total_val_down = 0
                total_val_correct_up = 0
                total_val_correct_stay = 0
                total_val_correct_down = 0
                with torch.no_grad():
                    for X, y in self.val_loader:
                        X = X.to(self.gpu_id)
                        y = y.to(self.gpu_id)
                        output = self.model(X)
                        loss = self.criterion(output, y)

                        _, predicted = torch.max(output, 1)
                        total_val_correct += (predicted == y).sum().item()

                        for k, each_sample in enumerate(y):
                            if each_sample == 2:
                                total_val_up += 1
                                if predicted[k] == 2:
                                    total_val_correct_up += 1
                            if each_sample == 1:
                                total_val_stay += 1
                                if predicted[k] == 1:
                                    total_val_correct_stay += 1
                            if each_sample == 0:
                                total_val_down += 1
                                if predicted[k] == 0:
                                    total_val_correct_down += 1

                        total_val_loss += loss.item() * X.shape[0] 
                        total_val_size += X.shape[0]

                    epoch_val_loss = total_val_loss / total_val_size
                    validation_accuracy = total_val_correct / total_val_size

                    up_val_acc = total_val_correct_up/total_val_up
                    stay_val_acc = total_val_correct_stay/total_val_stay
                    down_val_acc = total_val_correct_down/total_val_down
                    logger.info('[Validation] epoch: {} | Val loss: {:.4f} | Val accuracy: {:.4f}'.format(epoch, epoch_val_loss, validation_accuracy))
                    logger.info('[Validation] epoch: {} | Val_up_acc: {:.4f} | Val_stay_acc: {:.4f} | Val_down_acc: {:.4f}'.format(epoch, up_val_acc, stay_val_acc, down_val_acc))
                    
                    # save best model
                    if epoch_val_loss < self.best_val_loss:
                        self.best_val_loss = epoch_val_loss
                        self.save_checkpoint(epoch, self.model_name)

                    # early_stopping
                    if self.check_early_stopping(epoch_val_loss):
                        logger.info("Early stopping triggered at epoch {}".format(epoch))
                        break


                    
    def predict(self):
        pass

    def save_checkpoint(self, epoch, model_name):
        ckp = self.model.module.state_dict()
        PATH = args.output_dir + '/checkpoint_' + model_name + '.pt'
        torch.save(ckp, PATH)
        logger.info(f'Epoch {epoch} | Training checkpoint saved at {PATH}')

    def check_early_stopping(self, up_val_acc):
        tolerance = 0.1
        if up_val_acc > self.best_up_val_acc:
            self.best_up_val_acc = up_val_acc
            self.early_stop_counter = 0
            return False
        else:
            if up_val_acc < self.best_up_val_acc * (1 + tolerance):
                self.early_stop_counter += 1
            if self.early_stop_counter >= self.args.early_stop_patience:
                return True
            return False


def load_model(Model, feature_size, model_name):
    model = Model(feature_size)
    PATH = args.output_dir + f'/checkpoint_{model_name}.pt'
    model.load_state_dict(torch.load(PATH))
    return model


