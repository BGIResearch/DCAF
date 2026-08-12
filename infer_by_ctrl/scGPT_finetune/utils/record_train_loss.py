import pandas as pd
import os


class TrainLossRecorder():
    """Record training loss of each iteration"""
    def __init__(self):
        self.train_loss_df = pd.DataFrame(columns=['epoch', 'iter', 'loss', 'exp_mse', 'exp_deg_mse',
                                                   'exp_mse_norm', 'exp_deg_mse_norm'])

    def record(self, epoch, iter, loss_val, exp_mse, exp_deg_mse, exp_mse_norm, exp_deg_mse_norm, every_iter=30):
        if (epoch == 0 and iter == 0) or (iter + 1) % every_iter == 0:
            self.train_loss_df = pd.concat([self.train_loss_df,
                                           pd.DataFrame([{'epoch': epoch,
                                                          'iter': iter,
                                                          'loss': loss_val,
                                                          'exp_mse': exp_mse,
                                                          'exp_deg_mse': exp_deg_mse,
                                                          'exp_mse_norm': exp_mse_norm,
                                                          'exp_deg_mse_norm': exp_deg_mse_norm}])],
                                           ignore_index=True)

    def save_history(self, loss_dir):
        self.train_loss_df.to_csv(os.path.join(loss_dir, 'train_loss.csv'))

