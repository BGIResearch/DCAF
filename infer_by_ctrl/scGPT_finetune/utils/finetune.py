"""train and predict functions.
"""

import time
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from .loss_and_acc import mse_gene_exp
from .early_stopping import EarlyStopping
from .evaluate_train_val_test import FinetuneEvaluator
from .record_train_loss import TrainLossRecorder

from .get_gene_exp_by_model import get_exp


def finetune(model,
             vocab,
             model_configs,
             train_loader,
             train_eval_loader,
             val_loader,
             device: str = 'cuda',
             lr: float = 1e-3,
             weight_decay: float = 5e-4,
             n_epoch=20,
             early_stop_on=False,
             path_save_model=None,
             loss_dir=None):
    """training.

    param model: llm with pre-training weights applied on.
    """
    model = model.to(device)

    # adopt a higher than default epsilon because of half precision
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4, weight_decay=weight_decay)
    schedular = StepLR(optimizer, step_size=1, gamma=0.5)

    if early_stop_on:
        early_stopping = EarlyStopping()

    evaluator = FinetuneEvaluator()
    train_loss_recorder = TrainLossRecorder()

    for epoch in range(n_epoch):
        # train
        model.train()
        ts = time.time()
        for iter_, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            print('gpu mem available after loading a training batch: ', torch.cuda.mem_get_info()[0] / 1024 ** 3)

            print('epoch {}, step {}'.format(epoch+1, iter_+1))

            train_exp_in = batch['exp_in']   # sum normed, log transformed
            train_exp_out = batch['exp_out']  # raw
            train_id = batch['id']

            train_exp_in = train_exp_in.to(device, torch.float32)
            train_exp_out = train_exp_out.to(device, torch.float32)
            train_id = train_id.to(device, torch.float32)

            ts_train = time.time()
            optimizer.zero_grad()
            print('gpu mem available after zero grad: ', torch.cuda.mem_get_info()[0] / 1024 ** 3)
            # here, empty_cache doesn't help in getting more mem

            train_exp_pred, mask = get_exp(train_exp_in, train_id, vocab, model_configs, model=model, mode='train')
            print('gpu mem available after forward pass: ', torch.cuda.mem_get_info()[0] / 1024 ** 3)

            loss = mse_gene_exp(train_exp_pred, train_exp_out, thresh_0=False)  # [mask]

            print('loss val {}'.format(loss))

            print('loss', loss.requires_grad)
            loss.backward()
            print('gpu mem available after computing gradient: ', torch.cuda.mem_get_info()[0] / 1024 ** 3)

            # todo: nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            optimizer.step()
            print('gpu mem available after updating parameters: ', torch.cuda.mem_get_info()[0] / 1024 ** 3)
            te_train = time.time()

            print('time for forward + backward', te_train - ts_train)

            # for recording train loss
            train_loss_recorder.record(epoch + 1, iter_ + 1, loss.item(), None, None, None, None, every_iter=1)
            # fixme, increase every_iter when training for long time
            if loss_dir is not None and (iter_ + 1) % 1 == 0:  # fixme: 2
                train_loss_recorder.save_history(loss_dir)

            # for monitoring val loss: todo: delete when validation dataset gets full size, or simply quite large
            if (iter_ + 1) % 100 == 0:
                with torch.no_grad():
                    # epoch, iter, model, vocab, model_configs, batch_loader, type, device
                    _ = evaluator.eval(epoch + 1, iter_ + 1, model, vocab, model_configs, train_eval_loader, 'train_eval', device)
                    val_loss = evaluator.eval(epoch + 1, iter_ + 1, model, vocab, model_configs, val_loader, 'val', device)
                print('epoch: {}, iter: {}, val loss: {}'.format(epoch + 1, iter_ + 1, val_loss))
                if loss_dir is not None:
                    evaluator.save_history(loss_dir)

            if path_save_model is not None and (iter_ + 1) % 100 == 0:
                torch.save(model.state_dict(), path_save_model)

        te = time.time()
        print('time for the epoch', te-ts)
        schedular.step()

        # evaluate on both train and validation data, at the end of an epoch
        model.eval()
        with torch.no_grad():
            _ = evaluator.eval(epoch + 1, iter_ + 1, model, vocab, model_configs, train_eval_loader, 'train_eval', device)
            val_loss = evaluator.eval(epoch + 1, iter_ + 1, model, vocab, model_configs, val_loader, 'val', device)
            print('epoch: {}, iter: {}, val loss: {}'.format(epoch + 1, iter_+1, val_loss))
            if loss_dir is not None:
                evaluator.save_history(loss_dir)

        # todo: save model if keyboard interrupt

        # evaluate on test data, and save model, when early-stopped
        if early_stop_on:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print('Early stopped.')

                if path_save_model is not None:
                    torch.save(model.state_dict(), path_save_model)
                return

    # todo: save model, at the last assigned epoch

    if path_save_model is not None:
        torch.save(model.state_dict(), path_save_model)




