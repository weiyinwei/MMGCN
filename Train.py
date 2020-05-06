import torch
import torch.nn as nn
from tqdm import tqdm


def train(epoch, length, dataloader, model, optimizer, batch_size, writer=None):    
    model.train()
    print('Now, training start ...')
    sum_loss = 0.0
    sum_model_loss = 0.0
    sum_reg_loss = 0.0
    sum_ent_loss = 0.0
    sum_weight_loss = 0.0
    step = 0.0
    pbar = tqdm(total=length)
    num_pbar = 0
    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()
        loss, model_loss, reg_loss, weight_loss, entropy_loss = model.loss(user_tensor, item_tensor)
        loss.backward(retain_graph=True)
        optimizer.step()
        pbar.update(batch_size)
        num_pbar += batch_size
        sum_loss += loss.cpu().item()
        sum_model_loss += model_loss.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()
        sum_ent_loss += entropy_loss.cpu().item()
        sum_weight_loss += weight_loss.cpu().item()
        step += 1.0
    pbar.close()
    print('----------------- loss value:{}  model_loss value:{}  reg_loss value:{} entropy_loss value:{} weight_loss value:{}--------------'
        .format(sum_loss/step, sum_model_loss/step, sum_reg_loss/step, sum_ent_loss/step, sum_weight_loss/step))
    if writer is not None:
        writer.add_scalar('loss', sum_loss/step, epoch)
        writer.add_scalar('model_loss', sum_model_loss/step, epoch)
        writer.add_scalar('reg_loss', sum_reg_loss/step, epoch)

    return loss
