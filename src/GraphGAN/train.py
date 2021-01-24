import torch
import torch.nn as nn
import tqdm
import numpy as np

import sys, os
sys.path.append("../..") #添加根目录
os.chdir(sys.path[0]) #以当前文件位置为工作目录

import graphGAN
import config

from src import utils
from src.evaluation import link_prediction as lp
from src.evaluation import evaluation


def show_progress(gGAN, item_name, epoch, batch_index, batch_num, loss):
    '''可视化并保存进度'''
    barlen = 40
    ok_len = int((batch_index+1) / batch_num * barlen)

    epoch_info = 'epoch:{}'.format(epoch+1)
    loss_info = 'loss: {}'.format(loss)
    batch_info = '{}/{}'.format(batch_index+1, batch_num)
    bar_str = '[' + '>'*ok_len + '-'*(barlen-ok_len) + ']'
    info_end = '\r' #行首
    info_list = [item_name, epoch_info, batch_info, bar_str, loss_info]

    if batch_index+1 == batch_num:
        val_info = evaluation.eval_while_train(gGAN, item_name, epoch, save=True)
        if val_info != '':
            info_list.append(val_info)
        info_end = '\n' #换行
    
    progress_info = ' '.join(info_list)
    print(progress_info, end=info_end, flush = True)


def D_step(gGAN):
    discriminator = gGAN.discriminator
    optimizer_D = torch.optim.Adam(discriminator.parameters() ,lr=config.lr_dis)
    center_nodes = []
    neighbor_nodes = []
    labels = []
    for d_epoch in range(config.n_epochs_dis):
        # generate new nodes for the discriminator for every dis_interval iterations
        if d_epoch % config.dis_interval == 0:
            center_nodes, neighbor_nodes, labels = gGAN.prepare_data_for_d()
        # training
        train_size = len(center_nodes)
        start_list = list(range(0, train_size, config.batch_size_dis))
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_dis
            
            loss = discriminator.loss(node_id=np.array(center_nodes[start:end]), 
                                      node_neighbor_id=np.array(neighbor_nodes[start:end]),
                                      label=np.array(labels[start:end]))
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i+1)
            show_progress(gGAN, 'dis', d_epoch, i, len(start_list), avg_loss)


def G_step(gGAN):
    generator = gGAN.generator
    discriminator = gGAN.discriminator
    optimizer_G = torch.optim.Adam(generator.parameters() ,lr=config.lr_gen)
    node_1 = []
    node_2 = []
    reward = []
    for g_epoch in range(config.n_epochs_gen):
        if g_epoch % config.gen_interval == 0:
            node_1, node_2, reward = gGAN.prepare_data_for_g()

        # training
        train_size = len(node_1)
        start_list = list(range(0, train_size, config.batch_size_gen))
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_gen
            score = generator.score(node_id=np.array(node_1[start:end]), 
                                    node_neighbor_id=np.array(node_2[start:end]))
            prob = discriminator(score)
            loss = generator.loss(prob=prob, 
                                  reward=reward[start:end])
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            
            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i+1)
            show_progress(gGAN, 'gen', g_epoch, i, len(start_list), avg_loss)


if __name__ == "__main__":  

    gGAN = graphGAN.graphGAN()
    '''
    # restore the model from the latest checkpoint if exists
    if os.path.isfile(config.dis_state_dict_filename) and os.path.isfile(config.gen_state_dict_filename):
        print('restore the model')
        checkpoint_dis, checkpoint_gen = torch.load(config.dis_state_dict_filename), torch.load(config.gen_state_dict_filename)
        gGAN.discriminator.load_state_dict(checkpoint_dis)
        gGAN.discriminator.load_state_dict(checkpoint_gen)
            
    '''
    gGAN.write_embeddings_to_file()
    evaluation.eval(gGAN, epoch='pre_train')

    print("start training...")
    for epoch in range(config.n_epochs):
        print("epoch %d" % epoch)

        # save the model
        if epoch > 0 and epoch % config.save_steps == 0:
            torch.save(gGAN.discriminator.state_dict(), config.dis_state_dict_filename)
            torch.save(gGAN.generator.state_dict(), config.gen_state_dict_filename)

        # D-steps
        D_step(gGAN)

        # G-steps
        G_step(gGAN)

        gGAN.write_embeddings_to_file()
        evaluation.eval(gGAN, epoch=epoch)
    print("training completes")
