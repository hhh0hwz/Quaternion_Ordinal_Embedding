import scipy.io
import numpy as np
import torch
from torch.autograd import grad
import time
import sys
from math import ceil
import logging_util
import logging
from torch.utils.data import Dataset
import pandas as pd 
import os


def oe(data,
        no_dims,
        device,
        learning_rate=0.1,
        batch_size=1024,
        epochs=500,
        error_change_threshold=1e-7,
        loss_type='Euclid',
        loss_number=1):
    """
    Learn the triplet embedding for the given triplets.
    Returns an array with shape (max(triplets)+1, no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.
    """
    best_triplet_error_test = dict()
    for i in range(0,epochs+1,5):
        best_triplet_error_test[str(i)] = 0.0
    # print(best_triplet_error_test.keys())
    triplets = data.train_triplets
    # 判断-1不在是数据集里，如果存在-1就报错
    assert -1 not in triplets 
    n_points = np.max(triplets) + 1  # points的数目
    triplets_num = len(triplets)  # 三元组的数目
    # Initialize embedding
    random_embeddings = torch.rand(size=(n_points, no_dims), dtype=torch.float) 
    X = torch.Tensor(random_embeddings).to(device)
    X.requires_grad = True
    
    # 迭代次数以及优化器
    optimizer = torch.optim.Adam(params=[X], lr=learning_rate, amsgrad=True)
    # 迭代的次数
    batches = 1 if batch_size > triplets_num else triplets_num // batch_size 
    loss_history = []
    triplet_error_history = []
    time_history = []

    triplets = torch.tensor(triplets).to(device).long()
    logger.info('Number of Batches = ' + str(batches))

    # 计算初始的误差
    if loss_type == 'quaternion' or loss_type == 'quaternion_1' or loss_type=='quaternion_ori':
        triplet_error_history.append(triplet_error_torch(X, triplets, loss_type, device))
        triplet_error_test = triplet_error_torch(X, data.test_triplets, loss_type, device)
        best_triplet_error_test['0'] = triplet_error_test
    else:
        # print(triplet_error_torch(X, triplets, loss_type, device))
        # print(triplet_error_torch(X, triplets, loss_type, device)[0])
        triplet_error_history.append(triplet_error_torch(X, triplets, loss_type, device)[0].item())
        triplet_error_test = triplet_error_torch(X, data.test_triplets, loss_type, device)[0].item()
        best_triplet_error_test['0'] = triplet_error_test

    # 计算初始的loss
    epoch_loss = 0
    for batch_ind in range(batches):
        # get triplet
        batch_trips = triplets[batch_ind*batch_size: (batch_ind+1)*batch_size, ] 
        batch_xs = X[batch_trips, :]
        batch_loss = oe_loss(batch_xs[:, 0, :].squeeze(),
                             batch_xs[:, 1, :].squeeze(),
                             batch_xs[:, 2, :].squeeze(),
                             loss_type,
                             loss_number,
                             device,
                             no_dims)
        epoch_loss += batch_loss.item()
    loss_history.append(epoch_loss / triplets_num)

    # 迭代
    best_X = X
    total_time = 0
    time_to_best = 0
    best_triplet_error = triplet_error_history[0] 
    for it in range(epochs):
        intermediate_time = time.time()
        epoch_loss = 0
        for batch_ind in range(batches):
            batch_trips = triplets[batch_ind*batch_size: (batch_ind+1)*batch_size, ]

            batch_xs = X[batch_trips, :]

            batch_loss = oe_loss(
                                batch_xs[:, 0, :].squeeze(),
                                batch_xs[:, 1, :].squeeze(),
                                batch_xs[:, 2, :].squeeze(),
                                loss_type,
                                loss_number,
                                device,
                                no_dims
                            )
            optimizer.zero_grad()
            try:
                batch_loss.backward()
            except AttributeError:
                return best_X.cpu().detach().numpy(), best_triplet_error_test
            optimizer.step()
            epoch_loss += batch_loss.item()

        end_time = time.time()
        total_time += (end_time-intermediate_time)
        epoch_loss = epoch_loss / triplets_num

        # 记录loss, time
        time_history.append(total_time)
        loss_history.append(epoch_loss)

        if it % 5 == 0 or it == epochs -1:
            if error_change_threshold != -1:
                # 计算误差
                if loss_type == 'quaternion' or loss_type == 'quaternion_1' or loss_type=='quaternion_ori':
                    triplet_error = triplet_error_torch(X, triplets, loss_type, device)
                    triplet_error_test = triplet_error_torch(X, data.test_triplets, loss_type, device)  
                else:
                    triplet_error = triplet_error_torch(X, triplets, loss_type, device)[0].item()
                    triplet_error_test = triplet_error_torch(X, data.test_triplets, loss_type, device)[0].item()

                # triplet_error = triplet_error_torch(X, triplets, loss_type)[0].item()
                triplet_error_history.append(triplet_error)
                if it == epochs - 1:
                    best_triplet_error_test[str(it+1)] = triplet_error_test
                else:
                    best_triplet_error_test[str(it)] = triplet_error_test
                if triplet_error < best_triplet_error - error_change_threshold:
                    best_X = X
                    best_triplet_error = triplet_error
                    # best_triplet_error_test[str(it)] = triplet_error_test
                    time_to_best = total_time
                    logger.info('Found new best in Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error) + 
                                 ' Test error: ' + str(triplet_error_test))
                logger.info('Epoch: ' + str(it) + ' Loss: ' + str(epoch_loss) + ' Triplet error: ' + str(triplet_error) + ' Test error: ' + str(triplet_error_test))
                sys.stdout.flush()
            else:
                best_X = X
                time_to_best = total_time
                logger.info('Epoch: ' + str(it) + 'Loss: ' + str(epoch_loss))
                sys.stdout.flush()
    return best_X.cpu().detach().numpy(), best_triplet_error_test


def oe_loss(x_i, x_j, x_k, loss_type, loss_number, device, dims, delta=1):
    if loss_type == 'Euclid':
        return(oe_loss_Euclid(x_i, x_j, x_k, loss_number, dims, delta))
    if loss_type == 'hyperbolic':
        return(oe_loss_hyperbolic(x_i, x_j, x_k, loss_number, device,dims, delta))
    if loss_type == 'quaternion':
        return(os_loss_quaternion(x_i, x_j, x_k, loss_number, device, delta))
    if loss_type == 'quaternion_1':
        return(os_loss_quaternion_1(x_i, x_j, x_k, loss_number, device, dims, delta))
    # if loss_type == 'daul':
    #     return(os_loss_daul(x_i, x_j, x_k, loss_number, device, delta))
    if loss_type == 'quaternion_ori':
        return(os_loss_quaternion_ori(x_i, x_j, x_k, loss_number, device, dims, delta))

def oe_loss_Euclid(x_i, x_j, x_k, loss_number, dims, delta=1):
    # x:  batch_size * embedding_dims
    if loss_number == 1: 
        loss = delta + torch.norm(x_i-x_j, p=2, dim=1) - torch.norm(x_i-x_k, p=2, dim=1)
        loss_p = loss[loss>0]
    elif loss_number == 2:
        loss = delta + torch.norm(x_i-x_j, p=2, dim=1) - torch.norm(x_i-x_k, p=2, dim=1)
        loss = torch.pow(loss,exponent=2)
        loss_p = loss[loss>0]
    elif loss_number == 3:
        loss = torch.div(torch.exp(-torch.norm(x_i-x_j, p=2, dim=1)),
                        torch.exp(-torch.norm(x_i-x_j, p=2, dim=1)) + torch.exp(-torch.norm(x_i-x_k, p=2, dim=1)))
        loss_p = -torch.log(loss)
    elif loss_number == 4:
        loss = torch.div( (delta + torch.norm(x_i-x_j, p=2, dim=1)/(dims-1))**(dims/2),
                        (delta + torch.norm(x_i-x_j, p=2, dim=1)/(dims-1))**(dims/2) 
                         + (delta + torch.norm(x_i-x_k, p=2, dim=1)/(dims-1))**(dims/2))
        loss_p = torch.log(loss)
    elif loss_number == 5:
        loss = torch.div(torch.exp(torch.norm(x_i-x_j, p=2, dim=1)),
                        torch.exp(torch.norm(x_i-x_j, p=2, dim=1)) + torch.exp(torch.norm(x_i-x_k, p=2, dim=1)))
        loss_p = torch.log(loss)
    return torch.sum(loss_p)

def oe_loss_hyperbolic(x_i, x_j, x_k, loss_number, device, dims, delta=1):
    # x:  batch_size * embedding_dims
    # G:  embedding_dims * embedding_dims
    G = torch.eye(x_i.shape[1]).to(device)
    G[0][0] = (G[0][0]*-1)
    d1 = torch.matmul(
        torch.matmul(G, torch.reshape(x_i, (x_i.shape[0],x_i.shape[1],1))).reshape(x_i.shape[0],1,x_i.shape[1]),
        x_j.reshape(x_j.shape[0],x_j.shape[1],1)
        ).reshape(x_j.shape[0]).float()
    d1 = torch.log(d1+torch.sqrt(d1*d1 +1))

    d2 = torch.matmul(
            torch.matmul(G, torch.reshape(x_i, (x_i.shape[0],x_i.shape[1],1))).reshape(x_i.shape[0],1,x_i.shape[1]), 
            x_k.reshape(x_k.shape[0],x_k.shape[1],1)
            ).reshape(x_k.shape[0]).float()
    d2 = torch.log(d2+torch.sqrt(d2*d2 +1))

    if loss_number == 1: 
        loss = delta + d1 - d2
        # loss: torch.Size([1024])
        loss_p = loss[loss>0]
    elif loss_number == 2:
        loss = delta + d1 - d2
        loss = torch.pow(loss, exponent=2)
        loss_p = loss[loss>0]
    elif loss_number == 3:
        loss = torch.div(torch.exp(-d1*d1), torch.exp(-d1*d1) + torch.exp(-d2*d2))
        loss_p = -torch.log(loss)
    elif loss_number == 4:
        loss = torch.div((1+d1/(dims-1))**(dims/2), (1+d1/(dims-1))**(dims/2) + (1+d2/(dims-1))**(dims/2))
        loss_p = torch.log(loss)
    elif loss_number == 5:
        loss = torch.div(torch.exp(d1*d1), torch.exp(d1*d1) + torch.exp(d2*d2))
        loss_p = torch.log(loss)
    return torch.sum(loss_p)

def get_matrix(tmp_i,j, device):
    # [ 1-2y2-2z2 , 2xy-2wz , 2xz+2wy ]
    # [ 2xy+2wz , 1-2x2-2z2 , 2yz-2wx ]
    # [ 2xz-2wy , 2yz+2wx , 1-2x2-2y2 ]
    Pi = torch.eye(3).to(device)
    Pi[0][0] = 1.0-2*tmp_i[j][2]*tmp_i[j][2]-2*tmp_i[j][3]*tmp_i[j][3]
    Pi[0][1] = 2*tmp_i[j][1]*tmp_i[j][2]-2*tmp_i[j][0]*tmp_i[j][3]
    Pi[0][2] = 2*tmp_i[j][1]*tmp_i[j][3]+2*tmp_i[j][0]*tmp_i[j][2]
    Pi[1][0] = 2*tmp_i[j][1]*tmp_i[j][2]+2*tmp_i[j][0]*tmp_i[j][3]
    Pi[1][1] = 1.0-2*tmp_i[j][1]*tmp_i[j][1]-2*tmp_i[j][3]*tmp_i[j][3]
    Pi[1][2] = 2*tmp_i[j][2]*tmp_i[j][3]-2*tmp_i[j][0]*tmp_i[j][1]
    Pi[2][0] = 2*tmp_i[j][1]*tmp_i[j][3]-2*tmp_i[j][0]*tmp_i[j][2]
    Pi[2][1] = 2*tmp_i[j][2]*tmp_i[j][3]+2*tmp_i[j][0]*tmp_i[j][1]
    Pi[2][2] = 1.0-2*tmp_i[j][1]*tmp_i[j][1]-2*tmp_i[j][2]*tmp_i[j][2]
    return Pi

def get_matrix_ori(tmp_i,j, device):
    # w  z  -y x  
    # -z w  -x -y
    # y  x  w -z
    # -x y  z w
    Pi = torch.eye(4).to(device)
    Pi[0][0] = tmp_i[j][0]
    Pi[0][1] = tmp_i[j][3]
    Pi[0][2] = -tmp_i[j][2]
    Pi[0][3] = tmp_i[j][1]
    Pi[1][0] = -tmp_i[j][3]
    Pi[1][1] = tmp_i[j][0]
    Pi[1][2] = -tmp_i[j][1]
    Pi[1][3] = -tmp_i[j][2]
    Pi[2][0] = tmp_i[j][2]
    Pi[2][1] = tmp_i[j][1]
    Pi[2][2] = tmp_i[j][0]
    Pi[2][3] = -tmp_i[j][3]
    Pi[3][0] = -tmp_i[j][1]
    Pi[3][1] = tmp_i[j][2]
    Pi[3][2] = tmp_i[j][3]
    Pi[3][3] = tmp_i[j][0]
    return Pi

def get_matrix_1(tmp_i,j, device):
    # [  , 2xy-2wz , 2xz+2wy ]
    # [ 2xy+2wz , , 2yz-2wx ]
    # [ 2xz-2wy , 2yz+2wx , ]
    Pi = torch.eye(3).to(device)
    Pi[0][0] = tmp_i[j][0]*tmp_i[j][0]+tmp_i[j][1]*tmp_i[j][1]-tmp_i[j][2]*tmp_i[j][2]-tmp_i[j][3]*tmp_i[j][3]
    Pi[0][1] = 2*tmp_i[j][1]*tmp_i[j][2]-2*tmp_i[j][0]*tmp_i[j][3]
    Pi[0][2] = 2*tmp_i[j][1]*tmp_i[j][3]+2*tmp_i[j][0]*tmp_i[j][2]
    Pi[1][0] = 2*tmp_i[j][1]*tmp_i[j][2]+2*tmp_i[j][0]*tmp_i[j][3]
    Pi[1][1] = tmp_i[j][0]*tmp_i[j][0]-tmp_i[j][1]*tmp_i[j][1]+tmp_i[j][2]*tmp_i[j][2]-tmp_i[j][3]*tmp_i[j][3]
    Pi[1][2] = 2*tmp_i[j][2]*tmp_i[j][3]-2*tmp_i[j][0]*tmp_i[j][1]
    Pi[2][0] = 2*tmp_i[j][1]*tmp_i[j][3]-2*tmp_i[j][0]*tmp_i[j][2]
    Pi[2][1] = 2*tmp_i[j][2]*tmp_i[j][3]+2*tmp_i[j][0]*tmp_i[j][1]
    Pi[2][2] = tmp_i[j][0]*tmp_i[j][0]-tmp_i[j][1]*tmp_i[j][1]-tmp_i[j][2]*tmp_i[j][2]+tmp_i[j][3]*tmp_i[j][3]
    return Pi

def os_loss_quaternion(x_i, x_j, x_k, loss_number,device, delta=1):
    # 首先讲embedding: batch_size*embedding_dim----->batch_size*dims*4
    # 对于dims*4: 
    # for dims:
    #   get matrix
    #   cal theta
    loss = 0
    for i in range(int(x_i.shape[0])):
        tmp_i = x_i[i].reshape(int(x_i.shape[1]/4),4)
        tmp_j = x_j[i].reshape(int(x_j.shape[1]/4),4)
        tmp_k = x_k[i].reshape(int(x_k.shape[1]/4),4)
        d1 = 0
        d2 = 0
        for j in range(int(x_i.shape[1]/4)):
            Pi = get_matrix(tmp_i,j)
            Pj = get_matrix(tmp_j,j)
            Pk = get_matrix(tmp_k,j)
            d1 += torch.trace(torch.mm(Pi,Pj))
            d2 += torch.trace(torch.mm(Pi,Pk))
        d1 = torch.sqrt(3-d1)
        d2 = torch.sqrt(3-d2)
        if delta+d1-d2>0:
            loss += delta+d1-d2
    return loss

def os_loss_quaternion_1(x_i, x_j, x_k, loss_number, device, dims, delta=1):
    # 首先讲embedding: batch_size*embedding_dim----->batch_size*dims*4
    # 对于dims*4: 
    # for dims:
    #   get matrix
    #   cal theta
    loss = 0
    for i in range(int(x_i.shape[0])):
        # 对每个向量做操作
        tmp_i = x_i[i].reshape(int(x_i.shape[1]/4),4)
        tmp_j = x_j[i].reshape(int(x_j.shape[1]/4),4)
        tmp_k = x_k[i].reshape(int(x_k.shape[1]/4),4)
        d1 = 0
        d2 = 0
        for j in range(int(x_i.shape[1]/4)):
            # 对向量的每4维做运算
            # 得到矩阵
            Pi = get_matrix_1(tmp_i,j, device)
            Pj = get_matrix_1(tmp_j,j, device)
            Pk = get_matrix_1(tmp_k,j, device)
            d1 += torch.trace(torch.mm(Pi,Pj))
            d2 += torch.trace(torch.mm(Pi,Pk))
        # 计算距离
        d1 = torch.sqrt(3-d1)
        d2 = torch.sqrt(3-d2)
        # 选择不同的loss function
        if loss_number == 1:
            if delta+d1-d2>0:
                loss += delta+d1-d2
        elif loss_number == 2:
            if delta+d1-d2>0:
                loss += (delta+d1-d2)*(delta+d1-d2)   
        elif loss_number == 3:
            tmp = torch.div(torch.exp(-d1*d1), torch.exp(-d1*d1)+torch.exp(-d2*d2))
            if tmp > 0:
                loss = loss - torch.log(tmp)
        elif loss_number == 4:
            tmp = torch.div( (delta+d1*d1/(dims-1))**(dims/2), 
                            (delta+d1*d1/(dims-1))**(dims/2) + (delta+d2*d2/(dims-1))**(dims/2))
            if tmp > 0:
                loss = loss + torch.log(tmp)
        elif loss_number == 5:
            tmp = torch.div(torch.exp(d1*d1), torch.exp(d1*d1)+torch.exp(d2*d2))
            if tmp > 0:
                loss = loss + torch.log(tmp)
    return loss

def os_loss_quaternion_ori(x_i, x_j, x_k, loss_number, device, dims, delta=1):
    loss = 0
    for i in range(int(x_i.shape[0])):
        tmp_i = x_i[i].reshape(int(x_i.shape[1]/4),4)
        tmp_j = x_j[i].reshape(int(x_j.shape[1]/4),4)
        tmp_k = x_k[i].reshape(int(x_k.shape[1]/4),4)
        d1 = 0
        d2 = 0
        for j in range(int(x_i.shape[1]/4)):
            Pi = get_matrix_ori(tmp_i,j, device)
            Pj = get_matrix_ori(tmp_j,j, device)
            Pk = get_matrix_ori(tmp_k,j, device)
            d1 += torch.trace(torch.mm(Pi,Pj))
            d2 += torch.trace(torch.mm(Pi,Pk))
        d1 = torch.sqrt(4-d1)
        d2 = torch.sqrt(4-d2)
        if loss_number == 1:
            if delta+d1-d2>0:
                loss += delta+d1-d2
        elif loss_number == 2:
            if delta+d1-d2>0:
                loss += (delta+d1-d2)*(delta+d1-d2)   
        elif loss_number == 3:
            tmp = torch.div(torch.exp(-d1*d1), torch.exp(-d1*d1)+torch.exp(-d2*d2))
            if tmp > 0:
                loss = loss - torch.log(tmp)
        elif loss_number == 4:
            tmp = torch.div( (delta+d1*d1/(dims-1))**(dims/2), 
                            (delta+d1*d1/(dims-1))**(dims/2) + (delta+d2*d2/(dims-1))**(dims/2))
            if tmp > 0:
                loss = loss + torch.log(tmp)
        elif loss_number == 5:
            tmp = torch.div(torch.exp(d1*d1), torch.exp(d1*d1)+torch.exp(d2*d2))
            if tmp > 0:
                loss = loss + torch.log(tmp)
    return loss

def triplet_error_torch(emb, triplets, loss_type, device):
    if loss_type == 'Euclid':
        return(triplet_error_Euclid(emb, triplets))
    if loss_type == 'hyperbolic':
        return(triplet_error_hyperbolic(emb, triplets, device))
    if loss_type == 'quaternion':
        return(triplet_error_quaternion(emb, triplets))
    if loss_type == 'quaternion_1':
        return(triplet_error_quaternion_1(emb, triplets, device))
    # if loss_type == 'daul':
    #     return(triplet_error_daul(emb,triplets))
    if loss_type == 'quaternion_ori':
        return(triplet_error_quaternion_ori(emb, triplets, device))

def triplet_error_Euclid(emb, trips):
    """
    Description: Given the embeddings and triplet constraints, compute the triplet error.
    :param emb:
    :param trips:
    :return:
    """
    d1 = torch.sum((emb[trips[:, 0], :] - emb[trips[:, 1], :]) ** 2, dim=1)
    d2 = torch.sum((emb[trips[:, 0], :] - emb[trips[:, 2], :]) ** 2, dim=1)
    # torch.Size([6375])
    error_list = d2 < d1
    ratio = sum(error_list) / float(trips.shape[0])
    return ratio, error_list

def triplet_error_hyperbolic(emb, triplets, device):
    total_triplets = triplets.shape[0]

    x_i = emb[triplets[:, 0], :]
    x_j = emb[triplets[:, 1], :]
    x_k = emb[triplets[:, 2], :]
    G = torch.eye(x_i.shape[1]).to(device)
    G[0][0] = (G[0][0]*-1)
    d1 = torch.matmul(
            torch.matmul(G, torch.reshape(x_i, (x_i.shape[0],x_i.shape[1],1))).reshape(x_i.shape[0],1,x_i.shape[1]),
            x_j.reshape(x_j.shape[0],x_j.shape[1],1)
            ).reshape(x_j.shape[0]).float()
    
    d1 = torch.log(d1+torch.sqrt(d1*d1 +1))

    d2 = torch.matmul(
            torch.matmul(G, torch.reshape(x_i, (x_i.shape[0],x_i.shape[1],1))).reshape(x_i.shape[0],1,x_i.shape[1]), 
            x_k.reshape(x_k.shape[0],x_k.shape[1],1)
            ).reshape(x_k.shape[0]).float()
    d2 = torch.log(d2+torch.sqrt(d2*d2 +1))
    error_list = d2 < d1
    ratio = sum(error_list)/float(triplets.shape[0])
    return ratio, error_list

def triplet_error_quaternion(emb, triplets):
    """
    Description: Given the embeddings and triplet constraints, compute the triplet error.
    :param emb:
    :param trips:
    :return:
    """
    total_triplets = triplets.shape[0]
    number = 0
    x_i = emb[triplets[:, 0], :]
    x_j = emb[triplets[:, 1], :]
    x_k = emb[triplets[:, 2], :]
    for i in range(int(x_i.shape[0])):
        tmp_i = x_i[i].reshape(int(x_i.shape[1]/4),4)
        tmp_j = x_j[i].reshape(int(x_j.shape[1]/4),4)
        tmp_k = x_k[i].reshape(int(x_k.shape[1]/4),4)
        d1 = 0
        d2 = 0
        for j in range(int(x_i.shape[1]/4)):
            Pi = get_matrix(tmp_i,j)
            Pj = get_matrix(tmp_j,j)
            Pk = get_matrix(tmp_k,j)
            d1 += torch.trace(torch.mm(Pi,Pj))
            d2 += torch.trace(torch.mm(Pi,Pk))
        d1 = torch.sqrt(3-d1)
        d2 = torch.sqrt(3-d2)
        if d1 < d2:
            number += 1 
    ratio = number / float(triplets.shape[0])
    return ratio # , error_list

def triplet_error_quaternion_1(emb, triplets, device):
    """
    Description: Given the embeddings and triplet constraints, compute the triplet error.
    :param emb:
    :param trips:
    :return:
    """
    total_triplets = triplets.shape[0]
    number = 0
    x_i = emb[triplets[:, 0], :]
    x_j = emb[triplets[:, 1], :]
    x_k = emb[triplets[:, 2], :]
    for i in range(int(x_i.shape[0])):
        tmp_i = x_i[i].reshape(int(x_i.shape[1]/4),4)
        tmp_j = x_j[i].reshape(int(x_j.shape[1]/4),4)
        tmp_k = x_k[i].reshape(int(x_k.shape[1]/4),4)
        d1 = 0
        d2 = 0
        for j in range(int(x_i.shape[1]/4)):
            Pi = get_matrix_1(tmp_i,j, device)
            Pj = get_matrix_1(tmp_j,j, device)
            Pk = get_matrix_1(tmp_k,j, device)
            d1 += torch.trace(torch.mm(Pi,Pj))
            d2 += torch.trace(torch.mm(Pi,Pk))
        d1 = torch.sqrt(3-d1)
        d2 = torch.sqrt(3-d2)
        if d1 <= d2:
            number += 1 
    ratio = number / float(triplets.shape[0])
    return ratio # , error_list

def triplet_error_quaternion_ori(emb, triplets, device):
    total_triplets = triplets.shape[0]
    number = 0
    x_i = emb[triplets[:, 0], :]
    x_j = emb[triplets[:, 1], :]
    x_k = emb[triplets[:, 2], :]
    for i in range(int(x_i.shape[0])):
        tmp_i = x_i[i].reshape(int(x_i.shape[1]/4),4)
        tmp_j = x_j[i].reshape(int(x_j.shape[1]/4),4)
        tmp_k = x_k[i].reshape(int(x_k.shape[1]/4),4)
        d1 = 0
        d2 = 0
        for j in range(int(x_i.shape[1]/4)):
            Pi = get_matrix_ori(tmp_i,j, device)
            Pj = get_matrix_ori(tmp_j,j, device)
            Pk = get_matrix_ori(tmp_k,j, device)
            d1 += torch.trace(torch.mm(Pi,Pj))
            d2 += torch.trace(torch.mm(Pi,Pk))
        d1 = torch.sqrt(4-d1)
        d2 = torch.sqrt(4-d2)
        if d1 < d2:
            number += 1 
    ratio = number / float(triplets.shape[0])
    return ratio # , error_list

class MusicDataset(Dataset):
    def __init__(self,filename='',test_ratio=0.3, split_flag=False):
        path = "/home/wenzheng/Data/Music/"
        # 需要划分训练集和测试集
        if split_flag:  
            # get raw data A,B,C
            triplets = scipy.io.loadmat(filename)['triplets']        
            triplets -= 1 
            shuffled_indices=np.random.permutation(len(triplets))

            test_set_size=int(len(triplets)*test_ratio)
            test_indices = shuffled_indices[:test_set_size]
            train_indices = shuffled_indices[test_set_size:]
            self.train_triplets = triplets[train_indices]
            self.test_triplets = triplets[test_indices]

            # 保存文件
            train = pd.DataFrame(self.train_triplets)
            train.to_csv(path+'train_music.csv',index=None)
            test = pd.DataFrame(self.test_triplets)
            test.to_csv(path+'test_music.csv',index=None)

        # 不需要划分训练集和测试集
        else:
            self.train_triplets = pd.read_csv(path+'train_music.csv').values
            self.test_triplets = pd.read_csv(path+'test_music.csv').values

class NewDataset(Dataset):
   def __init__(self,filename='',test_ratio=0.3, split_flag=False):
        path = "/home/wenzheng/Data/ar/"
        # 需要划分训练集和测试集
        if split_flag:  
            # get raw data A,B,C
            print('do not hava the ordinal dataset!')

        # 不需要划分训练集和测试集
        else:
            self.train_triplets = pd.read_csv(path+'train_new.csv').values
            self.test_triplets = pd.read_csv(path+'test_new.csv').values

class New200Dataset(Dataset):
   def __init__(self,filename='',test_ratio=0.3, split_flag=False):
        path = "/home/wenzheng/Data/200/"
        # 需要划分训练集和测试集
        if split_flag:  
            print('do not hava the ordinal dataset!')
        # 不需要划分训练集和测试集
        else:
            self.train_triplets = pd.read_csv(path+'train_200.csv').values
            self.test_triplets = pd.read_csv(path+'test_200.csv').values

class New300Dataset(Dataset):
   def __init__(self,filename='',test_ratio=0.3, split_flag=False):
        path = "/home/wenzheng/Data/300/"
        # 需要划分训练集和测试集
        if split_flag:  
            print('do not hava the ordinal dataset!')
        # 不需要划分训练集和测试集
        else:
            self.train_triplets = pd.read_csv(path+'train_300.csv').values
            self.test_triplets = pd.read_csv(path+'test_300.csv').values

class FoodDataset(Dataset):
    def __init__(self,filename='/home/wenzheng/Data/Food/all-triplets.csv',test_ratio=0.3, split_flag=False):
        path = "/home/wenzheng/Data/Food/"
        # 需要划分训练集和测试集
        if split_flag:  
            # get raw data A,B,C
            triplets = pd.read_csv(filename, header=None, sep=';') 
            imgs = triplets[0].unique()
            img_No = dict()
            number = 0
            for img in imgs:
                if img not in img_No:
                    img_No[img] = number
                    number = number + 1
            imgs = triplets[1].unique()
            for img in imgs:
                if img not in img_No:
                    img_No[img] = number
                    number = number + 1
            imgs = triplets[2].unique()
            for img in imgs:
                if img not in img_No:
                    img_No[img] = number
                    number = number + 1
            triplets.iloc[:, 0] = triplets.iloc[:, 0].apply(lambda x: img_No[x])
            triplets.iloc[:, 1] = triplets.iloc[:, 1].apply(lambda x: img_No[x])
            triplets.iloc[:, 2] = triplets.iloc[:, 2].apply(lambda x: img_No[x])
            # print(triplets)
            tol_triplets = 1000
            shuffled_indices=np.random.permutation(len(triplets))

            test_set_size=int(tol_triplets*test_ratio)
            test_indices = shuffled_indices[:test_set_size]
            train_indices = shuffled_indices[test_set_size:tol_triplets]
            self.train_triplets = triplets.loc[train_indices]
            self.test_triplets = triplets.loc[test_indices]

            # 保存文件
            train = pd.DataFrame(self.train_triplets)
            train.to_csv(path+'train_food.csv',index=None)
            test = pd.DataFrame(self.test_triplets)
            test.to_csv(path+'test_food.csv',index=None)
        # 不需要划分训练集和测试集
        else:
            self.train_triplets = pd.read_csv(path+'train_food.csv').values
            self.test_triplets = pd.read_csv(path+'test_food.csv').values

class BirdsDataset(Dataset):
    def __init__(self,filename='E:/2-vipl/1-paper/3/data/segmentations/',test_ratio=0.3, split_flag=False):
        path = "/home/wenzheng/Data/bird/"
        # 需要划分训练集和测试集
        if split_flag:  
            bird_list = ['001.Black_footed_Albatross',
            '002.Laysan_Albatross',
            '003.Sooty_Albatross',
            '004.Groove_billed_Ani',
            '005.Crested_Auklet',
            '006.Least_Auklet',
            '007.Parakeet_Auklet',
            '008.Rhinoceros_Auklet']
            # get raw data A,B,C
            img_No = dict()
            number = 0
            triplets = pd.DataFrame(columns={'0','1','2'})
            for f in bird_list:
                tmp_path = filename + f
                if os.path.exists(tmp_path):
                    for name in os.listdir(tmp_path):
                        if name not in img_No:
                            img_No[name] = number
                            number += 1
            np.random.seed(1234)
            # 8 * 10 * 10 * 7 * 10 
            for f in tqdm(bird_list):
                tmp_path = filename + f
                if os.path.exists(tmp_path):                
                    for name in os.listdir(tmp_path):
                        for name1 in os.listdir(tmp_path):
                            # p = np.random.rand(1)[0]
                            # if p > 0.1:
                            #     continue
                            if name == name1:
                                continue
                            for f1 in bird_list:
                                if f1 != f:
                                    tmp = filename + f1 
                                    for name2 in os.listdir(tmp):
                                        # p = np.random.rand(1)[0]
                                        # if p > 0.1:
                                        #     continue
                                        triplets = triplets.append({'0':img_No[name], '1':img_No[name1], '2':img_No[name2]}, ignore_index=True)
            print(triplets)
            print(len(triplets))

            tol_triplets = 5000
            shuffled_indices=np.random.permutation(len(triplets))

            test_set_size=int(tol_triplets*test_ratio)
            test_indices = shuffled_indices[:test_set_size]
            train_indices = shuffled_indices[test_set_size:tol_triplets]
            self.train_triplets = triplets.loc[train_indices]
            self.test_triplets = triplets.loc[test_indices]

            # 保存文件
            train = pd.DataFrame(self.train_triplets)
            train.to_csv(path+'train_bird.csv',index=None)
            test = pd.DataFrame(self.test_triplets)
            test.to_csv(path+'test_bird.csv',index=None)

        # 不需要划分训练集和测试集
        else:
            self.train_triplets = pd.read_csv(path+'train_bird.csv').values
            self.test_triplets = pd.read_csv(path+'test_bird.csv').values

if __name__=='__main__':
    # 选择数据集
    dataset_name = sys.argv[1]
    if dataset_name == 'music':
        data = MusicDataset(filename='/home/wenzheng/Data/Music/music_triplets.mat',split_flag=False)
    elif dataset_name == '100':
        data = NewDataset(filename='', split_flag=False)
    elif dataset_name == 'food':
        data = FoodDataset(filename='/home/wenzheng/Data/Food/all-triplets.csv', split_flag=False)
    elif dataset_name == 'bird':
        data = BirdsDataset(filename='', split_flag=False)
    elif dataset_name == '200':
        data = New200Dataset(filename='', split_flag=False)
    elif dataset_name == '300':
        data = New300Dataset(filename='', split_flag=False)
    
    # 日志
    log_dir = './logs/'
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging_path = os.path.join(log_dir, '2.log')
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    
    # 选择方法的类型
    loss_type = sys.argv[2] # hyperbolic, Euclid, quaternion 
    # 向量的维度
    dims = int(sys.argv[3])
    # 设备名称
    device_number = str(sys.argv[4])
    # loss function 类型
    loss_number = int(sys.argv[5])
    
    device = torch.device("cuda:"+device_number if torch.cuda.is_available() else "cpu")

    logger.info('loss_type: ' + str(loss_type))
    logger.info('dims: ' + str(dims))
    logger.info('Computing OE...')
    # 进行20次实验，得到mean 和 std
    mean_error = dict()
    std_error = dict()
    epochs=200
    for i in range(0, epochs+1,5):
        mean_error[str(i)] = 0.0
        std_error[str(i)] = 0.0
    for i in range(5):
        embedding, triplet_error = oe(data=data, 
                        no_dims=dims,
                        epochs=epochs,
                        batch_size=512,
                        learning_rate=0.2,
                        error_change_threshold=0.0005,
                        loss_type=loss_type,
                        device=device,
                        loss_number=loss_number)
        logger.info('time: '+ str(i) + ' Error: ' + str(triplet_error[str(epochs)]))
        for tmp_i in range(0,epochs+1,5):
            if float(triplet_error[str(tmp_i)]) > 0:
                # 增量公式计算方差和均值
                N = i+1
                std_error[str(tmp_i)] = std_error[str(tmp_i)] + 1.0*(N-1)/N*(float(triplet_error[str(tmp_i)])-mean_error[str(tmp_i)])*(float(triplet_error[str(tmp_i)])-mean_error[str(tmp_i)])
                mean_error[str(tmp_i)] = mean_error[str(tmp_i)] + (float(triplet_error[str(tmp_i)]) - mean_error[str(tmp_i)])/N  
                logger.info('time: '+ str(i) + ' epochs: ' + str(tmp_i) +' mean: ' + str(mean_error[str(tmp_i)]) + ' std: '+ str(std_error[str(tmp_i)]))