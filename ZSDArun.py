import argparse
import os
import time
from myutils import *
from ZSDAmodel import ZSDA
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import random
torch.manual_seed(0)
# command line args
parser = argparse.ArgumentParser(description='Zero shot domain adaptation Classifier')

# required
parser.add_argument('--output-dir', required=True, type=str, default=None,
                    help='output directory for checkpoints and figures')

# optional
parser.add_argument('--batch-size', type=int, default=500,
                    help='batch size (of datasets) for training (default: 512)')
parser.add_argument('--sample-size', type=int, default=1000,
                    help='number of samples per domain (default: 1000)')
parser.add_argument('--n-features', type=int, default=256,
                    help='number of features per sample (default: 256)')
parser.add_argument('--z-dim', type=int, default=6,
                    help='dimension of z variables (default: 2)')
parser.add_argument('--n_c', type=int, default=10,
                    help='the number class of x (default: 10)')
parser.add_argument('--J_dim', type=int, default=100,
                    help='dimension of z variables (default: 10)')
parser.add_argument('--hidden-dim', type=int, default=100,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 100)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all learnable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs for training (default: 300)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')
args = parser.parse_args()
assert args.output_dir is not None
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")


def run(model, criterion, optimizer, datasets):
     
    X_list, y_list, X_test, y_test, domain = datasets
    X_list, y_list,X_test,y_test = torch.tensor(X_list).float(), torch.tensor(y_list).long(),torch.tensor(X_test).float(),torch.tensor(y_test).long()
    
    r_dom = list(range(8))
    data_r_seet = list(range(1000))
    random.shuffle(data_r_seet)
    #val dataset
    X_val = X_list[:,data_r_seet[:200],:] 
    y_val = y_list[:,data_r_seet[:200]].flatten()  
    #train dataset
    X_list = X_list[:,data_r_seet[200:],:] 
    y_list = y_list[:,data_r_seet[200:]]
    #viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    #save_interval = args.epochs if args.save_interval == -1 else args.save_interval
    alpha = 1
    tbar = tqdm(range(args.epochs))
    
    r_dom = list(range(8))
    data_r_seet = list(range(800))
    random.shuffle(data_r_seet)
    vlb_list = []
    dom_data_seet = []
    losses = []
    # main training loop
    for epoch in tbar:
        """
        ドメインのデータセットをシャッフルさせながら選ぶ
        """
        # train step
        model.train()
        running_vlb = 0
        
        random.shuffle(r_dom)#DomainNumberをシャッフル
        #dom_data_seet = []
        #for i in domain:#各ドメインのデータの並び替え
        random.shuffle(data_r_seet)#DataNumberをシャッフル 
            #dom_data_seet.append(data_r_seet) 
            
        for i in r_dom:#Batch Learining
            #dom = i//2
            #if(i%2==0):
            #    seet = dom_data_seet[dom][:args.batch_size]
            #else:
            #    seet = dom_data_seet[dom][args.batch_size:]#number
            seet = data_r_seet[100*i:100*(1+i)]#100飛ばし
            
            X_batch = X_list[:,seet,:]#[domain:5, samples:100,  dim:256]
            y_batch = y_list[:,seet].flatten()#[5,100]=>[500]
       
            batch = (X_batch,y_batch)
            
            loss, vlb, loss_acc, z_outputs = model.step(batch, alpha, criterion , optimizer, clip_gradients=args.clip_gradients)
            vlb_list.append(vlb.data)
            z_mean, z_logvar = z_outputs
            print(z_mean,torch.exp(z_logvar))
            running_vlb += vlb
            losses.append(loss)#learning loss
            print(loss_acc)
        # reduce weight
        alpha *= 0.5

        #running_vlb /= X_batch.size(0)
        s = "VLB: {:.8f}".format(running_vlb)
        tbar.set_description(s)


        # show test set in context space at intervals
        if (epoch + 1) % 1 == 0:
            model.eval()
            contexts = []
            
            #validation
            X_val = Variable(X_val.cuda())#[domain : 5,samples : 200, dim:256]
            y_val = Variable(y_val.cuda())#[domain *samples : 1000 ]
            mean, logvar = model.inference_network(X_val)
            z = model.reparameterize_gaussian(mean, logvar)
            y_pred = model.observation_classifier(X_val, z)
            _, predicted = torch.max(y_pred,1)#予測ラベル
            correct = (predicted == y_val).sum().item()
            val_acc = (float(correct) / y_val.size(0))*100
            print("val_acc",val_acc)
            #test data
            inputs = Variable(X_test.view(1,args.sample_size,args.n_features).cuda(), volatile=True)
            y = Variable(y_test.cuda())
            mean, logvar = model.inference_network(inputs)#((z_mean, z_logvar),(y, y_pred))   
            z = model.reparameterize_gaussian(mean, logvar)
        
            
            y_pred = model.observation_classifier(inputs,z)
            _, predicted = torch.max(y_pred,1)#予測ラベル
            correct = (predicted == y).sum().item()
            val_acc = (float(correct) / y.size(0))*100
            print(mean,logvar)
            print("test_acc",val_acc)
            vlb_list.append(val_acc)# loss val
            contexts.append(mean.data.cpu().numpy())
            #print(contexts)
            # show coloured by distribution
            """
            path = args.output_dir + '/figures/' + time_stamp + '-{}.pdf'.format(epoch + 1)
            scatter_contexts(contexts, test_dataset.data['labels'],
                             test_dataset.data['distributions'], savepath=path)

            # show coloured by mean
            path = args.output_dir + '/figures/' + time_stamp \
                   + '-{}-mean.pdf'.format(epoch + 1)
            contexts_by_moment(contexts, moments=test_dataset.data['means'],
                               savepath=path)

            # show coloured by variance
            path = args.output_dir + '/figures/' + time_stamp \
                   + '-{}-variance.pdf'.format(epoch + 1)
            contexts_by_moment(contexts, moments=test_dataset.data['variances'],
                               savepath=path)
            """
    return losses, vlb_list

def main():
    src_domains, (X_test, y_test) = load_rotated_mnist()
    X_list = []
    y_list = []
    domain = []
    for d in range(0, len(src_domains)):
        X, y = src_domains[d]
        #n_labels = len(np.unique(y))  # 分類クラスの数 = 10
        #y = np.eye(n_labels)[y]       # one hot表現に変換
        X_list.append(X)
        y_list.append(y)#0,15,30,45,60 
        domain.append(d)
    
    #train_dataset = data.TensorDataset(torch.tensor(X_list),torch.tensor(y_list))
    #test_dataset = data.TensorDataset(torch.tensor(X_test),torch.tensor(y_test))
    datasets = (X_list, y_list,X_test,y_test,domain)
    """
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=1,
                                   shuffle=True, num_workers=0, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1,
                                  shuffle=False, num_workers=0, drop_last=True)
    loaders = (train_loader, test_loader)
    """
    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'n_features': args.n_features,
        'z_dim': args.z_dim,
        'hidden_dim': args.hidden_dim,
        'n_c': args.n_c,
        'J_dim': args.J_dim,
        'print_vars': args.print_vars,
        'n_domain': domain
    }
    model = ZSDA(**model_kwargs)
    model.cuda()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    losses,vlb_list = run(model, criterion ,optimizer, datasets)
    plt.plot(losses)
    plt.show()
    plt.plot(vlb_list)
    plt.show()

if __name__ == '__main__':
    main()
