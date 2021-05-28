'''
Author: Jaw-Yuan, Chang 
'''
import torch
import argparse
from pathlib import Path
from utils import * 
from model import *
import torchvision.datasets as dset
import pickle
import torchvision.utils as vutils
def argument_parser():
    parser = argparse.ArgumentParser(
        description='Generate images')
    parser.add_argument('--gpu',
                        type=int,
                        help='GPU idx to run',
                        default=0)
    parser.add_argument('--save_dir',
                        type=str,
                        help='Write directory',
                        default='output')
    parser.add_argument('--lr_g',
                        type=float,
                        help='learning rate of generator',
                        default=1e-4)
    parser.add_argument('--lr_d',
                        type=float,
                        help='learning rate of discriminator',
                        default=4e-4)                        
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10000)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--n_c',
                        type=int,
                        help='Noise channel',
                        default=100)
    parser.add_argument('--w_d_g',
                        type=float,
                        help='Weight Decay of G',
                        default=0)
    parser.add_argument('--bg',
                        type=float,
                        help='Beta2 of G',
                        default=0.999)
    parser.add_argument('--bd',
                        type=float,
                        help='Beta2 of D',
                        default=0.999)    
    parser.add_argument('--w_d_d',
                        type=float,
                        help='Weight Decay of D',
                        default=0)
    parser.add_argument('--random_seed',
                        type=int,
                        default=87)
    parser.add_argument('--ck_pt',
                        default=None,
                        type= str,
                        help='checkpoint file' )
    parser.add_argument('--score_protion',
                        type=float,
                        help='score portion between real and fake',
                        default= 0.0)
    parser.add_argument('--traind',
                        type=int,
                        help='train triand times d in one epoch',
                        default= 1)
    return parser
    
def main(args):
    print(args)
    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    save_dir = 'results/' + args.save_dir
    m_s_dir = save_dir + '/model'
    i_s_dir = save_dir + '/image'

    Path(save_dir).mkdir(parents = True, exist_ok = True)
    Path(m_s_dir).mkdir(parents = True, exist_ok = True)
    Path(i_s_dir).mkdir(parents = True, exist_ok = True)
    
    with open(f'{save_dir}/opt.txt', 'w') as f:
        print(args, file = f)
        
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_g = Gmodel(noise_channel=args.n_c).to(device)
    model_d = Dmodel().to(device)
    optim_g = torch.optim.Adam(model_g.parameters(), lr=args.lr_g, betas=(0.9,args.bg), weight_decay=args.w_d_g)
    optim_d = torch.optim.Adam(model_d.parameters(), lr=args.lr_d, betas=(0.9,args.bd), weight_decay=args.w_d_d)
    trans = transform()
    real_data = dset.ImageFolder("./dset/", transform=trans)
    train_loader = torch.utils.data.DataLoader(real_data, 
                            batch_size = args.batch_size,
                            num_workers = 4,
                            shuffle = True,
                            pin_memory = True,
                            drop_last= True)

    if args.ck_pt is not None:
        ck_point = torch.load(args.ck_pt+'/model.pt')
        ck_ep = ck_point['epoch']
        model_g.load_state_dict(ck_point['G_state_dict'])
        model_d.load_state_dict(ck_point['D_state_dict'])
        optim_d.load_state_dict(ck_point['oD_state_dict'])
        optim_g.load_state_dict(ck_point['oG_state_dict'])
    else:
        ck_ep = 0
    
    criterion = nn.MSELoss(reduction = 'mean')
    loss_d_record = []
    loss_g_record = []

    for epoch in range(ck_ep, args.num_epochs):
        bloss_d = []
        bloss_g = []
        for i, data in enumerate(train_loader):
            for p in model_d.parameters():
                p.requires_grad = True
            for t_d in range(0,args.traind):
            ## Train D
                model_d.zero_grad()
                
                T_y = torch.ones(args.batch_size) - torch.rand(args.batch_size) * args.score_protion
                T_y = T_y.to(device)
                T_x = data[0].to(device)
                output = model_d(T_x) 
                loss_T_d = 0.5 * criterion(output.squeeze(), T_y)
                loss_T_d.backward()
            
                F_noise = sample_from_noise(args.batch_size, args.n_c)
                F_noise = F_noise.to(device)
                
                F_x = model_g(F_noise)
                F_y = torch.rand(args.batch_size) * args.score_protion
                F_y = F_y.to(device)
                output = model_d(F_x)
                loss_F_d = 0.5 * criterion(output.squeeze(), F_y)
                loss_F_d.backward()
            
                bloss_d.append(loss_F_d.item() + loss_T_d.item())
                Dx = loss_T_d.item() + loss_F_d.item()
                optim_d.step()
                
            ## Train G
            for p in model_d.parameters():
                p.requires_grad = False            
            model_g.zero_grad()
            T_y = torch.ones(args.batch_size) - torch.rand(args.batch_size) * args.score_protion
            T_y = T_y.to(device)
            F_noise = sample_from_noise(args.batch_size, args.n_c)
            F_noise = F_noise.to(device)
            F_x = model_g(F_noise)
            output = model_d(F_x)
            loss_G = 0.5 * criterion(output.squeeze(), T_y)
            loss_G.backward()
            optim_g.step()
            Dg = loss_G
            
            bloss_g.append(loss_G.item())
            
            if i % 10 == 0:
                print(f'D(x): {Dx}, D(G(z)): {Dg.item()}')
        ## Write loss record        

        loss_d_record.append(np.mean(bloss_d))
        loss_g_record.append(np.mean(bloss_g))
        with open(f'{save_dir}/loss_g.pkl', 'wb') as f:
            pickle.dump(loss_g_record, f)
        with open(f'{save_dir}/loss_d.pkl', 'wb') as f:
            pickle.dump(loss_d_record, f)
        ## Write check point
        if epoch % 100 == 0:
            torch.save({'G_state_dict': model_g.state_dict(),
                        'D_state_dict': model_d.state_dict(),
                        'epoch': epoch,
                        'oG_state_dict': optim_g.state_dict(),
                        'oD_state_dict': optim_d.state_dict()}, f'{m_s_dir}/model_{epoch}.pth')
            with torch.no_grad():
                F_noise = sample_from_noise(args.batch_size, args.n_c)
                F_noise = F_noise.to(device)
                sample_image = model_g(F_noise)
                vutils.save_image(sample_image, f'{i_s_dir}/img_{epoch}.png', normalize= True)

if __name__ == "__main__":
    parser = argument_parser()
    main(parser.parse_args())
