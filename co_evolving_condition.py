import os
from absl import flags
# import torch
# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import tabular_dataload
from torch.utils.data import DataLoader
from models.tabular_unet import tabularUnet
# from diffusion_discrete import MultinomialDiffusion
# import evaluation
import logging
# import numpy as np
# import pandas as pd
from utils import *

def train(FLAGS):

    FLAGS = flags.FLAGS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load Datasets
    train, train_cont_data, train_dis_data, test, (transformer_con, transformer_dis, meta), con_idx, dis_idx = tabular_dataload.get_dataset(FLAGS)
    still_condition = FLAGS.still_condition
    print('train_dis_data', type(train_dis_data), train_dis_data.shape)
    train_iter_cont = DataLoader(train_cont_data, batch_size=FLAGS.training_batch_size)
    # train_iter_dis = DataLoader(train_dis_data, batch_size=FLAGS.training_batch_size)
    datalooper_train_cont = infiniteloop(train_iter_cont)
    # datalooper_train_dis = infiniteloop(train_iter_dis)

    num_numeric=[]
    for i in transformer_con.output_info:
        num_numeric.append(i[0])
    num_numeric = np.array(num_numeric)
    print('num_numeric',num_numeric)

    num_class=[]
    for i in transformer_dis.output_info:
        num_class.append(i[0])
    num_class = np.array(num_class)
    print('num_class', num_class)

    still_cond_used_for_sampling = None
    k = 0
    for i in range(len(num_class)):
        if i == still_condition:
            still_cond_used_for_sampling = train_dis_data[:,k:k+num_class[i]]

    
    # if meta['problem_type'] == 'binary_classification':
    #     metric = 'binary_f1'
    # elif meta['problem_type'] == 'regression': metric = "r2"
    # else: metric = 'macro_f1'
    
    # Condtinuous Diffusion Model Setup
    FLAGS.input_size = train_cont_data.shape[1]
    FLAGS.cond_size = [still_cond_used_for_sampling.shape[1]]
    FLAGS.output_size = train_cont_data.shape[1]
    FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_con.split(',')))
    FLAGS.nf =  FLAGS.nf_con
    model_cont = tabularUnet(FLAGS, 0)
    optim_cont = torch.optim.Adam(model_cont.parameters(), lr=FLAGS.lr_con)
    sched_cont = torch.optim.lr_scheduler.LambdaLR(optim_cont, lr_lambda=warmup_lr)
    trainer_cont = GaussianDiffusionTrainer(model_cont, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_cont, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.mean_type, FLAGS.var_type).to(device)

    # for i in range(len(num_class)):
    #     # Discrete Diffusion Model Setup
    #
    #     FLAGS.input_size[i] = train_dis_data_list[i].shape[1]
    #     FLAGS.cond_size[i] = [each_cond.shape[1] for each_cond in train_dis_con_data_list[i]]
    #     FLAGS.output_size[i] = train_dis_data_list[i].shape[1]
    #     FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_dis.split(',')))
    #     FLAGS.nf =  FLAGS.nf_dis
    #     model_dis_list[i] = tabularUnet(FLAGS, i)
    #     optim_dis_list[i] = torch.optim.Adam(model_dis_list[i].parameters(), lr=FLAGS.lr_dis)
    #     sched_dis_list[i] = torch.optim.lr_scheduler.LambdaLR(optim_dis_list[i], lr_lambda=warmup_lr)
    #     trainer_dis_list[i] = MultinomialDiffusion(num_class[i], train_dis_data_list[i].shape, model_dis_list[i], FLAGS, timesteps=FLAGS.T,loss_type='vb_stochastic').to(device)

    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer_cont)
        net_sampler = torch.nn.DataParallel(net_sampler)



    # # num_params_con = sum(p.numel() for p in model_con.parameters())
    # num_params_dis = sum(p.numel() for p in model_dis.parameters())
    # # logging.info('Continuous model params: %d' % (num_params_con))
    # logging.info('Discrete model params: %d' % (num_params_dis))

    scores_max_eval = -10

    total_steps_both = FLAGS.total_epochs_both * int(train.shape[0]/FLAGS.training_batch_size+1)   # 20000, training times
    print('total_steps_both', total_steps_both)
    sample_step = FLAGS.sample_step * int(train.shape[0]/FLAGS.training_batch_size+1)   # 2000, sample times
    logging.info("Total steps: %d" %total_steps_both)
    logging.info("Sample steps: %d" %sample_step)
    logging.info("Continuous: %d, %d" %(train_cont_data.shape[0], train_cont_data.shape[1]))
    # logging.info("Discrete: %d, %d"%(train_dis_data.shape[0], train_dis_data.shape[1]))
    #
    # Start Training
    if FLAGS.eval==False:
        epoch = 0
        train_iter_cont = DataLoader(train_cont_data, batch_size=FLAGS.training_batch_size)
        # train_iter_dis_list[i] = DataLoader(train_dis_data_list[i], batch_size=FLAGS.training_batch_size)
        datalooper_train_cont = infiniteloop(train_iter_cont)
        # datalooper_train_dis_list[i] = infiniteloop(train_iter_dis_list[i])
        still_cond = DataLoader(still_cond_used_for_sampling, batch_size=FLAGS.training_batch_size)
        # datalooper_train_con = infiniteloop(train_iter_con)
        datalooper_still_cond = infiniteloop(still_cond)


        writer = SummaryWriter(FLAGS.logdir)
        writer.flush()
        for step in range(total_steps_both):
            model_cont.train()

            x_0_cont = next(datalooper_train_cont).to(device)
            x_cond = next(datalooper_still_cond).to(device)

                # ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
                # con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, transformer_dis, FLAGS)
            cont_loss = training_with(x_0_cont,trainer_cont, trainer_cont, FLAGS, x_cond)
            # loss_con = con_loss + FLAGS.lambda_con * con_loss_ns
            # loss_dis = dis_loss + FLAGS.lambda_dis * dis_loss_ns

            loss_cont = cont_loss

            optim_cont.zero_grad()
            loss_cont.backward()
            torch.nn.utils.clip_grad_norm_(model_cont.parameters(), FLAGS.grad_clip)
            optim_cont.step()
            sched_cont.step()

            # log
            writer.add_scalar('loss_continuous', cont_loss, step)
            # writer.add_scalar('loss_discrete', dis_loss_list[i], step)
            # writer.add_scalar('loss_continuous_ns', con_loss_ns, step)
            # writer.add_scalar('loss_discrete_ns', dis_loss_ns, step)
            # writer.add_scalar('total_continuous', loss_con, step)
            # writer.add_scalar('total_discrete', loss_dis, step)
            # model_dis_list[i].train(mode=False)

            if (step+1) % int(train.shape[0]/FLAGS.training_batch_size+1) == 0:

                # logging.info(f"Epoch :{epoch}, diffusion continuous loss: {con_loss:.3f}, discrete loss: {dis_loss:.3f}")
                # logging.info(f"Epoch :{epoch}, CL continuous loss: {con_loss_ns:.3f}, discrete loss: {dis_loss_ns:.3f}")
                # logging.info(f"Epoch :{epoch}, Total continuous loss: {loss_con:.3f}, discrete loss: {loss_dis:.3f}")
                logging.info(f"Epoch :{epoch}, continuous loss: {cont_loss:.6f}")
                epoch +=1

            if step > 0 and sample_step > 0 and step % sample_step == 0 or step==(total_steps_both-1):
                print('llll')
                model_cont.eval()
                with torch.no_grad():
                    x_T_cont = torch.randn(train_cont_data.shape[0], train_cont_data.shape[1]).to(device)
                    x_cont = sampling_with(x_T_cont, net_sampler, transformer_con, FLAGS, still_cond_used_for_sampling)
                sample_cont = transformer_con.inverse_transform(x_cont.detach().cpu().numpy())
                sample_dis = transformer_dis.inverse_transform(still_cond_used_for_sampling)
                # print('sample_dis',sample_dis.shape)
                sample = np.zeros([train_cont_data.shape[0], len(con_idx + dis_idx)])
                for i in range(len(con_idx)):
                    sample[:, con_idx[i]] = sample_cont[:, i]
                for i in range(len(dis_idx)):
                    sample[:, dis_idx[i]] = sample_dis[:, i]
                sample_pd = pd.DataFrame(sample).dropna()
                # scores, std, param = evaluation.compute_scores(train=train, test = None, synthesized_data=[sample], metadata=meta, eval=None)
                # div_mean, div_std = evaluation.compute_diversity(train=train, fake=[sample])
                # scores['coverage'] = div_mean['coverage']
                # std['coverage'] = div_std['coverage']
                # scores['density'] = div_mean['density']
                # std['density'] = div_std['density']
                # f1 = scores[metric]
                # logging.info(f"---------Epoch {epoch} Evaluation----------")
                # logging.info(scores)
                # logging.info(std)

                # if scores_max_eval < torch.tensor(f1):
                #     scores_max_eval = torch.tensor(f1)
                if True:
                    logging.info(f"Save model!")
                    ckpt = {
                        'model_con': model_cont.state_dict(),
                        'sched_con': sched_cont.state_dict(),
                        'optim_con': optim_cont.state_dict(),
                        'step': step,
                        'sample': sample,
                        # 'ml_param': param
                    }
                        # 'ml_param': param

                    torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
        logging.info(f"Evaluation best : {scores_max_eval}")

        #final test
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        model_cont.load_state_dict(ckpt['model_con'])
        model_cont.eval()
        # fake_sample=[]
        for ii in range(1):
            logging.info(f"sampling {ii}")
            with torch.no_grad():
                x_T_cont = torch.randn(train_cont_data.shape[0], train_cont_data.shape[1]).to(device)
                # log_x_T_dis_list[i] = log_sample_categorical(torch.zeros(train_dis_data_list[i].shape, device=device), num_class[i]).to(device)
                x_cont= sampling_with(x_T_cont, net_sampler, transformer_con, FLAGS, still_cond_used_for_sampling)
            sample_cont = transformer_con.inverse_transform(x_cont.detach().cpu().numpy())
            sample_dis = transformer_dis.inverse_transform(still_cond_used_for_sampling)
            # print('sample_dis',sample_dis.shape)
            sample = np.zeros([train_cont_data.shape[0], len(con_idx + dis_idx)])
            for i in range(len(con_idx)):
                sample[:,con_idx[i]]=sample_cont[:,i]
            for i in range(len(dis_idx)):
                sample[:,dis_idx[i]]=sample_dis[:,i]
            sample_pd = pd.DataFrame(sample).dropna()
            sample_pd.to_csv(os.path.join(FLAGS.logdir, f'test_sample_{ii}.csv'), index=False)
            # sample = np.array()
            # sample = np.zeros([train_dis_data.shape[0], len(dis_idx)])
            # for i in range(len(con_idx)):
            #     sample[:,con_idx[i]]=sample_con[:,i]
            # for i in range(len(dis_idx)):
            #     sample[:,dis_idx[i]]=sample_dis[:,i]
            # fake_sample.append(sample)
        # scores, std = evaluation.compute_scores(train=train, test = test, synthesized_data=fake_sample, metadata=meta, eval=ckpt['ml_param'])
        # div_mean, div_std = evaluation.compute_diversity(train=train, fake=fake_sample)
        # scores['coverage'] = div_mean['coverage']
        # std['coverage'] = div_std['coverage']
        # scores['density'] = div_mean['density']
        # std['density'] = div_std['density']
        # logging.info(f"---------Test----------")
        # logging.info(scores)
        # logging.info(std)

    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'),  map_location=torch.device('cpu') )
        model_cont.load_state_dict(ckpt['model_con'])
        model_cont.eval()
        # fake_sample=[]
        for ii in range(2):
            logging.info(f"sampling {ii}")
            with torch.no_grad():
                x_T_cont = torch.randn(train_cont_data.shape[0], train_cont_data.shape[1]).to(device)
                # log_x_T_dis_list[i] = log_sample_categorical(torch.zeros(train_dis_data_list[i].shape, device=device), num_class[i]).to(device)
                x_cont= sampling_with(x_T_cont, net_sampler, transformer_con, FLAGS, still_cond_used_for_sampling)
            sample_cont = transformer_con.inverse_transform(x_cont.detach().cpu().numpy())
            sample_dis = transformer_dis.inverse_transform(still_cond_used_for_sampling)
            # print('sample_dis',sample_dis.shape)
            sample = np.zeros([train_cont_data.shape[0], len(con_idx + dis_idx)])
            for i in range(len(con_idx)):
                sample[:,con_idx[i]]=sample_cont[:,i]
            for i in range(len(dis_idx)):
                sample[:,dis_idx[i]]=sample_dis[:,i]
            sample_pd = pd.DataFrame(sample).dropna()
            sample_pd.to_csv(os.path.join(FLAGS.logdir, f'test_sample_{ii}.csv'), index=False)
        # scores, std = evaluation.compute_scores(train=train, test = test, synthesized_data=fake_sample, metadata=meta, eval=ckpt['ml_param'])
        # div_mean, div_std = evaluation.compute_diversity(train=train, fake=fake_sample)
        # scores['coverage'] = div_mean['coverage']
        # std['coverage'] = div_std['coverage']
        # scores['density'] = div_mean['density']
        # std['density'] = div_std['density']
        # logging.info(f"---------Test----------")
        # logging.info(scores)
        # logging.info(std)