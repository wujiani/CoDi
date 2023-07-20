import os

import torch
from absl import flags
# import torch
# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import tabular_dataload
from torch.utils.data import DataLoader
from models.tabular_unet import tabularUnet
from diffusion_discrete import MultinomialDiffusion
# import evaluation
import logging
# import numpy as np
# import pandas as pd
from utils import *
from torchvision import transforms

def train(FLAGS):

    FLAGS = flags.FLAGS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Load Datasets
    train, train_cont_data, train_dis_data, test, attention_train_list, attention_test_list, (transformer_con, transformer_dis, meta), con_idx, dis_idx = tabular_dataload.get_dataset(FLAGS)
    # for att_i in attention_train
    attention_tensor_list = [torch.tensor(attention_train).to(device) for attention_train in attention_train_list]
    print('attention_tensor_list', attention_tensor_list[0].type(), attention_tensor_list[1].shape, attention_tensor_list[2].type() ,attention_tensor_list[3].shape, attention_tensor_list[4].type())

    FLAGS.still_condition = [int(each) for each in FLAGS.still_condition.split(',')]
    print('FLAGS.still_condition',FLAGS.still_condition)
    still_condition = FLAGS.still_condition
    # print('train_dis_data', type(train_dis_data), train_dis_data.shape)
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
    print('num_class',num_class)
    train_dis_data_list = []
    train_dis_con_data_list = []
    still_cond_used_for_sampling_list = []
    k = 0
    for i in range(len(num_class)):
        if i in still_condition:
            still_cond_used_for_sampling_list.append(train_dis_data[:,k:k+num_class[i]])
        train_dis_data_list.append(train_dis_data[:,k:k+num_class[i]])
        con_list = []
        kk = 0
        for j in range(len(num_class)):
            dis_data = train_dis_data[:,kk:kk+num_class[j]]
            kk+=num_class[j]
            if j != i:
                con_list.append(dis_data)
        k+=num_class[i]
        # con_list.append(train_cont_data) #0720
        train_dis_con_data_list.append(con_list)
    print('still_cond_used_for_sampling_list', still_cond_used_for_sampling_list)


    # if meta['problem_type'] == 'binary_classification':
    #     metric = 'binary_f1'
    # elif meta['problem_type'] == 'regression': metric = "r2"
    # else: metric = 'macro_f1'
    
    # Condtinuous Diffusion Model Setup
    FLAGS.cont_input_size = train_cont_data.shape[1]
    FLAGS.cont_cond_size = [each_cond.shape[1] for each_cond in train_dis_data_list]
    print('FLAGS.cont_cond_size',FLAGS.cont_cond_size)
    FLAGS.cont_output_size = train_cont_data.shape[1]
    FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_con.split(',')))
    FLAGS.nf =  FLAGS.nf_con
    model_cont = tabularUnet(FLAGS, '-1')
    optim_cont = torch.optim.Adam(model_cont.parameters(), lr=FLAGS.lr_con)
    sched_cont = torch.optim.lr_scheduler.LambdaLR(optim_cont, lr_lambda=warmup_lr)
    trainer_cont = GaussianDiffusionTrainer(model_cont, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_cont, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.mean_type, FLAGS.var_type).to(device)

    FLAGS.dis_input_size = [0]*len(num_class)
    FLAGS.dis_cond_size = [0]*len(num_class)
    FLAGS.dis_output_size = [0]*len(num_class)
    model_dis_list = [0]*len(num_class)
    optim_dis_list = [0]*len(num_class)
    sched_dis_list = [0]*len(num_class)
    trainer_dis_list = [0]*len(num_class)
    for i in range(len(num_class)):
        # Discrete Diffusion Model Setup

        FLAGS.dis_input_size[i] = train_dis_data_list[i].shape[1]
        FLAGS.dis_cond_size[i] = [each_cond.shape[1] for each_cond in train_dis_con_data_list[i]]
        print('FLAGS.dis_cond_size[i]',FLAGS.dis_cond_size[i])
        FLAGS.dis_output_size[i] = train_dis_data_list[i].shape[1]
        FLAGS.encoder_dim =  list(map(int, FLAGS.encoder_dim_dis.split(',')))
        FLAGS.nf =  FLAGS.nf_dis
        model_dis_list[i] = tabularUnet(FLAGS, i)
        optim_dis_list[i] = torch.optim.Adam(model_dis_list[i].parameters(), lr=FLAGS.lr_dis)
        sched_dis_list[i] = torch.optim.lr_scheduler.LambdaLR(optim_dis_list[i], lr_lambda=warmup_lr)
        trainer_dis_list[i] = MultinomialDiffusion(num_class[i], train_dis_data_list[i].shape, model_dis_list[i], FLAGS, timesteps=FLAGS.T,loss_type='vb_stochastic').to(device)

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
    logging.info("Discrete: %d, %d"%(train_dis_data.shape[0], train_dis_data.shape[1]))

    # Start Training
    if FLAGS.eval==False:
        train_iter_dis_list = [0]*len(num_class)
        datalooper_train_dis_list = [0]*len(num_class)
        x_0_dis_list = [0]*len(num_class)
        epoch = 0
        train_iter_cont = DataLoader(train_cont_data, batch_size=FLAGS.training_batch_size)
        train_iter_attention_list = [DataLoader(torch.tensor(attention_train), batch_size=FLAGS.training_batch_size) for attention_train in attention_train_list]
        datalooper_train_cont = infiniteloop(train_iter_cont)
        datalooper_train_attention_list = [infiniteloop(train_iter_attention) for train_iter_attention in train_iter_attention_list]
        for i in range(len(num_class)):
            # train_iter_con = DataLoader(train_con_data, batch_size=FLAGS.training_batch_size)
            train_iter_dis_list[i] = DataLoader(train_dis_data_list[i], batch_size=FLAGS.training_batch_size)
            # datalooper_train_con = infiniteloop(train_iter_con)
            datalooper_train_dis_list[i] = infiniteloop(train_iter_dis_list[i])
        writer = SummaryWriter(FLAGS.logdir)
        writer.flush()
        for step in range(total_steps_both):
            model_cont.train()
            x_0_cont = next(datalooper_train_cont).to(device)
                # ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
                # con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, transformer_dis, FLAGS)

            x_attention_list = [next(datalooper_train_attention).to(device) for datalooper_train_attention in datalooper_train_attention_list]
            for i, each in enumerate(x_attention_list):
                if i == 1 or i == 0 or i == 4:
                    x_attention_list[i] = each.permute(1, 0)

            for i in range(len(num_class)):
                if i not in FLAGS.still_condition:
                    # model_con.train()
                    model_dis_list[i].train()

                    # x_0_con = next(datalooper_train_con).to(device).float()
                x_0_dis_list[i] = next(datalooper_train_dis_list[i]).to(device)


                # ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
                # con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, transformer_dis, FLAGS)
            # !dis_loss_list = training_with(x_0_dis_list, trainer_dis_list, FLAGS)
            cont_loss, dis_loss_list = training_with(x_0_cont, x_0_dis_list, x_attention_list,
                                                     trainer_cont, trainer_dis_list,
                                                     trainer_cont, FLAGS,
                                                     still_cond_used_for_sampling_list)
            # loss_con = con_loss + FLAGS.lambda_con * con_loss_ns
            # loss_dis = dis_loss + FLAGS.lambda_dis * dis_loss_ns
            loss_cont = cont_loss
            optim_cont.zero_grad()
            loss_cont.backward()
            torch.nn.utils.clip_grad_norm_(model_cont.parameters(), FLAGS.grad_clip)
            optim_cont.step()
            sched_cont.step()
            writer.add_scalar('loss_continuous', cont_loss, step)
            for i in range(len(num_class)):
                # loss_con = con_loss + FLAGS.lambda_con * con_loss_ns
                # loss_dis = dis_loss + FLAGS.lambda_dis * dis_loss_ns
                if i not in FLAGS.still_condition:
                    loss_dis = dis_loss_list[i]
                    loss_dis.backward()
                    optim_dis_list[i].step()
                    sched_dis_list[i].step()
                    optim_dis_list[i].zero_grad()
                    torch.nn.utils.clip_grad_value_(trainer_dis_list[i].parameters(), FLAGS.grad_clip)  # , self.args.clip_value)
                    torch.nn.utils.clip_grad_norm_(trainer_dis_list[i].parameters(), FLAGS.grad_clip)  # , self.args.clip_norm)
                    writer.add_scalar('loss_discrete', dis_loss_list[i], step)

            # log
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
                for i in range(len(num_class)):
                    if i not in FLAGS.still_condition:
                        logging.info(f"Epoch :{epoch}, discrete loss: {dis_loss_list[i]:.6f}")
                epoch +=1

            if step > 0 and sample_step > 0 and step % sample_step == 0 or step==(total_steps_both-1):
                print('llll')
                log_x_T_dis_list = [0] * len(num_class)
                x_dis_list = [0] * len(num_class)
                sample_dis_list = [0] * len(num_class)
                sample_list = [0] * len(num_class)

                model_cont.eval()
                for i in range(len(num_class)):
                    if i not in FLAGS.still_condition:
                        model_dis_list[i].eval()
                for i, each in enumerate(attention_tensor_list):
                    if i == 1 or i == 0 or i == 4:
                        attention_tensor_list[i] = each.permute(1, 0)
                with torch.no_grad():
                    x_T_cont = torch.randn(train_cont_data.shape[0], train_cont_data.shape[1]).to(device)
                    for i in range(len(num_class)):
                        log_x_T_dis_list[i] = log_sample_categorical(torch.zeros(train_dis_data_list[i].shape, device=device), num_class[i]).to(device)
                    x_cont, x_dis_list = sampling_with(x_T_cont, log_x_T_dis_list, attention_tensor_list, net_sampler, trainer_dis_list, transformer_con, FLAGS, still_cond_used_for_sampling_list)
                sample_cont = transformer_con.inverse_transform(x_cont.detach().cpu().numpy())
                # sample_dis = transformer_dis.inverse_transform(still_cond_used_for_sampling)
                x_dis = torch.tensor(np.concatenate(x_dis_list, axis=1))
                x_dis = apply_activate(x_dis, transformer_dis.output_info)
                sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
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
                    for i in range(len(num_class)):
                        ckpt[f'model_dis_{i}'] = model_dis_list[i].state_dict()
                        ckpt[f'sched_dis_{i}'] = sched_dis_list[i].state_dict()
                        ckpt[f'optim_dis_{i}'] = optim_dis_list[i].state_dict()
                        # 'ml_param': param

                    torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
        logging.info(f"Evaluation best : {scores_max_eval}")

        #final test
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        model_cont.load_state_dict(ckpt['model_con'])
        model_cont.eval()
        for i in range(len(num_class)):
            if i not in FLAGS.still_condition:
                model_dis_list[i].load_state_dict(ckpt[f'model_dis_{i}'])
                # model_con.eval()
                model_dis_list[i].eval()


        # fake_sample=[]
        for ii in range(1):
            logging.info(f"sampling {ii}")
            with torch.no_grad():
                x_T_cont = torch.randn(train_cont_data.shape[0], train_cont_data.shape[1]).to(device)
                for i in range(len(num_class)):
                    log_x_T_dis_list[i] = log_sample_categorical(
                        torch.zeros(train_dis_data_list[i].shape, device=device), num_class[i]).to(device)
                x_cont, x_dis_list = sampling_with(x_T_cont, log_x_T_dis_list,attention_tensor_list, net_sampler, trainer_dis_list,
                                                   transformer_con, FLAGS, still_cond_used_for_sampling_list)
            sample_cont = transformer_con.inverse_transform(x_cont.detach().cpu().numpy())
            # sample_dis = transformer_dis.inverse_transform(still_cond_used_for_sampling)
            x_dis = torch.tensor(np.concatenate(x_dis_list, axis=1))
            x_dis = apply_activate(x_dis, transformer_dis.output_info)
            sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
            sample = np.zeros([train_cont_data.shape[0], len(con_idx + dis_idx)])
            for i in range(len(con_idx)):
                sample[:, con_idx[i]] = sample_cont[:, i]
            for i in range(len(dis_idx)):
                sample[:, dis_idx[i]] = sample_dis[:, i]
            sample_pd = pd.DataFrame(sample).dropna()
            sample_pd.to_csv(os.path.join(FLAGS.logdir, f'{FLAGS.logdir}_sample_{ii}.csv'), index=False)
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
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'), map_location=torch.device('cpu'))
        model_cont.load_state_dict(ckpt['model_con'])
        model_cont.eval()
        for i in range(len(num_class)):
            if i not in FLAGS.still_condition:
                model_dis_list[i].load_state_dict(ckpt[f'model_dis_{i}'])
                # model_con.eval()
                model_dis_list[i].eval()

        def transformer(x):
            data = [x]
            info = transformer_dis.meta[0]
            col_t = np.zeros([len(data), info['size']])
            idx = list(map(info['i2s'].index, data))
            col_t[np.arange(len(data)), idx] = 1
            data_t = col_t
            return data_t


        filename = FLAGS.data+'.csv'
        DATA_PATH = os.path.join(os.path.dirname(__file__), 'tabular_datasets')
        local_path = os.path.join(DATA_PATH, filename)
        gen = pd.read_csv(local_path)
        gen_res = []
        gen_wait = []
        gen_process = []
        gen_act = list(gen['act'])
        acts_prev = []
        res_prev = []
        for i in range(len(gen_act)):
            if gen_act[i] == 'Start':
                start_flag=i
                gen_res.append('Start')
                gen_wait.append(0)
                gen_process.append(0)
            elif gen_act[i] == 'End':
                gen_res.append('End')
                gen_wait.append(0)
                gen_process.append(0)
                # pass
            else:
                still_cond_used_for_sampling_list = [transformer(int(gen_act[i]))]
                cur_act = [[int(gen_act[i])]]
                if i == start_flag+1:
                    acts_prev = [[FLAGS.src_vocab_size_list[0]-1]]
                    res_prev = [[FLAGS.src_vocab_size_list[1]-1]]

                    acts_padding = [[True]]
                    res_padding = [[True]]
                elif i == start_flag+2:
                    acts_prev = [[]]
                    res_prev = [[]]
                    acts_prev[0].append(int(gen_act[i-1]))
                    res_prev[0].append(int(gen_res[i-1]))

                    acts_padding = [[False for each in acts_prev[0]]]
                    res_padding = [[False for each in res_prev[0]]]
                else:
                    acts_prev[0].append(int(gen_act[i - 1]))
                    res_prev[0].append(int(gen_res[i - 1]))

                    acts_padding = [[False for each in acts_prev[0]]]
                    res_padding = [[False for each in res_prev[0]]]

                attention_tensor_list = [torch.tensor(acts_prev).to(device), torch.tensor(res_prev).to(device),
                                             torch.tensor(acts_padding).to(device),torch.tensor(res_padding).to(device),
                                         torch.tensor(cur_act).to(device) ]
                for i, each in enumerate(attention_tensor_list):
                    if i == 1 or i == 0 or i == 4:
                        attention_tensor_list[i] = each.permute(1, 0)
                log_x_T_dis_list = [0] * len(num_class)
                x_dis_list = [0] * len(num_class)
                with torch.no_grad():
                    x_T_cont = torch.randn(1, train_cont_data.shape[1]).to(device)
                    for i in range(len(num_class)):
                        log_x_T_dis_list[i] = log_sample_categorical(
                            torch.zeros((1, train_dis_data_list[i].shape[1]), device=device), num_class[i]).to(
                            device)
                    x_cont, x_dis_list = sampling_with(x_T_cont, log_x_T_dis_list, attention_tensor_list,
                                                       net_sampler, trainer_dis_list,
                                                       transformer_con, FLAGS, still_cond_used_for_sampling_list)
                sample_cont = transformer_con.inverse_transform(x_cont.detach().cpu().numpy())
                # sample_dis = transformer_dis.inverse_transform(still_cond_used_for_sampling)
                x_dis = torch.tensor(np.concatenate(x_dis_list, axis=1))
                x_dis = apply_activate(x_dis, transformer_dis.output_info)
                sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
                sample = np.zeros([1, len(con_idx + dis_idx)])
                for i in range(len(con_idx)):
                    sample[:, con_idx[i]] = sample_cont[:, i]
                for i in range(len(dis_idx)):
                    sample[:, dis_idx[i]] = sample_dis[:, i]
                # sample_pd = pd.DataFrame(sample).dropna()
                print('sample', sample)
                new_res = sample[:, 1][0]

                gen_res.append(new_res)
                gen_wait.append(sample[:, 2][0])
                gen_process.append(sample[:, 3][0])
        gen['res'] = gen_res
        gen['wait'] = gen_wait
        gen['process'] = gen_process
        gen.to_csv(os.path.join(FLAGS.logdir, f'gen_sample.csv'), index=False)

        # # fake_sample=[]
        #     log_x_T_dis_list = [0]*len(num_class)
        #     x_dis_list = [0]*len(num_class)
        #     with torch.no_grad():
        #         x_T_cont = torch.randn(train_cont_data.shape[0], train_cont_data.shape[1]).to(device)
        #         for i in range(len(num_class)):
        #             log_x_T_dis_list[i] = log_sample_categorical(
        #                 torch.zeros((1,train_dis_data_list[i].shape[1]), device=device), num_class[i]).to(device)
        #         x_cont, x_dis_list = sampling_with(x_T_cont, log_x_T_dis_list, attention_tensor_list, net_sampler, trainer_dis_list,
        #                                            transformer_con, FLAGS, still_cond_used_for_sampling)
        #     sample_cont = transformer_con.inverse_transform(x_cont.detach().cpu().numpy())
        #     # sample_dis = transformer_dis.inverse_transform(still_cond_used_for_sampling)
        #     x_dis = torch.tensor(np.concatenate(x_dis_list, axis=1))
        #     x_dis = apply_activate(x_dis, transformer_dis.output_info)
        #     sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
        #     sample = np.zeros([1, len(con_idx + dis_idx)])
        #     for i in range(len(con_idx)):
        #         sample[:, con_idx[i]] = sample_cont[:, i]
        #     for i in range(len(dis_idx)):
        #         sample[:, dis_idx[i]] = sample_dis[:, i]
        #     sample_pd = pd.DataFrame(sample).dropna()
        #     new_res = sample[:,0][0]
            # sample_pd.to_csv(os.path.join(FLAGS.logdir, f'{FLAGS.logdir}_sample_{ii}.csv'), index=False)
        # scores, std = evaluation.compute_scores(train=train, test = test, synthesized_data=fake_sample, metadata=meta, eval=ckpt['ml_param'])
        # div_mean, div_std = evaluation.compute_diversity(train=train, fake=fake_sample)
        # scores['coverage'] = div_mean['coverage']
        # std['coverage'] = div_std['coverage']
        # scores['density'] = div_mean['density']
        # std['density'] = div_std['density']
        # logging.info(f"---------Test----------")
        # logging.info(scores)
        # logging.info(std)