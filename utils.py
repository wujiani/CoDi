import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random

def warmup_lr(step):
    return min(step, 5000) / 5000

def infiniteloop(dataloader):
    while True:
        for _, y in enumerate(dataloader):
            yield y

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'sigmoid':
            ed = st + item[0]
            data_t.append(data[:,st:ed])
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.softmax(data[:, st:ed]))
            st = ed
        else:
            assert 0
    return torch.cat(data_t, dim=1)

def log_sample_categorical(logits, num_class):
    full_sample = []
    # k=0
    # for i in range(len(num_classes)):
    logits_column = logits
    # k+=num_classes[i]
    uniform = torch.rand_like(logits_column)
    gumbel_noise = -torch.log(-torch.log(uniform+1e-30)+1e-30)
    sample = (gumbel_noise + logits_column).argmax(dim=1)
    col_t =np.zeros(logits_column.shape)
    col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
    full_sample.append(col_t)
    full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
    log_sample = torch.log(full_sample.float().clamp(min=1e-30))
    return log_sample


# def sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, trans, FLAGS):
def sampling_with(log_x_T_dis, trainer_dis, FLAGS, still_cond_used_for_sampling):
    x_t_dis = [0]*len(log_x_T_dis)
    for i in range(len(log_x_T_dis)):

        # if i == FLAGS.still_condition:
        #     x_t_dis[i] = torch.tensor(still_cond_used_for_sampling)
        # # x_t_con = x_T_con
        # else:
        x_t_dis[i] = log_x_T_dis[i]

    for time_step in reversed(range(FLAGS.T)):

        for i in range(len(log_x_T_dis)):
            if i != FLAGS.still_condition:

            # t = x_t_con.new_ones([x_t_con.shape[0], ], dtype=torch.long) * time_step
                t = x_t_dis[i].new_ones([x_t_dis[i].shape[0], ], dtype=torch.long) * time_step
                # mean, log_var = net_sampler.p_mean_variance(x_t=x_t_con, t=t, cond = x_t_dis.to(x_t_con.device), trans=trans)
                # if time_step > 0:
                    # noise = torch.randn_like(x_t_con)
                # elif time_step == 0:
                #     noise = 0
                # x_t_minus_1_con = mean + torch.exp(0.5 * log_var) * noise
                # x_t_minus_1_con = torch.clip(x_t_minus_1_con, -1., 1.)
                cond = []
                for j in range(len(log_x_T_dis)):

                    if j != i:
                        if j == FLAGS.still_condition:

                            cond.append(torch.tensor(still_cond_used_for_sampling).to(torch.float32).to(x_t_dis[j].device))
                        else:
                            cond.append(x_t_dis[j])
                        # print(x_t_dis[j].shape)
                x_t_minus_1_dis = trainer_dis[i].p_sample(x_t_dis[i], t, cond)
                # x_t_con = x_t_minus_1_con
                x_t_dis[i] = x_t_minus_1_dis

    x_t_dis[FLAGS.still_condition] = torch.tensor(still_cond_used_for_sampling).to(torch.float32).to(x_t_dis[FLAGS.still_condition].device)
    return  [x.detach().cpu() for x in x_t_dis]

# def training_with(x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, trans, FLAGS):
def training_with(x_0_dis, trainer_dis, FLAGS, neg_cond):

    # t = torch.randint(FLAGS.T, size=(x_0_con.shape[0], ), device=x_0_con.device)
    t = torch.randint(FLAGS.T, size=(x_0_dis[-1].shape[0], ), device=x_0_dis[-1].device)
    pt = torch.ones_like(t).float() / FLAGS.T

    #co-evolving training and predict positive samples
    # noise = torch.randn_like(x_0_con)
    # x_t_con = trainer.make_x_t(x_0_con, t, noise)
    log_x_start = [0]*len(x_0_dis)
    x_t_dis = [0]*len(x_0_dis)
    for i in range(len(x_0_dis)):
        if i != FLAGS.still_condition:
            log_x_start[i] = torch.log(x_0_dis[i].float().clamp(min=1e-30))
            x_t_dis[i] = trainer_dis[i].q_sample(log_x_start=log_x_start[i], t=t)
            # eps = trainer.model(x_t_con, t, x_t_dis.to(x_t_con.device))
            # ps_0_con = trainer.predict_xstart_from_eps(x_t_con, t, eps=eps)
            # con_loss = F.mse_loss(eps, noise, reduction='none')
            # con_loss = con_loss.mean()
    dis_loss = [0]*len(x_0_dis)
    ps_0_dis = [0]*len(x_0_dis)
    for i in range(len(x_0_dis)):
        if i != FLAGS.still_condition:
            cond = []
            for j in range(len(x_0_dis)):
                if j != i:
                    if j == FLAGS.still_condition:
                        cond.append(x_0_dis[j].to(torch.float32))
                    else:
                        cond.append(x_t_dis[j])

            kl, temp_ps_0_dis = trainer_dis[i].compute_Lt(log_x_start[i], x_t_dis[i], t, cond)

            ps_0_dis[i] = torch.exp(temp_ps_0_dis)
            kl_prior = trainer_dis[i].kl_prior(log_x_start[i])
            dis_loss[i] = (kl / pt + kl_prior).mean()

    # negative condition -> predict negative samples
    # noise_ns = torch.randn_like(ns_con)
    # ns_t_con = trainer.make_x_t(ns_con, t, noise_ns)
    # log_ns_start = torch.log(ns_dis.float().clamp(min=1e-30))
    # ns_t_dis = trainer_dis.q_sample(log_x_start=log_ns_start, t=t)
    # eps_ns = trainer.model(x_t_con, t, ns_t_dis.to(ns_t_dis.device))
    # ns_0_con = trainer.predict_xstart_from_eps(x_t_con, t, eps=eps_ns)
    triplet_dis = 0
    for i in range(len(x_0_dis)):
        if i != FLAGS.still_condition:
            _, ns_0_dis = trainer_dis[i].compute_Lt(log_x_start[i], x_t_dis[i], t, [neg_cond])

            ns_0_dis = torch.exp(ns_0_dis)

    # contrastive learning loss
    #         triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
            # triplet_con = triplet_loss(x_0_con, ps_0_con, ns_0_con)
            # st=0
            triplet_dis = []

            ps_dis = F.cross_entropy(ps_0_dis[i], torch.argmax(x_0_dis[i], dim=-1).long(), reduction='none')
            ns_dis = F.cross_entropy(ns_0_dis, torch.argmax(x_0_dis[i], dim=-1).long(), reduction='none')

            triplet_dis.append(max((ps_dis-ns_dis).mean()+1,0))

            triplet_dis = sum(triplet_dis)/len(triplet_dis)
    # return con_loss, triplet_con, dis_loss, triplet_dis
    return dis_loss, triplet_dis
def make_negative_condition(x_0_con, x_0_dis):

    device = x_0_con.device
    x_0_con = x_0_con.detach().cpu().numpy()
    x_0_dis = x_0_dis.detach().cpu().numpy()

    nsc_raw = pd.DataFrame(x_0_con)
    nsd_raw = pd.DataFrame(x_0_dis)
    nsc = np.array(nsc_raw.sample(frac=1, replace = False).reset_index(drop=True))
    nsd = np.array(nsd_raw.sample(frac=1, replace = False).reset_index(drop=True))
    ns_con = nsc
    ns_dis = nsd

    return torch.tensor(ns_con).to(device), torch.tensor(ns_dis).to(device)


def neg_condition(x_0_dis_list, res_no_act_list, transformer_dis, FLAGS, i, still_condition):

    idx = torch.argmax(x_0_dis_list[i], dim=1)
    a = list(map(transformer_dis.meta[i]['i2s'].__getitem__, idx))
    oo = []
    for each in a:
        oo.append(random.choice(res_no_act_list[str(each)]))

    col_t = np.zeros([len(oo), transformer_dis.meta[still_condition]['size']])
    idx = list(map(transformer_dis.meta[still_condition]['i2s'].index, oo))
    # print("info['i2s']", info['i2s'])
    col_t[np.arange(len(oo)), idx] = 1
    neg_cond = torch.tensor(col_t).to(torch.float32)
    return neg_cond