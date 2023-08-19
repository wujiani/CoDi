# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
# import torch
# import numpy as np
from tabular_transformer import GeneralTransformer
import json
import logging
import os
import numpy as np
import pm4py
import copy
import pandas as pd

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'tabular_datasets')


def _get_columns(metadata):
    categorical_columns = list()

    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)

    return categorical_columns


def _preprocessing(df, id_column, act_column, time_column, resource_column, state_column,
                   meta_filename,
                   preprocessed_data_filename
                   ):
    pd.options.mode.chained_assignment = None

    df['waiting_time'] = None
    df['process_time'] = None
    df['last_complete_event'] = None
    df['preceding_evts'] = None
    df['paired_event'] = None
    df['next'] = None
    df['index'] = df.index
    for key, group in df.groupby(id_column):
        flag = 0
        i = list(group.index)[0]
        preceding_evt = []
        not_complete_evt_idx = []
        j = 0
        df['waiting_time'].loc[i] = 0
        not_complete_evt_idx.append(i)
        i += 1
        last_complete_evt_idx = i
        while j < len(group) - 1:
            j += 1
            cur_act = df.loc[i]
            if cur_act[state_column] == 'complete':
                flag = 0
                preceding_evt.append(cur_act.name)
                last_complete_evt_idx = i

                for each_idx in not_complete_evt_idx:
                    to_pair = df.loc[each_idx]
                    if (cur_act[act_column] == to_pair[act_column]) \
                            and (cur_act[resource_column] == to_pair[resource_column]):
                        df['paired_event'].loc[i] = each_idx
                        not_complete_evt_idx.remove(each_idx)
                        df['process_time'].loc[i] = (
                                    df[time_column].loc[i] - df[time_column].loc[each_idx]).total_seconds()
                        df['last_complete_event'].loc[i] = df['last_complete_event'].loc[each_idx]
                        break

            else:
                if flag == 1:
                    df['preceding_evts'].loc[i] = df['preceding_evts'].loc[i - 1]
                else:
                    df['preceding_evts'].loc[i] = preceding_evt  # tuple(sorted, reverse=False))
                for each in df['preceding_evts'].loc[i]:
                    if df['next'].loc[each] == None:
                        df['next'].loc[each] = [i]
                    else:
                        temp = df['next'].loc[each]
                        temp.append(i)
                        df['next'].loc[each] = temp
                flag = 1
                preceding_evt = []
                not_complete_evt_idx.append(i)

                df['last_complete_event'].loc[i] = last_complete_evt_idx
                df['waiting_time'].loc[i] = (
                            df[time_column].loc[i] - df[time_column].loc[last_complete_evt_idx]).total_seconds()
            i += 1

    ############################
    def fillna_parallel_data(bb):
        i = bb.index[0]
        j = 0
        while j < len(bb):
            if bb[state_column].loc[i] == 'complete':
                paired = int(bb['paired_event'].loc[i])
                bb['process_time'].loc[paired] = bb['process_time'].loc[i]
                bb['paired_event'].loc[paired] = i
                bb['next'].loc[paired] = bb['next'].loc[i]
                bb['waiting_time'].loc[i] = bb['waiting_time'].loc[paired]
                bb['preceding_evts'].loc[i] = bb['preceding_evts'].loc[paired]
                bb['next'].loc[paired] = bb['next'].loc[i]
            i += 1
            j += 1

    fillna_parallel_data(df)

    ###########################
    df = df[df[state_column] == 'start']

    # df = df.rename(columns = {'concept:name':'activity'})
    file = df[[id_column, act_column, resource_column, 'waiting_time', 'process_time']]
    file = file[file[act_column] != 'Start']
    file = file[file[act_column] != 'End']
    file = file.reset_index(drop=True)

    file[act_column] = file[act_column].map(lambda x: '_'.join(x.split()))
    file[resource_column] = file[resource_column].map(lambda x: '_'.join(x.split()))

    ############################
    import math
    file['waiting_time'] = file['waiting_time'].map(lambda x: math.log(x + 1))
    file['process_time'] = file['process_time'].map(lambda x: math.log(x + 1))

    ############################
    from collections import Counter
    from sklearn import preprocessing

    def get_info(df, new_df, column_name, column_type):
        if column_name == id_column:
            new_df[column_name] = df[column_name]
        else:
            if column_type == 'categorical':
                df[column_name].fillna(f'NO_{column_name}', inplace=True)
                dict_ = {}
                a = Counter(df[column_name])
                dict_['name'] = column_name
                dict_['size'] = len(a.keys())
                dict_['type'] = column_type
                enc = preprocessing.LabelEncoder()
                enc = enc.fit(list(a.keys()))  # 训练LabelEncoder,将电脑，手表，手机编码为0,1,2
                data = enc.transform(df[column_name])  # 使用训练好的LabelEncoder对原数据进行编码，也叫归一化
                dict_['i2s'] = list(enc.classes_)
                new_df[column_name] = data
                return dict_
            else:
                dict_ = {}
                dict_['max'] = max(df[column_name])
                dict_['min'] = min(df[column_name])
                dict_['name'] = column_name
                dict_['type'] = column_type
                new_df[column_name] = df[column_name]
                return dict_

    new_data = pd.DataFrame()
    a = [act_column, resource_column]
    b = ['waiting_time', 'process_time']
    c = [act_column, resource_column]

    json_input = {}
    get_info(file, new_data, id_column, 'categorical')
    json_input['attention'] = [get_info(file, new_data, each, 'categorical') for each in c]
    json_input['columns'] = [get_info(file, new_data, each, 'categorical') for each in a]
    json_input['columns'].extend([get_info(file, new_data, each, 'continuous') for each in b])
    json_input['problem_type'] = 'no'

    new_data['prev_acts'] = None
    new_data['prev_res'] = None
    new_data['curr_act'] = None
    new_data['curr_res'] = None
    data_add_prev = pd.DataFrame()

    for key, group in new_data.groupby(id_column):
        for i in range(len(group)):
            if i >= 30:

                prev_acts = list(group[act_column].iloc[i - 30:i])
                prev_res = list(group[resource_column].iloc[i - 30:i])
            else:
                prev_acts = list(group[act_column].iloc[0:i])
                prev_res = list(group[resource_column].iloc[0:i])
            #         prev_acts_str = ' '.join(prev_acts)

            group['prev_acts'].iloc[i] = prev_acts

            #         prev_res_str = ' '.join(prev_res)
            group['prev_res'].iloc[i] = prev_res
            # print(group)
            group['curr_act'].iloc[i] = group[act_column].iloc[i]
            group['curr_res'].iloc[i] = group[resource_column].iloc[i]
        data_add_prev = pd.concat([data_add_prev, group])

    data_add_prev['curr_act'] = data_add_prev['curr_act'].map(lambda x: [x])
    data_add_prev['curr_res'] = data_add_prev['curr_res'].map(lambda x: [x])

    def padding(list_, pad_number, length):
        len_list = len(list_)
        if len_list < length:
            list_ = list_ + (length - len_list) * [pad_number]
        return list_

    def padding_key(list_, act_pad):
        if list_ == 0:
            return 0
        else:
            return [True if each == act_pad else False for each in list_]

    len_act = max(data_add_prev['prev_acts'].map(lambda x: len(x)))
    len_res = max(data_add_prev['prev_res'].map(lambda x: len(x)))
    act_pad = json_input['attention'][0]['size']
    res_pad = json_input['attention'][1]['size']

    data_add_prev['prev_acts_trans'] = data_add_prev['prev_acts'].map(
        lambda x: padding(x, act_pad, len_act) if len(x) > 0 else 0)
    data_add_prev['prev_acts_key_padding_trans'] = data_add_prev['prev_acts_trans'].map(
        lambda x: padding_key(x, act_pad))
    data_add_prev['prev_res_trans'] = data_add_prev['prev_res'].map(
        lambda x: padding(x, res_pad, len_res) if len(x) > 0 else 0)
    data_add_prev['prev_res_key_padding_trans'] = data_add_prev['prev_res_trans'].map(
        lambda x: padding_key(x, res_pad))

    data_add_prev['prev_acts'] = data_add_prev['prev_acts'].map(lambda x: padding(x, act_pad, len_act))
    data_add_prev['prev_acts_key_padding'] = data_add_prev['prev_acts'].map(lambda x: padding_key(x, act_pad))
    data_add_prev['prev_res'] = data_add_prev['prev_res'].map(lambda x: padding(x, res_pad, len_res))
    data_add_prev['prev_res_key_padding'] = data_add_prev['prev_res'].map(lambda x: padding_key(x, res_pad))

    json_input['attention'][0]['len'] = len_act
    json_input['attention'][1]['len'] = len_res

    data = data_add_prev[[act_column, resource_column, 'waiting_time', 'process_time', 'prev_acts', 'prev_res',
                          'prev_acts_key_padding', 'prev_res_key_padding', 'curr_act', 'curr_res']]
    data = data.reset_index(drop=True)

    import json
    with open(meta_filename, "w") as f:
        json.dump(json_input, f)

    # with open('test.json') as json_file:
    #     a = json.load(json_file)

    train = np.array(data)
    train_1 = train[:, 0:-6].astype('float32')
    train_attention = train[:, -6:-1]
    # train_transformer = np.array(data_transformer)[:,-6:]

    test = np.array(data)
    test_1 = test[:, 0:-6].astype('float32')
    test_attention = test[:, -6:-1]

    array_dict = {'train': train_1, 'test': test_1, 'train_attention': train_attention,
                  'test_attention': test_attention}
    np.savez(preprocessed_data_filename, **array_dict)

    return array_dict, json_input


def load_data(FLAGS, benchmark=False):
    # load event log xes
    local_path = os.path.join(DATA_PATH, FLAGS.data)
    data = pm4py.read.read_xes(local_path)
    meta_filename = os.path.join(DATA_PATH, FLAGS.data.split(".")[0] + "_meta_preprocessed.json")
    preprocessed_data_filename = os.path.join(DATA_PATH, FLAGS.data.split(".")[0] + "_data_preprocessed.npz")
    df = copy.deepcopy(pd.DataFrame(data, columns=[FLAGS.id_column,
                                                   FLAGS.act_column,
                                                   FLAGS.time_column,
                                                   FLAGS.resource_column,
                                                   FLAGS.state_column],
                                    ))
    # preprocessing
    preprocessed_data_array, meta = _preprocessing(df,
                                                   FLAGS.id_column,
                                                   FLAGS.act_column,
                                                   FLAGS.time_column,
                                                   FLAGS.resource_column,
                                                   FLAGS.state_column,
                                                   meta_filename,
                                                   preprocessed_data_filename
                                                  )

    categorical_columns = _get_columns(meta)
    print('categorical_columns',categorical_columns)
    train = preprocessed_data_array['train']
    print('train', train.shape)
    test = preprocessed_data_array['test']
    attention_train_list = [preprocessed_data_array['train_attention'][:,i].tolist()
                            for i in range(preprocessed_data_array['train_attention'].shape[1])]
    attention_test_list = [preprocessed_data_array['train_attention'][:,i].tolist()
                           for i in range(preprocessed_data_array['train_attention'].shape[1])]


    return train, test, (categorical_columns, meta), attention_train_list, attention_test_list

def get_dataset(FLAGS, evaluation=False):

  batch_size = FLAGS.training_batch_size if not evaluation else FLAGS.eval_batch_size

  # if batch_size % torch.cuda.device_count() != 0:
  #   raise ValueError(f'Batch sizes ({batch_size} must be divided by'
  #                    f'the number of devices ({torch.cuda.device_count()})')


  # Create dataset builders for tabular data.
  train, test, cols, attention_train_list, attention_test_list = load_data(FLAGS)
  cols_idx = list(np.arange(train.shape[1]))
  dis_idx = cols[0]
  con_idx = [x for x in cols_idx if x not in dis_idx]
  
  #split continuous and categorical
  train_con = train[:,con_idx]
  train_dis = train[:,dis_idx]
  
  #new index
  cat_idx_ = list(np.arange(train_dis.shape[1]))[:len(cols[0])]

  transformer_con = GeneralTransformer()
  transformer_dis = GeneralTransformer()

  transformer_con.fit(train_con, [])
  transformer_dis.fit(train_dis, cat_idx_)

  train_cont_data = transformer_con.transform(train_con)
  train_dis_data = transformer_dis.transform(train_dis)
  FLAGS.src_vocab_size_list = [each['size']+1 for each in cols[1]['attention']]

  FLAGS.tgt_vocab_size = FLAGS.src_vocab_size_list[0]
  print('FLAGS.src_vocab_size_list', FLAGS.tgt_vocab_size)

  return train, train_cont_data, train_dis_data, test, attention_train_list, attention_test_list, (transformer_con, transformer_dis, cols[1]), con_idx, dis_idx
      