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

from . import layers
import torch.nn as nn
import torch

from models.AttentionBlock import AttentionBlock

get_act = layers.get_act
default_initializer = layers.default_init

class tabularUnet(nn.Module):
  def __init__(self, FLAGS, i):
    super().__init__()

    self.embed_dim = FLAGS.nf # 16
    tdim = self.embed_dim*4 # 64
    self.act = get_act(FLAGS)

    modules = []
    modules.append(nn.Linear(self.embed_dim, tdim))   #[nn(16,64) ]
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)   #nn(16,64),权重值初始化
    nn.init.zeros_(modules[-1].bias)   #nn(16,64),bias初始化
    modules.append(nn.Linear(tdim, tdim))  #[nn(16,64),nn(64,64) ]
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)   #nn(64,64),权重值初始化
    nn.init.zeros_(modules[-1].bias)   #nn(64,64),bias初始化

    cond_out_list = []
    if i == '-1':
      for each_cond in range(len(FLAGS.cont_cond_size)):
        cond = FLAGS.cont_cond_size[each_cond]   # condition size
        cond_out = (FLAGS.cont_input_size)//2   # input/2
        if cond_out < 2:
          cond_out = FLAGS.cont_input_size   # input_size=3 or 2 or 2
        cond_out_list.append(cond_out)
        modules.append(nn.Linear(cond, cond_out))  #[nn(16,64),nn(64,64), nn(condition_size, cond_out(或为input的1半)) ]
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)   #weight初始化
        nn.init.zeros_(modules[-1].bias)   #bias初始化
    else:
      for each_cond in range(len(FLAGS.dis_cond_size[i])):
        cond = FLAGS.dis_cond_size[i][each_cond]  # condition size
        cond_out = (FLAGS.dis_input_size[i]) // 2  # input/2
        if cond_out < 2:
          cond_out = FLAGS.dis_input_size[i]  # input_size=3 or 2 or 2
        cond_out_list.append(cond_out)
        modules.append(nn.Linear(cond, cond_out))  # [nn(16,64),nn(64,64), nn(condition_size, cond_out(或为input的1半)) ]
        modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)  # weight初始化
        nn.init.zeros_(modules[-1].bias)  # bias初始化
    self.all_modules = nn.ModuleList(modules)


    # for each_cond in range(len(FLAGS.cond_size[i])):
    cond_size = sum(cond_out_list)
    if i == '-1':
      dim_in = FLAGS.cont_input_size + cond_size   #  input  是input data和condition layer的output的维度

    else:
      dim_in = FLAGS.dis_input_size[i] + cond_size
      # dim_in = FLAGS.dis_input_size[i] + cond_size + FLAGS.dmodel  #  input  是input data和condition layer的output的维度
    print('dim_in',i, dim_in, cond_size)
    # dim_in = FLAGS.input_size[i] #  input  是input data和condition layer的output的维度
    dim_out = list(FLAGS.encoder_dim)[0]
    self.inputs = nn.Linear(dim_in, dim_out) # input layer      nn(input, 64)

    self.encoder = layers.Encoder(list(FLAGS.encoder_dim), tdim, FLAGS) # encoder   Encoder([64,128,256],64, FLAGS)

    dim_in = list(FLAGS.encoder_dim)[-1]   # 256
    dim_out = list(FLAGS.encoder_dim)[-1]   # 256
    self.bottom_block = nn.Linear(dim_in, dim_out) #bottom_block_layer     nn(256,256)
    
    self.decoder = layers.Decoder(list(reversed(FLAGS.encoder_dim)), tdim, FLAGS) #decoder     Decoder([256,128,64],64, FLAGS)

    dim_in = list(FLAGS.encoder_dim)[0]
    if i == '-1':
      dim_out = FLAGS.cont_output_size
    else:
      dim_out = FLAGS.dis_output_size[i]
    self.outputs = nn.Linear(dim_in, dim_out) #output layer    nn(64, output)

    # self.attention = AttentionBlock(FLAGS.src_vocab_size_list, FLAGS.tgt_vocab_size, len(FLAGS.src_vocab_size_list))

  def forward(self, x, time_cond, cond, x_attention, if_cont):
    modules = self.all_modules   #[nn(16,64),nn(64,64), nn(condition_size, cond_out(或为input的1半)) ]
    m_idx = 0

    #time embedding
    temb = layers.get_timestep_embedding(time_cond, self.embed_dim)   # output的维度 self.embed_dim=16
    temb = modules[m_idx](temb)    # nn(16,64)       # input=16, output=64
    m_idx += 1
    temb= self.act(temb)      # relu
    temb = modules[m_idx](temb)    # nn(64,64)     # input=64, output=64
    m_idx += 1
    
    #condition layer
    all_cond = None
    for each_cond in range(len(cond)):
      cond_ = modules[m_idx](cond[each_cond])   # nn(condition_size, cond_out)    # input=condition_size, output=cond_out(或为input的1半)
      m_idx += 1

      if each_cond == 0:
        all_cond = cond_
      else:
        all_cond = torch.cat([all_cond, cond_], dim=1).float()

    if if_cont:
      x = torch.cat([x, all_cond], dim=1).float()  # x是continuous data或者discrete data加上condition的维度
    else:
    # attention


      # attention = self.attention(src_list=x_attention[:-3], tgt=x_attention[-1], src_key_padding_mask=x_attention[-3:-1])
      # x = torch.cat([x, all_cond, attention], dim=1).float()   #x是continuous data或者discrete data加上condition的维度
      x = torch.cat([x, all_cond], dim=1).float()
    # x = torch.cat([x], dim=1).float()  # x是continuous data或者discrete data加上condition的维度
    inputs = self.inputs(x) #input layer   nn(input, 64)    #   input  是input data和condition layer的output ,
    # output=64 asa inputs(value)=64
    skip_connections, encoding = self.encoder(inputs, temb)   #encoder input=64, output=256（layers第104行，x=64->128->256)
    encoding = self.bottom_block(encoding)   #nn(256,256)  input=256, output=256
    encoding = self.act(encoding)    # relu output=256
    x = self.decoder(skip_connections, encoding, temb) # decoder([128,256],256,t=64),  output的x=64

    outputs = self.outputs(x)    #   nn(64, output)

    return outputs
