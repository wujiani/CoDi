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
"""Common layers for defining score networks.
"""
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def get_act(FLAGS):
  if FLAGS.activation.lower() == 'elu':
    return nn.ELU()
  elif FLAGS.activation.lower() == 'relu':
    return nn.ReLU()
  elif FLAGS.activation.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif FLAGS.activation.lower() == 'swish':
    return nn.SiLU()
  elif FLAGS.activation.lower() == 'tanh':
    return nn.Tanh()
  elif FLAGS.activation.lower() == 'softplus':
    return nn.Softplus()
  else:
    raise NotImplementedError('activation function does not exist!')

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  
  half_dim = embedding_dim // 2
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1: 
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

class Encoder(nn.Module):
  def __init__(self, encoder_dim, tdim, all_cond, FLAGS):
    super(Encoder, self).__init__()
    self.encoding_blocks = nn.ModuleList()
    for i in range(len(encoder_dim)):   # encoder_dim=[64,128,256]
      if (i+1)==len(encoder_dim): break
      encoding_block = EncodingBlock(encoder_dim[i], encoder_dim[i+1], tdim, all_cond, FLAGS)
      self.encoding_blocks.append(encoding_block)   #encoding_blocks=[(in=64,out=128),(in=128,out=256)]
          #这些并不是nn的input和output，而是定义nn的一些初始值，input和output在forward里体现

  def forward(self, x, t, cond):
    skip_connections = []
    for encoding_block in self.encoding_blocks:
      x, skip_connection = encoding_block(x, t, cond) #有点难理解，看122-125，进去是多少，出来就是两个它的两倍  # input=128, 128 or 256, 256
      skip_connections.append(skip_connection)   # [128, 256]
    return skip_connections, x  #[128,256] , 256

class EncodingBlock(nn.Module):
  def __init__(self, dim_in, dim_out, tdim, all_cond, FLAGS):   #每次out都是in的两倍
    super(EncodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(dim_in, dim_out),  #nn(in, out)
        get_act(FLAGS)
    ) 
    self.temb_proj = nn.Sequential(
        nn.Linear(tdim, dim_out),  #nn(tdim, out)
        get_act(FLAGS)
    )
    self.cond_proj = nn.Sequential(
        nn.Linear(all_cond, dim_out),  #nn(tdim, out)
        get_act(FLAGS)
    )

    self.layer2 = nn.Sequential(
        nn.Linear(dim_out, dim_out),   #nn(out, out)
        get_act(FLAGS)
    )
    
  def forward(self, x, t, cond):
    x = self.layer1(x).clone()   #nn(in, out), output=out
    x += self.temb_proj(t)   #nn(tdim, out), result=x+out 维度是out，但是值是两个out维度的值相加
    if isinstance(cond, int):
      pass
    else:
      x += self.cond_proj(cond)
    x = self.layer2(x)   #nn(out, out) output=out
    skip_connection = x   # 用于之后decoder里的residual network
    return x, skip_connection   #两个都是out维度

class Decoder(nn.Module):
  def __init__(self, decoder_dim, tdim, cond, FLAGS):
    super(Decoder, self).__init__()
    self.decoding_blocks = nn.ModuleList()
    for i in range(len(decoder_dim)):
      if (i+1)==len(decoder_dim): break
      decoding_block = DecodingBlock(decoder_dim[i], decoder_dim[i+1], tdim, cond, FLAGS)   #[(256,128),(128,64)]
      self.decoding_blocks.append(decoding_block)

  def forward(self, skip_connections, x, t, cond):
    zipped = zip(reversed(skip_connections), self.decoding_blocks)
    for skip_connection, decoding_block in zipped:
      x = decoding_block(skip_connection, x, t, cond)  #(256,256,t=64)->output的x=128   /循环，重复利用x    #(128,128,t=64)->output的x=64
    return x

class DecodingBlock(nn.Module):
  def __init__(self, dim_in, dim_out, tdim, cond,FLAGS):
    super(DecodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(dim_in*2, dim_in), # nn(in*2, in)  #nn(256*2, 256) , nn(128*2, 128)
        get_act(FLAGS)  # relu
    )
    self.temb_proj = nn.Sequential(   #这个layer只是让时间变得和input一样的维度，对于time的embedding还是在tabular_unet里
        nn.Linear(tdim, dim_in),   #nn(t, in)  # nn(64, 256),    nn(64, 128)
        get_act(FLAGS)
    )
    self.cond_proj = nn.Sequential(   #这个layer只是让时间变得和input一样的维度，对于time的embedding还是在tabular_unet里
        nn.Linear(cond, dim_in),   #nn(t, in)  # nn(64, 256),    nn(64, 128)
        get_act(FLAGS)
    )
    self.layer2 = nn.Sequential(
        nn.Linear(dim_in, dim_out),  # nn(in, out)   # nn(256, 128),    nn(128, 64)
        get_act(FLAGS)
    )
    
  def forward(self, skip_connection, x, t, cond):
    
    x = torch.cat((skip_connection, x), dim=1)   # residual network,              skip_connection=256, x=256 -> x=256*2  or x=128*2
    x = self.layer1(x).clone()  # nn(in*2, in)   # output=in                                   #nn(256*2, 256) , nn(128*2, 128)
    x += self.temb_proj(t)   # nn(t, in) input=t, output=in, 前一个x+这次的output=in             # nn(64, 256),    nn(64, 128)
    if isinstance(cond, int):
      pass
    else:
      x += self.cond_proj(cond)
    x = self.layer2(x)   # nn(in, out), output=out                                            # nn(256, 128),    nn(128, 64)

    return x