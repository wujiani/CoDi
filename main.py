import os
import warnings
from absl import app, flags
# import tensorflow as tf
import torch
import logging
import numpy as np
import pandas as pd
import co_evolving_condition
from utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore", category=DeprecationWarning)



FLAGS = flags.FLAGS
flags.DEFINE_string('data', 'train_Production.xes', help='dataset')
flags.DEFINE_string('id_column', 'caseid', help='dataset')
flags.DEFINE_string('act_column', 'concept:name', help='dataset')
flags.DEFINE_string('time_column', 'time:timestamp', help='dataset')
flags.DEFINE_string('resource_column', 'user', help='dataset')
flags.DEFINE_string('state_column', 'lifecycle:transition', help='dataset')
flags.DEFINE_string('logdir', './codi_exp', help='log directory')
flags.DEFINE_bool('train', True, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate')
flags.DEFINE_string('still_condition', "0", help='encoder_dim_con')
flags.DEFINE_integer('seed', 2022, help='random sample')
flags.DEFINE_string('gen_seq_output', '', help='dataset')

# Network Architecture
flags.DEFINE_multi_integer('encoder_dim', None, help='encoder_dim')
flags.DEFINE_string('encoder_dim_con', "64,128,256", help='encoder_dim_con')
flags.DEFINE_string('encoder_dim_dis', "64,128,256", help='encoder_dim_dis')
flags.DEFINE_integer('nf', None, help='nf')
flags.DEFINE_integer('nf_con', 16, help='nf_con')
flags.DEFINE_integer('nf_dis', 64, help='nf_dis')
flags.DEFINE_integer('input_size', None, help='input_size')
flags.DEFINE_integer('cond_size', None, help='cond_size')
flags.DEFINE_integer('output_size', None, help='output_size')
flags.DEFINE_string('activation', 'relu', help='activation')

flags.DEFINE_integer('cont_input_size', None, help='input_size')
flags.DEFINE_integer('cont_cond_size', None, help='cond_size')
flags.DEFINE_integer('cont_output_size', None, help='output_size')
flags.DEFINE_integer('dis_input_size', None, help='input_size')
flags.DEFINE_integer('dis_cond_size', None, help='cond_size')
flags.DEFINE_integer('dis_output_size', None, help='output_size')

flags.DEFINE_integer('src_vocab_size_list', None, help='attention')
flags.DEFINE_integer('tgt_vocab_size', None, help='attention')
flags.DEFINE_integer('dmodel', 20, help='attention')

# Training
flags.DEFINE_integer('training_batch_size', 2100, help='batch size')
flags.DEFINE_integer('eval_batch_size', 2100, help='batch size')
flags.DEFINE_integer('T', 50, help='total diffusion steps')
flags.DEFINE_float('beta_1', 0.00001, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_float('lr_con', 2e-03, help='target learning rate')
flags.DEFINE_float('lr_dis', 2e-03, help='target learning rate')
flags.DEFINE_integer('total_epochs_both', 2000, help='total training steps')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_bool('parallel', False, help='multi gpu training')

# Sampling
flags.DEFINE_integer('sample_step', 2000, help='frequency of sampling')

# Continuous diffusion model
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedsmall', ['fixedlarge', 'fixedsmall'], help='variance type')

# Contrastive Learning
flags.DEFINE_integer('ns_method', 0, help='negative condition method')
flags.DEFINE_float('lambda_con', 0.2, help='lambda_con')
flags.DEFINE_float('lambda_dis', 0.2, help='lambda_dis')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(argv):
    randomSeed = FLAGS.seed
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(randomSeed)

    if FLAGS.eval == True:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(FLAGS.logdir,exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.logdir, 'eval.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    else:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(FLAGS.logdir,exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.logdir, 'train.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    
    logging.info("Co-evolving Conditional Diffusion models")
    co_evolving_condition.train(FLAGS)

if __name__ == '__main__':
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    app.run(main)
