# Techniques for Discovering Event-Log Generative Models: Diffusion model

## Requirements
Run the following to install requirements:
```setup
pip install -r req.txt
```


## Training
* You can train our diffusion model from scratch by run (example):
```bash
python main.py --data train_PurchasingExample.xes --total_epochs_both 1000 --training_batch_size 50 --logdir exp_final_p2p --T 100 --lr_dis 1e-4
```
`--data`: xes file for training

`--total_epochs_both` (optional): num epoch for both discret and continuous model

`--logdir`: where to save model checkpoint file and outputs

`--T`: diffusion time steps

`--lr_dis`: learning rate for discret model

Other arguments are available, I found that the default values are generally good, as future work it is useful to explore other hyperparameters.



## Sampling
* you can generate event logs by running (example):
```bash
python main.py --data train_PurchasingExample.csv --gen_seq_output gen_seq_train_PurchasingExample_0.csv --eval True --logdir exp_final_p2p --T 100 --seed 10

```
`--gen_seq_output`: output csv file of gen_seq from probabilistic method 

`--eval`: True indicates to sample new data from the model, default is False

`--logdir`: where to save model checkpoint file and outputs, can read model from this directory

`--T`: diffusion time steps, same as in the training step

`--seed`: random seed, for generating different resources and times 


## Evaluation
The output of diffusion model is not complete event logs since time duration need to be transformed into timestamps based on the start time of each case. The procedures of generating a complete event log and evaluate the performance of models is described in https://github.com/wujiani/EventLogsGenerator.git.