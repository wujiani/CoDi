# Techniques for Discovering Event-Log Generative Models: Diffusion model

## Requirements
Run the following to install requirements:
```setup
pip install -r req.txt
```

## Training
* You can train our diffusion model from scratch by run (example):
```bash
python main.py --data train_Production.xes --total_epochs_both 1000 --training_batch_size 50 --logdir exp_final_p2p --T 100 --lr_dis 1e-4
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
python main.py ???```
