# CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis
This code is the official implementation of "CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis".
(https://arxiv.org/abs/2304.12654)

## Requirements
Run the following to install requirements:
```setup
conda env create --file environment.yaml
```

## Usage
* Train and evaluate CoDi through `main.py`:
```sh
main.py:
  --data: tabular dataset
  --eval : train or eval
  --logdir: Working directory
```

## Training
* You can train our CoDi from scratch by run:
```bash
python main.py --data heart --logdir CoDi_exp
```

## Evaluation
* By run the following script, you can reproduce our experimental result: 
    binary classification result of CoDi on Heart in Table 10. 
```bash
python main.py --data heart --eval True --logdir CoDi_exp
```

先把我对题主问题的回答写在最前面：因为我们预测的虽然是噪声，但其实预测的是x(t-1)的分布，为什么预测分布而不直接预测x(t-1)这张图？因为为了保证随机性，我们不希望每个x(T)都只唯一对应一个x(0)
由于去除噪声的过程中我们使用的是同一个网络，而每一步的噪声的总量并不相同，因此可以给网络一个额外的输入T，告诉网络现在是在哪一步了。大佬的blog中指出网络可以根据噪声等级的不同，自动选择关注全局特征还是局部特征。因为图像中每一个像素添加的高斯噪声是独立的，当噪声较多时（对应上图左侧）网络只能将更大范围里所有像素的特征合并起来考虑，这样各个像素特征中的高斯噪声能够在融合的过程中相互抵消，此时网络只能恢复出图像的大致轮廓。当噪声比较少时（对应上图右侧），网络可以更关注更小范围内的细节，从而恢复出图像细节。