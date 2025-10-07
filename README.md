## Enhanced Label Distribution Aware Margins with Strategic Reweighting and Representation Optimization
#Author names
_________________

This is the official implementation of LDAM-DRW in the paper [Enhanced Label Distribution Aware Margins with Strategic Reweighting and Representation-Optimization].

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.2
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [scikit-learn](https://scikit-learn.org/stable/)

### Dataset

- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.
- The paper also reports results on Tiny TissueMnist, OrganCmnist, OrganSmnist, Mnist and Fmnist. We have the code for those datasets present.

### Training 

We provide several training examples with this repo:

- To train the ERM baseline on long-tailed imbalance with ratio of 100 for Cifar10

```bash
python method_normal.py --gpu 0 --imb_type exp --imb_factor 0.01 --dataset cifar10 --cnum 10 --loss_type CE --train_rule None
```


- To train the LDAM Loss along with DMA(our method) with Supervised Contrastive Loss training on long-tailed imbalance with ratio of 100 for Cifar10

```bash
python method_supcon.py --gpu 0 --imb_type exp --imb_factor 0.01 --dataset cifar10 --cnum 10 --loss_type LDAM --train_rule DMA
```
- To train the LDAM Loss along with DMA(our method) with Supervised Contrastive Loss training for OrgranCmnist

```bash
python method_supcon.py --gpu 0 --dataset organcmnist --cnum 11 --loss_type LDAM --train_rule DMA
```
- To train the LDAM Loss along with DMA(our method) with Supervised Contrastive Loss training for Tissuemnist

```bash
python method_supcon.py --gpu 0 --dataset tissuemnist --cnum 8 --loss_type LDAM --train_rule DMA
```
- To train the LDAM Loss along with DMA(our method) with Supervised Contrastive Loss training for OrgranSmnist

```bash
python method_supcon.py --gpu 0 --dataset organsmnist --cnum 11 --loss_type LDAM --train_rule DMA
```
### Reference

