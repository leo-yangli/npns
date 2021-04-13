# NPNs

This repository contains the code for [Neural Plasticity Networks ](https://arxiv.org/abs/1908.08118).

## Demo
1. The evolutions of decision boundaries of the learned networks on synthetic "Moons" dataset: (left) Network Sparsification, and (right) Network Expansion.
<p align="center">
    <img width="350" alt="moons_spa" src="https://github.com/leo-yangli/npns/blob/master/demo/moons_spa.gif?raw=true"/>
    <img width="350" alt="moons_exp" src="https://github.com/leo-yangli/npns/blob/master/demo/moons_exp.gif?raw=true"/>
</p>
2. Visualization of part of the neurons in conv-layer and fully-connected layer of the LeNet-5-Caffe sparsified / expanded by NPNs. To achieve computational efficiency, only neuron-level (instead of weight-level) sparsification is considered.
<p align="center">
    <img width="350" alt="exp_conv" src="https://github.com/leo-yangli/npns/blob/master/demo/exp_conv.gif?raw=true"/>
    <img width="350" alt="spar_conv" src="https://github.com/leo-yangli/npns/blob/master/demo/spar_conv.gif?raw=true"/><br/>
    <img width="350" alt="exp_fc" src="https://github.com/leo-yangli/npns/blob/master/demo/exp_fc.gif?raw=true"/>
    <img width="350" alt="spar_fc" src="https://github.com/leo-yangli/npns/blob/master/demo/spar_fc.gif?raw=true"/>
</p>

## Requirements
    pytorch==1.3
    tensorboard
    matplotlib

## Usage
    python train_syn.py --k 7 --mode sparse --init_size 100 80 --stage1 500 --stage2 1000 --lambas 0.35 14
    python train_syn.py --k 1 --mode expand --init_size 3 3 --stage1 100 --stage2 1000 --lambas 0.35 14
    python train_lenet.py --k 7 --mode sparse --init_size 20 50 500
    python train_lenet.py --k 1 --mode expand --init_size 3 3 3    
    python train_resnet56.py --num_class 10 --mode expand --init_factor 0.5 0.5 0.3 0.2 0.8 --lamba 0
    python train_resnet56.py --num_class 100 --mode expand --init_factor 0.5 0.5 0.3 0.2 0.8 --lamba 0
    python train_resnet56.py --num_class 10 --mode sparse --init_factor -1 --lamba 1e-5
    python train_resnet56.py --num_class 100 --mode sparse --init_factor -1 --lamba 1e-5
        
## Citation
If you found this code useful, please cite our paper.

    @article{npn2021,
      title={Neural Plasticity Networks},
      author={Li, Yang and Ji, Shihao},
      journal={International Joint Conference on Neural Networks (IJCNN)},
      year={2021}
    }
