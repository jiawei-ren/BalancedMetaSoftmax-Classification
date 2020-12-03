# BalancedMetaSoftmax - Classification

Code for the paper "Balanced Meta-Softmax for Long-Tailed Visual Recognition" on long-tailed visual recognition datasets.

**[Balanced Meta-Softmax for Long-Tailed Visual Recognition](https://arxiv.org/abs/2007.10740)**  
Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li  
NeurIPS 2020

## Snapshot
```python

def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

```
## Requirements 
* Python 3
* [PyTorch](https://pytorch.org/) (version == 1.4)
* [yaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
* [higher](https://github.com/facebookresearch/higher)(version == 0.2.1)



## Training
### End-to-end Training
- Base model (Representation Learning)
```bash
python main.py --cfg ./config/CIFAR10_LT/softmax_imba200.yaml
```
Alternatively, you may download a pretrained model [here](https://drive.google.com/file/d/1laY5ce0-sw3HrHBQZ2Gseo9acbG2CavK/view?usp=sharing) and put it in the corresponding log folder. 
- Balanced Softmax
```bash
python main.py --cfg ./config/CIFAR10_LT/balanced_softmax_imba200.yaml
```
### Decoupled Training
After obataining the base model, train the classifier with the following command:
- Balanced Softmax
```bash
python main.py --cfg ./config/CIFAR10_LT/decouple_balanced_softmax_imba200.yaml
```
- BALMS
```bash
python main.py --cfg ./config/CIFAR10_LT/balms_imba200.yaml
```
## Evaluation

Model evaluation can be done using the following command:
```bash
python main.py --cfg ./config/CIFAR10_LT/balms_imba200.yaml --test
```

## Experiment Results
The results could be slightly different from the results reported in the paper, since we originally used an internal repository for the experiments in the paper.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom", align="left">Dataset</th>
<th valign="bottom", align="left">Backbone</th>
<th valign="bottom", align="left">Method</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">download</th>

<!-- TABLE BODY -->
<tr>
<td align="left">CIFAR-10 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">Softmax</td>
<td align="center">74.0</td>
<td align="center"><a href="https://drive.google.com/file/d/1laY5ce0-sw3HrHBQZ2Gseo9acbG2CavK/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1KEyA1kaMXXJxzKaZxQK_JyFz6smvP9ib/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">CIFAR-10 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">Balanced Softmax (end-to-end)</td>
<td align="center">79.8</td>
<td align="center"><a href="https://drive.google.com/file/d/17AsyPy5mXavxXJvLiGzWIjgIyoukFSRQ/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1QwJq9BpkSaCVRLJcthONM7oGOD7Hqu3m/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">CIFAR-10 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">Balanced Softmax (decouple)</td>
<td align="center">81.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1VztEqUdA_RCzg0oXk3rebv5EPAF5B6es/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/13q6vDu8zMSSX9NFUGqWZe_YxH-__GCTl/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">CIFAR-10 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">BALMS</td>
<td align="center">82.2</td>
<td align="center"><a href="https://drive.google.com/file/d/1CK0mDg8tpPAxnh5ZEx4-6eX3sEdQoXGi/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1WhQbiUvmjxJOIS4HiBUQm2y-N5u7xXCY/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">CIFAR-100 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">Softmax</td>
<td align="center">41.2</td>
<td align="center"><a href="https://www.dropbox.com/s/63q8cf7i62aveo6/model_final.pth?dl=0">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1xsOPbCNZsHbRUkmobFaUCHBPulRRP8Rh/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">CIFAR-100 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">Balanced Softmax (end-to-end)</td>
<td align="center">46.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1Dyutyp7InoaQdePJXZDSrvhSts7y2Z6-/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1tMa88v-ZWPuIMzF0N0XYg9r6xCrpEn0i/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">CIFAR-100 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">Balanced Softmax (decouple)</td>
<td align="center">47.2</td>
<td align="center"><a href="https://drive.google.com/file/d/144mXXEP58hWS1y9RlNo0ThbpdThxT191/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1slP1eln4qq-dG_piLZ8TlGqvR_Ms8a7G/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">CIFAR-100 (IF 200)</td>
<td align="left">ResNet-32</td>
<td align="left">BALMS</td>
<td align="center">48.0</td>
<td align="center"><a href="https://drive.google.com/file/d/1Qc4-F1qFu6ebVEeZSJMdnE_DnMh_siit/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1t4uA6jqMeoeoz_UhgTA_Iog3ZhchX6P7/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">ImageNet-LT</td>
<td align="left">ResNet-10</td>
<td align="left">Softmax</td>
<td align="center">34.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1QWoj418KRhW5JhnTk-Mu_a5ram0V9X9V/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1llnGZWHv7Kt5c7VE1SQbywmC_qPnhI5d/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">ImageNet-LT</td>
<td align="left">ResNet-10</td>
<td align="left">BALMS</td>
<td align="center">41.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1v6G1xGkku5px4tombqtR8xJI-Qj1F0dI/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1EUfTFVocx59CZigElUALgvR6OYoiuhtD/view?usp=sharing">log</a></td>
</tr>
<tr>
<td align="left">Places-LT</td>
<td align="left">ResNet-152</td>
<td align="left">Softmax</td>
<td align="center">30.2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/classifier-balancing/Places_LT/models/resnet152_uniform.pth">model</a>&nbsp;|&nbsp;<a href="">log</a></td>
</tr>
<tr>
<td align="left">Places-LT</td>
<td align="left">ResNet-152</td>
<td align="left">BALMS</td>
<td align="center">38.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1hPW6CrXmBpvinU1rhmp2TpnkUNl9NUf0/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1jpcM2Su3YVJke8o78gluUEPEvua0Z4kH/view?usp=sharing">log</a></td>
</tr>

</tbody></table>


## Cite BALMS
```bibtex
@inproceedings{
    Ren2020balms,
    title={Balanced Meta-Softmax for Long-Tailed Visual Recognition},
    author={Jiawei Ren and Cunjun Yu and Shunan Sheng and Xiao Ma and Haiyu Zhao and Shuai Yi and Hongsheng Li},
    booktitle={Proceedings of Neural Information Processing Systems(NeurIPS)},
    month = {Dec},
    year={2020}
}
```

## Instance Segmentation

For BALMS on instance segmentation, please try out this [**repo**](https://github.com/Majiker/BalancedMetaSoftmax-InstanceSeg).

## Reference 
- The code is based on [classifier-balancing](https://github.com/facebookresearch/classifier-balancing).
- CIFAR-LT dataset is from [A Strong Single-Stage Baseline for Long-Tailed Problems](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch)
- ResNet-32 is from [BBN](https://github.com/Megvii-Nanjing/BBN)
- Cutout augmentation is from [Cutout](https://github.com/uoguelph-mlrg/Cutout)
- CIFAR auto augmentation is from [AutoAugment](https://github.com/DeepVoltaire/AutoAugment)

