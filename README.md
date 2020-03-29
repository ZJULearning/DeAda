# DeAda
## Introduction

DeAda is a new training method for person re-identification (Re-ID) networks. It is published recently on Neurocomputing - [Decouple Co-adaptation: Classifier Randomization for Person Re-identification](https://www.sciencedirect.com/science/article/abs/pii/S0925231219316972).
DeAda could decouple co-adaptation in Re-ID networks, so that performance of networks could be improved. DeAda does not increase computational cost during training and testing.

This project is the implementation of DeAda on some commonly used baseline networks. Our code is adapted from the open-reid library (https://github.com/Cysu/open-reid).

## Datasets
* [Market-1501](http://www.liangzheng.com.cn/Project/project_reid.html)
  
    Download using: 
        
      wget http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip <path/to/where/you/want>
      unzip <path/to/>/Market-1501-v15.09.15.zip
  
* [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)

  1. Download cuhk03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
  2. Unzip the file and you will get the cuhk03_release dir which include cuhk-03.mat
  3. Download "cuhk03_new_protocol_config_detected.mat" from [here](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03) and put it with cuhk-03.mat. We need this new protocol to split the dataset.
  
  NOTICE: You need to change num_classes in network depend on how many people in your train dataset! e.g. 751 in Market1501.

The data structure should look like:
    
  ```
  data/
      bounding_box_train/
      bounding_box_test/
      query/
      train.txt   
      val.txt
      query.txt
      gallery.txt
  ```
  Here each *.txt file consists lines of the format: image file name, person id, camera id.
  train.txt consists images from bounding_box_train/, val.txt and query.txt consists images from query/, and gallery.txt consists images from bounding_box_test/.

## RUN
### Prerequisites

+ cudnn 7
+ CUDA 9
+ Pytorch v0.4.1
+ Python 2.7
+ torchvision
+ scipy
+ numpy
+ scikit_learn

### Baseline ReID Methods

+ [ResNet](https://arxiv.org/abs/1512.03385). We choose two configurations: ResNet50 and ResNet152.
+ [DenseNet](https://arxiv.org/abs/1608.06993). We choose two configurations: DenseNet121 and DenseNet161.

### Train and Evaluate
We provie two training methods: plain (traditional SGD optimization) and deada (our proposed DeAda optimization). The training method could be specified by the argument training_method in run.sh

* Train and Evaluate by running
  ```
  bash run.sh
  ```

### Results
Evaluation metric: mAP (%) and CMC Rank-1 (%). 

<table>
  <tr>
    <th>Models + Training_method</th> 
    <th colspan="2">Market-1501</th>
    <th colspan="2">CUHK03</th>
    <th colspan="2">DukeMTMC-reID</th>
  </tr>
  <tr>
    <td></td>
    <td>mAP</td>
    <td>Rank-1</td>
    <td>mAP</td>
    <td>Rank-1</td>
    <td>mAP</td>
    <td>Rank-1</td>
  </tr>
  <tr>
    <td>ResNet50 + SGD</td>
    <td>70.9</td>
    <td>86.8</td>
    <td>41.5</td>
    <td>41.4</td>
    <td>62.6</td>
    <td>79.6</td>
  </tr>
  <tr>
    <td>ResNet50 + DeAda</td>
    <td>72.3</td>
    <td>87.7</td>
    <td>43.4</td>
    <td>44.4</td>
    <td>63.5</td>
    <td>80.5</td>
  </tr>
  <tr>
    <td>DenseNet121 + SGD</td>
    <td>73.2</td>
    <td>89.1</td>
    <td>40.5</td>
    <td>41.1</td>
    <td>64.7</td>
    <td>81.0</td>
  </tr>
  <tr>
    <td>DenseNet121 + DeAda</td>
    <td>76.9</td>
    <td>90.5</td>
    <td>41.8</td>
    <td>41.9</td>
    <td>66.9</td>
    <td>82.0</td>
  </tr>
  <tr>
    <td>ResNet152 + SGD</td>
    <td>74.8</td>
    <td>89.2</td>
    <td>47.1</td>
    <td>47.1</td>
    <td>65.9</td>
    <td>80.7</td>
  </tr>
  <tr>
    <td>ResNet152 + DeAda</td>
    <td>76.0</td>
    <td>89.4</td>
    <td>49.4</td>
    <td>51.7</td>
    <td>66.2</td>
    <td>81.6</td>
  </tr>
  <tr>
    <td>DenseNet161 + SGD</td>
    <td>75.6</td>
    <td>90.0</td>
    <td>45.5</td>
    <td>45.7</td>
    <td>66.7</td>
    <td>81.9</td>
  </tr>
  <tr>
    <td>DenseNet161 + DeAda</td>
    <td>78.6</td>
    <td>92.1</td>
    <td>46.4</td>
    <td>46.1</td>
    <td>68.7</td>
    <td>84.1</td>
  </tr>
</table>


## Reference

Please cite our paper when you use DeAda in your research:

Long Wei, Zhenyong Wei, Zhongming Jin, Qianxiao Wei, Jianqiang Huang, Xian-Sheng Hua, Deng Cai, and Xiaofei He. "Decouple co-adaptation: Classifier randomization for person re-identification." Neurocomputing 383 (2020): 1-9.
