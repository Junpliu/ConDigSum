## Topic-Aware Contrastive Learning for Abstractive Dialogue Summarization
Junpeng Liu, Yanyan Zou, Hainan Zhang, Hongshen Chen, Zhuoye Ding, Caixia Yuan, Xiaojie Wang ```EMNLP 2021``` [Paper](https://arxiv.org/abs/2109.04994)  

### Requirements and Installation
Conda is highly recommended to manage your Python environment. 
* Python 3.6
* Pytorch >= 1.6.0
* [Files2ROUGE](https://github.com/pltrdy/files2rouge)

```
pip install --editable ./
pip install requests rouge==1.0.0
pip install transformers==4.4.0 bert-score==0.3.8
```


### Training ```ConDigSum``` model
Before training ConDigSum, please download BART-Large from [here](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz), and update ```PRETRAIN_PATH``` to the path of ```model.pt``` in training scripts.

For SAMSum and MediaSum datasets, you can download preprocessed data files directly 
([SAMSum](https://drive.google.com/file/d/1IzdkmuVQfhrH-D_WuLV5uKQds-vus2P8/view?usp=sharing), [MediaSum](https://drive.google.com/file/d/15VxjmyHlkLH4GHgOufYUHtTVt53oJwe5/view?usp=sharing)), 
which results in ```train_sh/SAMSumInd/``` and ```train_sh/mediasum/```. 

Change working directory and download: 
```
cd train_sh
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```
#### SAMSum dataset
```
# SAMSum
./train_samsum.sh [training_comment] [gpu_id]
```
#### MediaSum dataset
```
# MediaSum
./train_mediasum.sh [training_comment] [gpu_id]
```
#### Custom dataset 
To facilitate training on custom datasets, a demo dataset is provided in ```train_sh/customdata/``` directory, please prepare your own data files following the ```*.jsonl``` files. Then, pre-processing steps are as follows:

```
./bpe.sh
./binarize.sh
```


### Testing ```ConDigSum``` model

#### Downloading pretrained ConDigSum models
Pretrained models and predictions are provided at Google Drive: [SAMSum](https://drive.google.com/file/d/1t6ATjT4r8_wpVWDKT36Vu8NmMLY_oSRD/view?usp=sharing), [MediaSum](https://drive.google.com/file/d/1WwH2hFlLfhsrxbinG21X4EWy78dQ10PW/view?usp=sharing).
After downloading, ```train_sh/SAMSum.condigsum/checkpoint_best.pt``` and ```train_sh/MediaSum.condigsum/checkpoint_best.pt``` will be gotten.
#### Evaluating models
```
# dataname=SAMSumInd or dataname=mediasum or dataname=customdata
# checkpoint_dir=SAMSum.condigsum or checkpoint_dir=MediaSum.condigsum

# generate predictions
cd train_sh
CUDA_VISIBLE_DEVICES=${GPU} python ./test.py --log_dir ${checkpoint_dir} --dataset ${dataname}

# get file2rouge scores
files2rouge ${dataname}/test.target ${checkpoint_dir}/test.hypo

# calculate bert-score scores
CUDA_VISIBLE_DEVICES=${GPU} bert-score -r ${dataname}/test.target -c ${checkpoint_dir}/test.hypo --lang en --rescale_with_baseline
```
### Citation
```
@inproceedings{liu-etal-2021-topic-aware,
    title = "Topic-Aware Contrastive Learning for Abstractive Dialogue Summarization",
    author = "Liu, Junpeng  and
      Zou, Yanyan  and
      Zhang, Hainan  and
      Chen, Hongshen  and
      Ding, Zhuoye  and
      Yuan, Caixia  and
      Wang, Xiaojie",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.106",
    doi = "10.18653/v1/2021.findings-emnlp.106",
    pages = "1229--1243",
    abstract = "Unlike well-structured text, such as news reports and encyclopedia articles, dialogue content often comes from two or more interlocutors, exchanging information with each other. In such a scenario, the topic of a conversation can vary upon progression and the key information for a certain topic is often scattered across multiple utterances of different speakers, which poses challenges to abstractly summarize dialogues. To capture the various topic information of a conversation and outline salient facts for the captured topics, this work proposes two topic-aware contrastive learning objectives, namely coherence detection and sub-summary generation objectives, which are expected to implicitly model the topic change and handle information scattering challenges for the dialogue summarization task. The proposed contrastive objectives are framed as auxiliary tasks for the primary dialogue summarization task, united via an alternative parameter updating strategy. Extensive experiments on benchmark datasets demonstrate that the proposed simple method significantly outperforms strong baselines and achieves new state-of-the-art performance. The code and trained models are publicly available via .",
}
```

### MISC

1. To install Files2ROUGE on centos system, you may need to install dependencies to avoid some problems. 
```
yum install -y "perl(XML::Parser)"
yum install -y "perl(XML::LibXML)"
yum install -y "perl(DB_File)"
```