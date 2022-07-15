# Improve-Discourse-Dependency-Parsing-with-Contextualized-Representations
Implementation of the paper 'Improve Discourse Dependency Parsing with Contextualized Representations'

Findings of NAACL 2022

The state-of-the-art (2022) transformer-based multiple-levels discourse dependency parser.

Paper link:
https://arxiv.org/abs/2205.02090

![DDP model overview](https://user-images.githubusercontent.com/83000332/165659676-c641cc42-6500-44ce-afec-b64cfd8192d9.png)


## Requirements
* GPU
* python
* pyTorch
* transformers
* pickle
* tqdm

## Results
This simple ready to use discourse parsing framework achieves the following state-of-the-art results:
| Dataset     | UAS         | LAS     |
| :---        |    :----:   |          ---: |
| CDTB        |    82.2     | 64.8   |
| SciDTB      |  80.2       |   65.4   |

## Dataset -preprocess
### Download Dataset
* [SciDTB](https://github.com/PKU-TANGENT/SciDTB/tree/master/dataset): public in github repository
* [CDTB](https://arxiv.org/abs/2101.00167): Please contact the author for dataset

### Preprocess Dataset
We provide script for preprocessing CDTB and SciDTB. It's also easy to adapt the code in <code>preprocess_dataset.py</code> for your own application
* <code>mkdir preprocessed_data</code>
* For SciDTB: <code>python3 preprocess_dataset.py --dataset scidtb --path [your path to downloaded scidtb dataset] --target_dir preprocessed_data</code>
* For CDTB: <code>python3 preprocess_dataset.py --dataset cdtb --path [your path to downloaded cdtb dataset] --target_dir preprocessed_data</code>

## Replicate the Results
To replicate the main results shown in the paper, we provide our trained models [here](https://drive.google.com/drive/folders/1NXbtM9HbZcJrN-Ymj57dGvoLC-fOPxTM?usp=sharing). Put the folder <code>trained_models</code> in the root directory.
* For SciDTB, run: <code>python3 evaluate.py --dataset scidtb --path_test_data preprocessed_data/sci_test.data --path_in_sentence_model trained_models/SciDTB/sciDTB_in_sentence.pt --path_between_sentence_model trained_models/SciDTB/sciDTB_between_sentence.pt --path_relation_bert trained_models/SciDTB/sciDTB_relation_bert.pt --path_between_bert trained_models/SciDTB/sciDTB_between_bert.pt --path_lstm_tagger trained_models/SciDTB/sciDTB_lstm_tagger.pt --path_between_tagger trained_models/SciDTB/sciDTB_between_tagger.pt</code>

* To replicate only UAS parsing results (i.e. no labels), run: <code>python3 evaluate.py --tokenizer_name bert-base-chinese --model_name bert-base-chinese --path_in_sentence_model trained_models/CDTB/cdtb_in_sentence.pt --path_between_sentence_model trained_models/CDTB/cdtb_between_sentence.pt  --path_test_data preprocessed_data/cdtb_test.data --relation_labeling_option none </code>

## Train your own model
We provide a sample script for training the whole architecture for SciDTB:
run: <code>./run.sh</code>
Note that this is only a single training iteration. To achieve the best performance, each module will need to be validated separately.

## Cite our paper
```bibtex
@inproceedings{zhou-feng-2022-improve,
    title = "Improve Discourse Dependency Parsing with Contextualized Representations",
    author = "Zhou, Yifei  and
      Feng, Yansong",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.173",
    pages = "2250--2261",
    abstract = "Previous works show that discourse analysis benefits from modeling intra- and inter-sentential levels separately, where proper representations for text units of different granularities are desired to capture both the information of the text units and their relation to the context. In this paper, we propose to take advantage of transformers to encode different contextualized representations of units of different levels to dynamically capture the information required for discourse dependency analysis on intra- and inter-sentential levels. Motivated by the observation of writing patterns shared across articles to improve discourse analysis, we propose to design sequence labeling methods to take advantage of such structural information from the context that substantially outperforms traditional direct classification methods. Experiments show that our model achieves state-of-the-art results on both English and Chinese datasets.",
}
```
