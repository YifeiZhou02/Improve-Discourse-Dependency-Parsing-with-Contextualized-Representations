# Improve-Discourse-Dependency-Parsing-with-Contextualized-Representations
Implementation of the paper 'Improve Discourse Dependency Parsing with Contextualized Representations'
Findings of NAACL 2022
The state-of-the-art transformer-based multiple-levels discourse dependency parser.

### Requirements
* GPU
* python
* pyTorch
* transformers
* pickle
* tqdm

### Results
This simple ready to use discourse parsing framework achieves the following state-of-the-art results:
| Dataset     | UAS         | LAS     |
| :---        |    :----:   |          ---: |
| CDTB        |    82.2     | 64.8   |
| SciDTB      |  80.2       |   65.4   |

### Dataset -preprocess
* [SciDTB](https://github.com/PKU-TANGENT/SciDTB/tree/master/dataset)
* [CDTB](https://arxiv.org/abs/2101.00167)
