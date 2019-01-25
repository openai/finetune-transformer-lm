## `finetune-transformer-lm`: Code for Improving Language Understanding by Generative pre-Training

This project contains the code and model for the paper ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). 

**Note: This project is no longer actively developed. This code is provided as-is, and no updates are expected.**

From the abstract:

> We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task... we achieve absolute improvements of 8.9% on commonsense reasoning (Stories Cloze Test), 5.7% on question answering (RACE), and 1.5% on textual entailment (MultiNLI).

The blog post describing this work is ["Improving Language Understanding with Unsupervised Learning"](https://blog.openai.com/language-unsupervised/). 

Authors: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever

### License

This code is Copyright OpenAI and published under the MIT License.

### Requirements

This code is verified to run on Python 2.7 and 3.3.6 in a clean conda environment. It requires the following modules:

```
ftfy
joblib
numpy
pandas
sklearn
spacy
tensorflow
tqdm
```

### Setup

To install requirements, run:

```
pip install -r requirements.txt
```

#### Python 2.7

Note: `tqdm` requires `ftfy`, which dropped Python 2 support after version 4.4.3. This is handled in `requirements.txt`.

#### Spacy Models

You need to download the `en` model for spacy:

```
python -m spacy download en
```

#### Data

You need to download all of the ROCStories 2016 datasets to train the model. Doing so requires filling out a form so the data's creators can track who is using it. They can be found at [website](http://cs.rochester.edu/nlp/rocstories/). The location of the data is a command line argument.

Once you've downloaded them, the files should look something like this:

```
data/ROCStories__spring2016 - ROCStories_spring2016.csv
data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv
data/cloze_test_test__spring2016 - cloze_test_ALL_test.tsv
data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv
data/cloze_test_val__spring2016 - cloze_test_ALL_val.tsv
```

#### Model

The model is precomputed and stored in the [`model`](model) directory.

### Training

Currently this code implements the ROCStories Cloze Test result reported in the paper by running:

```
python train.py --dataset rocstories --desc rocstories --submit --analysis --data_dir [path to data here]
```

You can put the data files anywhere and change the `--data-dir` value.

Note: The code is currently non-deterministic due to various GPU ops. The median accuracy of 10 runs with this codebase (using default hyperparameters) is 85.8% - slightly lower than the reported single run of 86.5% from the paper. 

### Testing

This code was tested by training the model in Python 2.7 and 3 on Ubuntu Linux 17.10/artful with the 4.13.0-46-generic kernel. Each of the two Python processes consumed 24GB of RAM (12GB remained free) on a 12 core 64GB/RAM machine with a GTX 1080, and used all CPU 12 cores in addition to the GPUs. It ran overnight. 

Your results may vary.

