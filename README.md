# finetune-transformer-lm
Code and model for the paper ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).

This research is also documented in blog post format on the [OpenAI blog](https://blog.openai.com/language-unsupervised/).

Installing Dependencies
------------

```bash
pip install -r requirements.txt
```

Usage
-----
Currently this code implements the ROCStories Cloze Test result reported in the paper by running:

```bash
python3 -m finetune.train --dataset rocstories --desc rocstories --submit --analysis --data_dir [data_path]
```

The ROCStories dataset can be downloaded from the University of Rochester [ROCStories Corpora webpage](http://cs.rochester.edu/nlp/rocstories).

Note: The code is currently non-deterministic due to various GPU ops. The median accuracy of 10 runs with this codebase (using default hyperparameters) is 85.8% - slightly lower than the reported single run of 86.5% from the paper.
