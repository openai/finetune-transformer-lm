**Status:** Archive (code is provided as-is, no updates expected)

# finetune-transformer-lm
Code and model for the paper "Improving Language Understanding by Generative Pre-Training"

Currently this code implements the ROCStories Cloze Test result reported in the paper by running:
`python train.py --dataset rocstories --desc rocstories --submit --analysis --data_dir [path to data here]`

Note: The code is currently non-deterministic due to various GPU ops. The median accuracy of 10 runs with this codebase (using default hyperparameters) is 85.8% - slightly lower than the reported single run of 86.5% from the paper. 

The ROCStories dataset can be downloaded from the associated [website](http://cs.rochester.edu/nlp/rocstories/).
