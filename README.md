# Step-by-step Regex Generation via Chain of Inference
This is implemantation of the paper **Step-by-step Regex Generation via Chain of Inference**
## Summary
Automatically generating regular expressions from natural language description (NL2RE) has been an emerging research area. Prior works treat regex as a linear sequence of tokens and generate the final expressions autoregressively in a single pass. They did not take into account the step-by-step internal text-matching processes behind the final results. This significantly hinders the efficacy and interpretability of regex generation by neural language models. In this paper, we propose a new paradigm called InfeRE, which decomposes the generation of regexes into chains of step-by-step inference. To enhance the robustness, we introduce a self-consistency decoding mechanism that ensembles multiple outputs sampled from different models. Experimental studies on two public benchmarks demonstrate that InfeRE remarkably outperforms previous methods and achieves state-of-the-art performance.

## Chains of Inference
We convert plain regexes into chains of inference, each representing an inferred sub-regex that denotes a text-matching process. Then, we train a sequence-to-sequence model to generate chains of inference from natural language queries and revert them into regexes.

First, we parse the original plain regex into trees based on predefined rules(Read paper for details). We then traverse the tree in a post-order. Whenever we encounter an operator node, we regard the sub-regex corresponding to its sub-tree as a step in the chain of inference. For the i-th node, we represent the sub-regex as step i in the chain of inference. Then, we replace the sub-tree of the current operator with a step-i node. The size of the regex tree decreases 
continuously as the traversal proceeds. We repeat this process until there is only a single step node left in the tree, which means the completion of the chain of inference.
![image](https://github.com/Smallqqqq/InfeRE/blob/main/chain.png)

## Data
Train/valid/test data is under **./data**

## Install from source
'''pip install -r requirements.txt'''
## Train
'
source train.sh
'
## Eval
'
source eval.sh
'
