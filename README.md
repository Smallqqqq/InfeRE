# Step-by-step Regex Generation via Chain of Inference
This is implemantation of the paper **Step-by-step Regex Generation via Chain of Inference**
## Summary
Automatically generating regular expressions from natural language description (NL2RE) has been an emerging research area. Prior works treat regex as a linear sequence of tokens and generate the final expressions autoregressively in a single pass. They did not take into account the step-by-step internal text-matching processes behind the final results. This significantly hinders the efficacy and interpretability of regex generation by neural language models. In this paper, we propose a new paradigm called InfeRE, which decomposes the generation of regexes into chains of step-by-step inference. To enhance the robustness, we introduce a self-consistency decoding mechanism that ensembles multiple outputs sampled from different models. Experimental studies on two public benchmarks demonstrate that InfeRE remarkably outperforms previous methods and achieves state-of-the-art performance.

## Install from source
pip install -r requirements.txt

## train
