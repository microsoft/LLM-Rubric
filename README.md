# LLM-Rubric
LLM-Rubric introduces a framework for the automated evaluation of natural language texts. A manually constructed rubric describes how to assess multiple dimensions of interest. To evaluate a text, a large language model (LLM) is prompted with each rubric question and produces a distribution over potential responses. The LLM predictions often fail to agree well with human judges—indeed, the humans do not fully agree with one another. However, the multiple LLM distributions can be combined to predict each human judge's annotations on all questions, including a summary question that assesses overall quality or relevance. LLM-Rubric accomplishes this by training a small feed-forward neural network that includes both judge-specific and judge-independent parameters. When evaluating dialogue systems in a human-AI information-seeking task, we find that LLM-Rubric with 9 questions (assessing dimensions such as naturalness, conciseness, and citation quality) predicts human judges' assessment of overall user satisfaction, on a scale of 1–4.

For more information, please read the [LLM-Rubric paper](https://aclanthology.org/2024.acl-long.745/) published at ACL 2024.

**Disclaimer**: this repository was created after the original experimentation for the LLM-Rubric paper. By running these codes, you will experience some minor performance differences with the ones reported in the paper due to the updated data sampling and reimplementation in Pytorch (the results presented in the paper are based on a TensorFlow implementation of LLM-Rubric). These minor performance differences do not impact the findings, claims, and generality of the results presented in the original paper.

# Citation Information
Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, Chris Kedzie. "LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts" In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13806–13834, 2024.

```
@inproceedings{hashemi-etal-2024-llm,
    title = "{LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts}",
    author = "Hashemi, Helia  and
      Eisner, Jason  and
      Rosset, Corby  and
      Van Durme, Benjamin  and
      Kedzie, Chris",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.745/",
    doi = "10.18653/v1/2024.acl-long.745",
    pages = "13806--13834",
}
```



# Installation

1. Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).
2. Create and activate a virtual environment with `python 3.10`. For example with conda: 
```bash
conda create -n llm-rubric python=3.10
conda activate llm-rubric
```
**Make sure your virtual environment is activated when performing steps 3 and 4.**

3. Install `pytorch 2.3.0`.
```bash

# OSX
pip install torch==2.3.0

# Linux and Windows
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```
4. Install LLM-Rubric.
```bash
git clone https://github.com/microsoft/LLM-Rubric.git
cd LLM-Rubric
poetry install
```
   
# Experiments

## Real Conversation Data

### Baselines

To compute baselines, run `bash scripts/run_real_data_baselines.sh`.

It will print out the results as follows:

```sh
                            rmse   pearson  spearman   kendall    N      mean       std
criterion system
Q0        random        1.453309 -0.054438 -0.043663 -0.036858  223  2.587444  1.152196
          sample_llm    1.207228  0.039810  0.045245  0.040967  223  3.322870  0.794469
          argmax_llm    1.201643  0.140091  0.086990  0.081134  223  3.614350  0.563597
          expected_llm  0.918676  0.177301  0.086675  0.065928  223  3.282864  0.300516
```

Please refer to the LLM-Rubric paper for the definition of baseline methods.

### Train LLM-Rubric on synthetic data and evaluate on real data.

#### Hyperparameter setting (no action needed)
The best hyperparameters based on the synthetic data are stored in `experiments/real_data/best_synth_cross-validation_hp.json`. This file contains the following selected hyperparameters:
```sh
{"input_size": 36, "output_size": 9, "num_judges": 13, "all_data_size": 223, "finetune_output": -1, "num_answers": 4, "batch_size": 64, "learning_rate": 0.001, "layer1_size": 25, "layer2_size": 25, "pretraining_epochs": 20, "finetuning_epochs": 30, "random_seed": 43}
```
If you try to find the best hyperparameters on your own data, you can use the following script: `bash scripts/find_hyperparams_for_real_data_eval.sh`

#### Training LLM-Rubric on synthetic data
Train LLM-Rubric with best hyperparameters on synthetic data using the following script: `bash scripts/train_llm_rubric_for_real_data_eval.sh`

#### Prediction and evaluation
Predict LLM-Rubric on real data and evaluate the results using the following script: `bash scripts/predict_llm_rubric_for_real_data_eval.sh`.
This step produces the following output:
```sh
Number of judges
24
Total rows: 223
9 4
test pearsonr 0.31304019363805424
test spearmanr 0.367664375396213
test kendallt 0.29084198763627717
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
