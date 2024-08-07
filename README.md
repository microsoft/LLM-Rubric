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
   
# Paper Experiments

## Real Conversation Data

### Baselines

To compute baselines, run `bash scripts/run_real_data_baselines.sh`.

It will print out results for each criteria like so:

```sh
                              rmse   pearson  spearman   kendall    N      mean           std
criterion system
Q1        random          1.589779 -0.087296 -0.072304 -0.061741  146  2.390411  1.124894e+00
          group_constant  0.900420       NaN       NaN       NaN  146  3.213605  0.000000e+00
          ann_constant    1.055296 -0.008433 -0.017450 -0.011765  146  3.044042  5.892783e-01
          sample_llm      1.069503 -0.026344 -0.036366 -0.032842  146  2.773973  5.706347e-01
          argmax_llm      0.914121 -0.198756 -0.226180 -0.211648  146  2.972603  1.632380e-01
          expected_llm    0.894389  0.040721  0.048520  0.036537  146  2.773374  1.292079e-01
Q2        random          1.630703 -0.036446  0.012197  0.012741  223  2.484305  1.103795e+00
          group_constant  0.947655       NaN       NaN       NaN  223  3.397611  8.881784e-16
          ann_constant    0.785368  0.547249  0.455538  0.375845  223  3.259603  3.976770e-01
          sample_llm      1.161799 -0.139302 -0.172219 -0.160052  223  2.843049  4.898372e-01
          argmax_llm      0.965782 -0.058163 -0.068355 -0.064493  223  2.991031  1.336293e-01
          expected_llm    1.004312 -0.078251 -0.050007 -0.037454  223  2.864326  1.398593e-01
...
```

**TODO** Add definition of baselines.

### Train LLM-Rubric on synthetic data and evaluate on real data.

1. Find the best hyperparemeters on the synthetic data. `bash scripts/find_hyperparams_for_real_data_eval.sh`
2. Train LLM-Rubric with best hyperparameters on synthetic data. `bash scripts/train_llm_rubric_for_real_data_eval.sh`
3. Predict LLM-Rubric on real data. `bash scripts/predict_llm_rubric_for_real_data_eval.sh`. **TODO** separate this into two steps, 1 step to write predictions to file, and 2nd step to run evaluate.py on results.


## Data Preprocessing

Note: This section is for recording the experiment process for reproducibility.
These steps have already be done and you should just use the preprocessed data
in `data/synth_data` and `data/real_data/`.

### Real Human Conversations Data

Raw data collected from the dialogue annotation interface is located in `data/real_data/raw_real_data.pkl`. To extract a tsv file run 

```bash
python scripts/extract_db_convs.py \
  --raw-data-pkl data/real_data/raw_real_data.pkl \
  --output-path data/real_data/human_judges_real_convs.tsv
```


Run fixing script to make question numbering consistent with paper (ie Q0 was originally Q11)... TBD


Anonymize the annotator names for privacy.

```bash
python scripts/anon_data.py data/synth_data/human_judges_synth_all_FIXED.tsv data/real_data/human_judges_real_convs_FIXED.tsv 
```


# Project


> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

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
