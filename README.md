


# Installation

1. Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).
2. Create and activate a virtual environment with `python 3.10`. For example with conda: 
```bash
conda create -n llm-rubric python=3.10
conda activate llm-rubric
```
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


## Real Conversation Data

### Baselines

```bash
## LLM Baselines: sample_llm, argmax_llm, expected_llm ##

python scripts/compute_machine_baseline.py \
  --machine-evaluations data/real_data/gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv \
  --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
  --output-path data/real_data/predictions/real_convs_machine_baselines.tsv 

## Random Baseline: random ##

python scripts/compute_random_baseline.py \
  --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
  --output-path data/real_data/predictions/real_convs_random_baseline.tsv

## Human Baselines: Overall Mean (group_constant), Per Annotator Mean (ann_constant) ##

python scripts/compute_human_baseline.py \
  --train-human-judgments data/synth_data/human_judges_synth_all_FIXED_ANON.tsv \
  --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
  --output-path data/real_data/predictions/real_convs_human_baselines.tsv

## Compute Metrics Results ##

python scripts/evaluate.py \
    data/real_data/predictions/real_convs_random_baseline.tsv \
    data/real_data/predictions/real_convs_human_baselines.tsv \
    data/real_data/predictions/real_convs_machine_baselines.tsv \
    --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
    --systems "random,group_constant,ann_constant,sample_llm,argmax_llm,expected_llm" 
```

### Train Calibration Network 

First find the best parameter settings via cross validation on the synthetic data.

```bash
# This might speed things up:
# EXPORT OMP_NUM_THREADS=2

python scripts/cross_validate_llm_rubric.py \
    --ids data/synth_data/synth_dialogue_ids.tsv \
    --human-judgments data/synth_data/human_judges_synth_all_FIXED_ANON.tsv \
    --machine-evaluations data/synth_data/gpt-3.5-turbo-16k_synth_evaluations_FIXED.tsv \
    --output-path data/real_data/synth_cross-validation_hps/ \
    --judge-column annotator_id \
    --num-procs 4

python scripts/find_best_hyperparams.py \
  --hps data/real_data/synth_cross-validation_hps/ \
  --output data/real_data/synth_cross-validation_hps/best_hp.json

ALL_DATA_SIZE=`cat data/real_data/synth_cross-validation_hps/best_hp.json | jq .all_data_size`
LAYER1_SIZE=`cat data/real_data/synth_cross-validation_hps/best_hp.json | jq .layer1_size`
LAYER2_SIZE=`cat data/real_data/synth_cross-validation_hps/best_hp.json | jq .layer2_size`
LEARNING_RATE=`cat data/real_data/synth_cross-validation_hps/best_hp.json | jq .learning_rate`
BATCH_SIZE=`cat data/real_data/synth_cross-validation_hps/best_hp.json | jq .batch_size`
PT_EPOCHS=`cat data/real_data/synth_cross-validation_hps/best_hp.json | jq .pretraining_epochs`
FT_EPOCHS=`cat data/real_data/synth_cross-validation_hps/best_hp.json | jq .finetuning_epochs`

python scripts/train_llm_rubric.py \
  --ids data/synth_data/synth_dialogue_ids.tsv \
  --human-judgments data/synth_data/human_judges_synth_all_FIXED_ANON.tsv \
  --machine-evaluations data/synth_data/gpt-3.5-turbo-16k_synth_evaluations_FIXED.tsv \
  --all-data-size $ALL_DATA_SIZE \
  --layer1-size $LAYER1_SIZE \
  --layer2-size $LAYER2_SIZE \
  --learning-rate $LEARNING_RATE \
  --batch-size $BATCH_SIZE \
  --pretraining-epochs $PT_EPOCHS \
  --finetuning-epochs $FT_EPOCHS \
  --model-path data/real_data/model_params.pkl \
  --judge-map data/real_data/judge_map.json
```

# TODO
# predict on individual example
# predict batch data with model
# evaluate model predictions

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
