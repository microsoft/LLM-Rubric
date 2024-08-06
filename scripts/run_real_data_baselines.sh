# Real Data Baselines

### LLM Baselines:
###   sample_llm
###   argmax_llm
###   expected_llm
python scripts/compute_machine_baseline.py \
  --machine-evaluations data/real_data/gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv \
  --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
  --output-path experiments/real_data/predictions/real_convs_machine_baselines.tsv 


### Random Baseline:
###   random
python scripts/compute_random_baseline.py \
  --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
  --output-path experiments/real_data/predictions/real_convs_random_baseline.tsv


### Human Baselines: 
###   Overall Mean (group_constant)
###   Per Annotator Mean (ann_constant)
python scripts/compute_human_baseline.py \
  --train-human-judgments data/synth_data/human_judges_synth_all_FIXED_ANON.tsv \
  --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
  --output-path experiments/real_data/predictions/real_convs_human_baselines.tsv


### Compute Metrics Results
python scripts/evaluate.py \
    experiments/real_data/predictions/real_convs_random_baseline.tsv \
    experiments/real_data/predictions/real_convs_human_baselines.tsv \
    experiments/real_data/predictions/real_convs_machine_baselines.tsv \
    --human-judgments data/real_data/human_judges_real_convs_FIXED_ANON.tsv \
    --systems "random,group_constant,ann_constant,sample_llm,argmax_llm,expected_llm" 
