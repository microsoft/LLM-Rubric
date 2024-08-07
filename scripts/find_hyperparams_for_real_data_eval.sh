# This might speed things up:
export OMP_NUM_THREADS=2

python scripts/cross_validate_llm_rubric.py \
    --ids data/synth_data/synth_dialogue_ids.tsv \
    --human-judgments data/synth_data/human_judges_synth_all_FIXED_ANON.tsv \
    --machine-evaluations data/synth_data/gpt-3.5-turbo-16k_synth_evaluations_FIXED.tsv \
    --output-path experiments/real_data/synth_cross-validation_hps/ \
    --judge-column annotator_id \
    --num-procs 4

python scripts/find_best_hyperparams.py \
  --hps experiments/real_data/synth_cross-validation_hps/ \
  --output experiments/real_data/best_synth_cross-validation_hp.json
