ALL_DATA_SIZE=`cat experiments/real_data/best_synth_cross-validation_hp.json | jq .all_data_size`
LAYER1_SIZE=`cat experiments/real_data/best_synth_cross-validation_hp.json | jq .layer1_size`
LAYER2_SIZE=`cat experiments/real_data/best_synth_cross-validation_hp.json | jq .layer2_size`
LEARNING_RATE=`cat experiments/real_data/best_synth_cross-validation_hp.json | jq .learning_rate`
BATCH_SIZE=`cat experiments/real_data/best_synth_cross-validation_hp.json | jq .batch_size`
PT_EPOCHS=`cat experiments/real_data/best_synth_cross-validation_hp.json | jq .pretraining_epochs`
FT_EPOCHS=`cat experiments/real_data/best_synth_cross-validation_hp.json | jq .finetuning_epochs`

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
  --model-path experiments/real_data/llm_rubric_full_q0_model_params.pkl \
  --judge-map experiments/real_data/judge_map.json \
  --judge-column annotator_id
