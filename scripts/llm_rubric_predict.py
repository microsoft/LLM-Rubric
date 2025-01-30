from pathlib import Path
import random
from typing_extensions import Annotated
import json

import typer
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats 

import torch

from llm_rubric import kfold_utils, pd_utils
from llm_rubric.model.torch_impl import (
    Hyperparameters, PersonalizedCalibrationNetwork, pretrain_loop, finetune_loop
)


def main(
    test_text_ids_path: Annotated[Path, typer.Option("--ids")],
    human_path: Annotated[Path, typer.Option("--human-judgments")],
    machine_path: Annotated[Path, typer.Option("--machine-evaluations")],
    model_path: Annotated[Path, typer.Option(default=...)],
    judge_map_path: Annotated[Path, typer.Option(default="--judge-map")],
    text_id_column: str = "text_id",
    criterion_column: str = "criterion",
    input_criteria: list[str] = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q0"],
    output_criteria: list[str] = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q0"],
    num_answers: int = 4,
    judge_column: str = "annotator_name", 
    judge_id_column: str = "judge_id", 
    random_seed: int = 43,
    all_data_size: int = typer.Option(None),
    layer1_size: int = typer.Option(None),
    layer2_size: int = typer.Option(None),
    batch_size: int = typer.Option(None),
    learning_rate: float = typer.Option(None),
    pretraining_epochs: int = typer.Option(None),
    finetuning_epochs: int = typer.Option(None),
): 
    text_ids = pd.read_csv(test_text_ids_path, sep="\t")[text_id_column].tolist()
    human_df = pd.read_csv(human_path, sep="\t")
    human_df = human_df[human_df["Q0"] != 0]
    rowwise_machine_df = pd.read_csv(machine_path, sep="\t")
    machine_df = pd_utils.join_rowwise_predictions(
        rowwise_machine_df,
        text_id_column,
        criterion_column,
        input_criteria,
        num_answers
    ) 
    print(f"Loaded {len(human_df)} human judgments from {human_path}")
    print(f"Loaded {len(rowwise_machine_df)} machine evaluations from {machine_path}")

    data_df = human_df.merge(machine_df, on=text_id_column, how="left")
#    data_df[fold_column] = data_df[text_id_column].apply(fold_map.get)
    data_df = data_df.dropna()


    with judge_map_path.open("r") as fh:
        judge_id_map = json.load(fh)

    pd_utils.add_judge_ids(data_df, judge_column, judge_id_column, judge_id_map=judge_id_map)
    num_judges = len(judge_id_map)
    print("Number of judges")
    print(num_judges)

    data_df = data_df[data_df[text_id_column].isin(text_ids)] 
    print(f"Total rows: {len(data_df)}")


    print(len(output_criteria), num_answers)
    input_size = len(input_criteria) * num_answers # TODO allow for different number of outputs per rubric item.
    output_size = len(output_criteria)


    hp_args = {}
#    if all_data_size is not None:
#        hp_args["all_data_size"] = all_data_size
    if layer1_size is not None:
        hp_args["layer1_size"] = layer1_size
    if layer2_size is not None:
        hp_args["layer2_size"] = layer2_size
    if batch_size is not None:
        hp_args["batch_size"] = batch_size
    if learning_rate is not None:
        hp_args["learning_rate"] = learning_rate
    if pretraining_epochs is not None:
        hp_args["pretraining_epochs"] = pretraining_epochs
    if finetuning_epochs is not None:
        hp_args["finetuning_epochs"] = finetuning_epochs

    hp = Hyperparameters(
        input_size=input_size,
        output_size=output_size,
        num_judges=num_judges,
        all_data_size=len(data_df),
        **hp_args,
    )
    model = PersonalizedCalibrationNetwork(hp)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ds_test = pd_utils.make_dataset(data_df, input_criteria, judge_id_column, output_criteria)
    
    # print("test Loss (FT)", model.loss(ds_test.X, ds_test.A, ds_test.Y, I=[-1]).detach().numpy()) 
    yhat = model.decode(ds_test.X, ds_test.A, I=[-1])[:, 0].detach().numpy()
    y = ds_test.Y[:,-1].detach().numpy()
    # rmse = metrics.root_mean_squared_error(y, yhat) -- For RMSE calculation, make sure to train the model with the rmse loss.
    p_corr, _ = stats.pearsonr(y, yhat)
    s_corr, _ = stats.spearmanr(y, yhat)
    t_corr, _ = stats.kendalltau(y, yhat)
    # print("test rmse", rmse) -- see the note above about RMSE calculation 
    print("test pearsonr", p_corr) 
    print("test spearmanr", s_corr) 
    print("test kendallt", t_corr) 


if __name__ == "__main__":
    typer.run(main)
