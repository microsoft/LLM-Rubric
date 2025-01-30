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
    train_text_ids_path: Annotated[Path, typer.Option("--ids")],
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
    text_ids = pd.read_csv(train_text_ids_path, sep="\t")[text_id_column].tolist()
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
    random.seed(random_seed)
    np.random.seed(random_seed)
    random.shuffle(text_ids)
    fold_map = kfold_utils.make_fold_map(text_ids)

#    fold_column = "fold"
#    num_folds = 5
#    data_df = human_df.merge(machine_df, on=text_id_column, how="left")
#    data_df[fold_column] = data_df[text_id_column].apply(fold_map.get)
#    data_df = data_df.dropna()

    data_df = human_df.merge(machine_df, on=text_id_column, how="left")
#    data_df[fold_column] = data_df[text_id_column].apply(fold_map.get)
    data_df = data_df.dropna()

    judge_id_map = pd_utils.add_judge_ids(data_df, judge_column, judge_id_column)
    num_judges = len(judge_id_map)
    print("Number of judges")
    print(num_judges)

    data_df = data_df[data_df[text_id_column].isin(text_ids)] 
    print(f"Total rows: {len(data_df)}")
    
    print(len(output_criteria), num_answers)
    input_size = len(input_criteria) * num_answers # TODO allow for different number of outputs per rubric item.
    output_size = len(output_criteria)


    hp_args = {}
    if all_data_size is not None:
        hp_args["all_data_size"] = all_data_size
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
        #all_data_size=len(data_df),
        **hp_args,
    )
    model = PersonalizedCalibrationNetwork(hp)

#    kfold_iter = kfold_utils.train_test_dataframe_iter(data_df, fold_column=fold_column, num_folds=num_folds)
#    for fold, (train_df, test_df) in enumerate(kfold_iter):
#        print("LEN", len(train_df), len(test_df))
#        ds_train = pd_utils.make_dataset(train_df, input_criteria, judge_id_column, output_criteria)
#        ds_test = pd_utils.make_dataset(test_df, input_criteria, judge_id_column, output_criteria)
#        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
#        
#        print("Pretraining")
#        pretrain_loop(model, ds_train, optimizer)
#        print("Finetuning")
#        finetune_loop(model, ds_train, optimizer)
#        
#        print("Train Loss (FT)", model.loss(ds_train.X, ds_train.A, ds_train.Y, I=[-1]).detach().numpy()) 
#        yhat = model.decode(ds_train.X, ds_train.A, I=[-1])[:, 0].detach().numpy()
#        y = ds_train.Y[:,-1].detach().numpy()
#        rmse = metrics.root_mean_squared_error(y, yhat)
#        p_corr, _ = stats.pearsonr(y, yhat)
#        s_corr, _ = stats.spearmanr(y, yhat)
#        t_corr, _ = stats.kendalltau(y, yhat)
#        print("Train rmse", rmse) 
#        print("Train pearsonr", p_corr) 
#        print("Train spearmanr", s_corr) 
#        print("Train kendallt", t_corr) 
#
#        print("test Loss (FT)", model.loss(ds_test.X, ds_test.A, ds_test.Y, I=[-1]).detach().numpy()) 
#        yhat = model.decode(ds_test.X, ds_test.A, I=[-1])[:, 0].detach().numpy()
#        y = ds_test.Y[:,-1].detach().numpy()
#        rmse = metrics.root_mean_squared_error(y, yhat)
#        p_corr, _ = stats.pearsonr(y, yhat)
#        s_corr, _ = stats.spearmanr(y, yhat)
#        t_corr, _ = stats.kendalltau(y, yhat)
#        print("test rmse", rmse) 
#        print("test pearsonr", p_corr) 
#        print("test spearmanr", s_corr) 
#        print("test kendallt", t_corr) 
#
#
#    exit()
#
    ds_train = pd_utils.make_dataset(data_df, input_criteria, judge_id_column, output_criteria)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    
    print("Pretraining")
    pretrain_loop(model, ds_train, optimizer)
    print("Finetuning")
    finetune_loop(model, ds_train, optimizer)
    
    print("Train Loss (FT)", model.loss(ds_train.X, ds_train.A, ds_train.Y, I=[-1]).detach().numpy()) 
    yhat = model.decode(ds_train.X, ds_train.A, I=[-1])[:, 0].detach().numpy()
    y = ds_train.Y[:,-1].detach().numpy()
    # rmse = metrics.root_mean_squared_error(y, yhat)
    p_corr, _ = stats.pearsonr(y, yhat)
    s_corr, _ = stats.spearmanr(y, yhat)
    t_corr, _ = stats.kendalltau(y, yhat)
    # print("Train rmse", rmse) 
    print("Train pearsonr", p_corr) 
    print("Train spearmanr", s_corr) 
    print("Train kendallt", t_corr) 

    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)
    judge_map_path.parent.mkdir(exist_ok=True, parents=True)
    with judge_map_path.open("w") as fh:
        print(json.dumps(judge_id_map, indent="  "), file=fh)


if __name__ == "__main__":
    typer.run(main)
