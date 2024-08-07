import json
import hashlib
from pathlib import Path
import random
from typing_extensions import Annotated
import multiprocessing as mp
import dataclasses 
from itertools import product

import typer
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from scipy import stats 
from pydantic import Field
from pydantic.dataclasses import dataclass
import tqdm

from llm_rubric import kfold_utils, pd_utils
from llm_rubric.model.torch_impl import (
    PersonalizedCalibrationNetwork, Hyperparameters, pretrain_loop, finetune_loop
)


@dataclass
class HyperparameterGrid:
    input_size: int
    output_size: int
    num_judges: int
    all_data_size: int
    batch_sizes: list[int] = Field(default_factory=lambda: [32, 64, 128])
    learning_rates: list[float] = Field(default_factory=lambda: [0.01, 0.005, 0.001, 0.0005, 0.0001])
    layer1_sizes: list[int] = Field(default_factory=lambda: [100, 50, 25])
    layer2_sizes: list[int] = Field(default_factory=lambda: [100, 50, 25])
    pretraining_epochs: list[int] = Field(default_factory=lambda: [5, 10, 15, 20, 30, 40, 50])
    finetuning_epochs: list[int] = Field(default_factory=lambda: [5, 10, 15, 20, 30, 40, 50])

    def enumerate(self, offset: int = 0):
        for i, hp in enumerate(
            product(
                self.batch_sizes,
                self.learning_rates,
                self.layer1_sizes,
                self.layer2_sizes,
                self.pretraining_epochs,
                self.finetuning_epochs,
            ),
            offset,
        ):
            (
                batch_size,
                learning_rate,
                layer1_size,
                layer2_size,
                pretraining_epochs,
                finetuning_epochs,
            ) = hp
            yield i, Hyperparameters(
                input_size=self.input_size,
                output_size=self.output_size,
                num_judges=self.num_judges,
                all_data_size=self.all_data_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                layer1_size=layer1_size,
                layer2_size=layer2_size,
                pretraining_epochs=pretraining_epochs,
                finetuning_epochs=finetuning_epochs,
                
            )

    def __len__(self) -> int:
        return (
            len(self.batch_sizes)
            * len(self.learning_rates)
            * len(self.layer1_sizes)
            * len(self.layer2_sizes)
            * len(self.pretraining_epochs)
            * len(self.finetuning_epochs)
        ) 



def run_job(args):
    fold, ds_train, ds_test, hp, path = args
    
    if path.exists():
        return 

    model = PersonalizedCalibrationNetwork(hp)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    pretrain_loop(model, ds_train, optimizer)
    finetune_loop(model, ds_train, optimizer)

    loss = model.loss(ds_test.X, ds_test.A, ds_test.Y, I=[-1])
    Yhat = model.decode(ds_test.X, ds_test.A, I=[-1])
    
    yhat = Yhat[:,0].detach().numpy()
    y = ds_test.Y[:,-1].detach().numpy()
    rmse = metrics.root_mean_squared_error(y, yhat)
    p_corr, _ = stats.pearsonr(y, yhat)
    s_corr, _ = stats.spearmanr(y, yhat)
    t_corr, _ = stats.kendalltau(y, yhat)

    result = {
        "fold": fold,
        "hyperparameters": json.dumps(dataclasses.asdict(hp)),
        "pearson": p_corr,
        "spearman": s_corr,
        "kendall": t_corr,
        "rmse": rmse,
        "log_loss_finetune": float(loss),
        "finetune_target": "Q0",
        # TODO make this configurable
    }

    with path.open("w") as fh:
        print(json.dumps(result, indent="  "), file=fh)


def main(
    eval_text_ids_path: Annotated[Path, typer.Option("--ids")],
    human_path: Annotated[Path, typer.Option("--human-judgments")],
    machine_path: Annotated[Path, typer.Option("--machine-evaluations")],
    output_path: Annotated[Path, typer.Option(default=...)],
    num_folds: int = 5,
    text_id_column: str = "text_id",
    criterion_column: str = "criterion",
    fold_column: str = "fold",
    input_criteria: list[str] = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q0"],
    output_criteria: list[str] = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q0"],
    num_answers: int = 4,
    judge_column: str = "annotator_name", 
    judge_id_column: str = "judge_id", 
    random_seed: int = 43,
    num_procs: int = 4,
): 

    text_ids = pd.read_csv(eval_text_ids_path, sep="\t")[text_id_column].tolist()
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
    print(f"Running {num_folds}-fold cross validation on the following ids: {', '.join(text_ids)}")
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    random.shuffle(text_ids)
    fold_map = kfold_utils.make_fold_map(text_ids)

    data_df = human_df.merge(machine_df, on=text_id_column, how="left")
    data_df[fold_column] = data_df[text_id_column].apply(fold_map.get)
    data_df = data_df.dropna()
    print(f"Total rows: {len(data_df)}")

    judge_id_map = pd_utils.add_judge_ids(data_df, judge_column, judge_id_column)
    num_judges = len(judge_id_map)

    kfold_iter = kfold_utils.train_test_dataframe_iter(data_df, fold_column=fold_column, num_folds=num_folds)

    input_size = len(input_criteria) * num_answers # TODO allow for different number of outputs per rubric item.
    output_size = len(output_criteria)
    hp_grid = HyperparameterGrid(
        input_size=input_size,
        output_size=output_size,
        num_judges=num_judges,
        all_data_size=len(data_df),
    )

    output_path.mkdir(exist_ok=True, parents=True)
    def make_job_iter():
        for fold, (train_df, test_df) in enumerate(kfold_iter):
            ds_train = pd_utils.make_dataset(train_df, input_criteria, judge_id_column, output_criteria)
            ds_test = pd_utils.make_dataset(test_df, input_criteria, judge_id_column, output_criteria)

            for i, hp in hp_grid.enumerate():
                data_md5 = hashlib.md5(json.dumps((fold, dataclasses.asdict(hp)), sort_keys=True).encode("utf-8")).hexdigest()

                path = output_path / f"{data_md5}.json"

                yield fold, ds_train, ds_test, hp, path

    job_iter = make_job_iter()

    pool = mp.Pool(num_procs)

    num_jobs = num_folds * len(hp_grid)
    for result in tqdm.tqdm(pool.imap(run_job, list(job_iter)), total=num_jobs):
        pass


if __name__ == "__main__":
    typer.run(main)
