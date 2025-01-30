from typing_extensions import Annotated
from pathlib import Path
from functools import reduce

import typer
import numpy as np
import pandas as pd
from sklearn import metrics 
import scipy.stats


def load_predictions(paths: list[Path], pred_columns: list[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(path, sep="\t") for path in paths]
    columns = ["text_id", "annotator_id", "dialogue_system", "criterion"]
    df = reduce(lambda x, y: x.merge(y, on=columns, how="left"), dfs)

    for path, df_i in zip(paths, dfs):
        if len(df_i) != len(df):
            raise Exception("Missing data in predictions files.")

    return df[columns + pred_columns]

def main(
    prediction_paths: list[Path], 
    #predictions_path: Path = typer.Option(default=...),
    human_path: Annotated[Path, typer.Option("--human-judgments")],
    systems: str = typer.Option(default=...),
    #output_path: Path,
    criteria: str = "Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q0",
):
    systems = systems.split(",")
    criteria = criteria.split(",")
    pred_df = load_predictions(prediction_paths, systems)
    print(pred_df)
    reference_df = pd.read_csv(human_path, sep="\t")
    print(reference_df)

    results = []
    for criterion in criteria:
        for system in systems:
            refq_df = reference_df[reference_df[criterion] > 0]
            predq_df = pred_df[pred_df["criterion"] == criterion]

            if len(refq_df) != len(predq_df):
                raise Exception("Incompatible number of rows between reference and prediction.")

            columns = ["text_id", "annotator_id", "dialogue_system"]
            refq_predq_df = pd.merge(refq_df, predq_df, on=columns, how="left")
            if len(refq_df) != len(refq_predq_df):
                raise Exception("Incompatible number of rows between reference and prediction.")
            y = refq_predq_df[criterion].values
            pred_y = refq_predq_df[system].values
            rmse = metrics.root_mean_squared_error(y, pred_y)
            num_examples = len(predq_df)
            pearsonr = scipy.stats.pearsonr(y, pred_y).statistic
            spearmanr = scipy.stats.spearmanr(y, pred_y).statistic
            kendallt = scipy.stats.kendalltau(y, pred_y).statistic
            results.append([
                criterion, system, rmse, pearsonr, spearmanr, kendallt, num_examples, np.mean(pred_y), np.std(pred_y)
            ])

    results_df = pd.DataFrame(results, columns=["criterion", "system", "rmse", "pearson", "spearman", "kendall", "N", "mean", "std"])
    results_df = results_df.set_index(["criterion", "system"])
    print(results_df)
    

if __name__ == "__main__":
    typer.run(main)
