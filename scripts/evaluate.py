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
    exit()


    predictions_df = pd.read_csv(predictions_path, sep="\t")
    criteria = criteria.split(",")
    
    results = []
    for criterion in criteria:
        sub_df = predictions_df[predictions_df[criterion_column] == criterion]
        refs = sub_df[reference_column].values
        N = len(sub_df)
        for system in systems:
            predictions = sub_df[system].values
            rmse = metrics.root_mean_squared_error(refs, predictions)
            pearsonr = scipy.stats.pearsonr(refs, predictions).statistic
            spearmanr = scipy.stats.spearmanr(refs, predictions).statistic
            kendallt = scipy.stats.kendalltau(refs, predictions).statistic

            results.append({
                "criterion": criterion,
                "system": system,
                "rmse": rmse,
                "pearsonr": pearsonr,
                "spearmanr": spearmanr,
                "kendallt": kendallt,
      
                "N": N,
            })
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index(["criterion", "system"])
    print(results_df)
        







    return



    np.random.seed(random_seed)
    predictions_df = pd.read_csv(predictions_path, sep="\t")
    human_df = pd.read_csv(human_path, sep="\t")
    print(len(human_df))
    human_df = human_df[human_df["11"] != 0]
    print(len(human_df))
    print(human_df["dialogue_id"].unique().tolist())
    print(human_df[human_df["dialogue_id"] == "1"])
    exit()
    
    for criterion in [11]:
        df_pred_criterion = predictions_df[predictions_df["criterion"] == criterion]
        df_pred_criterion = human_df.merge(df_pred_criterion, on="dialogue_id", how="left")
        df_pred_criterion = df_pred_criterion.dropna()
        print(df_pred_criterion)

        print("sample_llm", metrics.mean_squared_error(df_pred_criterion[str(criterion)], df_pred_criterion["answer"]))
        print("expected_llm", metrics.mean_squared_error(df_pred_criterion[str(criterion)], df_pred_criterion["expected_llm"]))
        print("argmax_llm", metrics.mean_squared_error(df_pred_criterion[str(criterion)], df_pred_criterion["argmax_llm"]))
        print("random", metrics.mean_squared_error(df_pred_criterion[str(criterion)], df_pred_criterion["random"]))





    exit()








    answer_prob_columns = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]
    scores = np.array([[1.0, 2.0, 3.0, 4.0]]).T
    machine_df["expected_llm"] = machine_df[answer_prob_columns].values @ scores
    machine_df["argmax_llm"] = machine_df[answer_prob_columns].values.argmax(axis=1) + 1
    machine_df["random"] = np.random.choice([1, 2, 3, 4], size=[len(machine_df), 1])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    machine_df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
