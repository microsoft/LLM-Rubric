from pathlib import Path
from typing_extensions import Annotated
import math

import typer
import numpy as np
import pandas as pd


def main(
    train_human_path: Annotated[Path, typer.Option("--train-human-judgments")],
    human_path: Annotated[Path, typer.Option("--human-judgments")],
    output_path: Path = typer.Option(default=...),
    criteria: str = "Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q0",
) -> None:
    train_human_df = pd.read_csv(train_human_path, sep="\t")
    criteria = criteria.split(",")
    
    constants = {}
    ann_constants = {}
    for criterion in criteria:
        vals = [val for val in train_human_df[criterion].tolist() if val != 0 and val == val]
        constant = np.mean(vals)
        constants[criterion] = constant
        for ann in train_human_df["annotator_id"].unique().tolist():
            vals = [val for val in train_human_df[train_human_df["annotator_id"] == ann][criterion].tolist() if val != 0 and val == val]            
            ann_constant = np.mean(vals)
            ann_constants[(ann, criterion)] = ann_constant

    print(pd.DataFrame([constants], index=["mean"]).T)
    print(pd.DataFrame([ann_constants], index=["mean"]).T)

    human_df = pd.read_csv(human_path, sep="\t")

    data = []
    for _, row in human_df.iterrows():
        for criterion in criteria:
            if not row[criterion] in set([1.0, 2.0, 3.0, 4.0]):
                continue
            if math.isnan(row[criterion]):
                continue
            data.append([
                row["text_id"],
                row["annotator_id"],
                row["dialogue_system"],
                criterion,
                constants[criterion],
                ann_constants[(row["annotator_id"], criterion)],
            ])
    data_df = pd.DataFrame(data, columns=["text_id", "annotator_id", "dialogue_system", "criterion", "group_constant", "ann_constant"])
    print(data_df)
  
    output_path.parent.mkdir(exist_ok=True, parents=True)
    data_df.to_csv(output_path, sep="\t", index=False)
    exit()






    machine_df = pd.read_csv(machine_path, sep="\t")
    llm_answer_prob_columns = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]
    scores = np.array([[1.0, 2.0, 3.0, 4.0]]).T
    machine_df["expected_llm"] = machine_df[llm_answer_prob_columns].values @ scores
    machine_df["argmax_llm"] = machine_df[llm_answer_prob_columns].values.argmax(axis=1) + 1
    machine_df["random"] = np.random.choice([1, 2, 3, 4], size=[len(machine_df), 1])

    print(machine_df)

    data_df = machine_df.merge(human_df, on="text_id", how="left")
    data_df = data_df[data_df["criterion"].isin(criteria)]
    print(data_df)
    data_df["constant"] = data_df["criterion"].apply(lambda x: constants[x])
    print(data_df)

    def get_human_ref(row: pd.Series) -> int:
        c = row["criterion"]
        return row[c]
    data_df["human_judgment"] = data_df.apply(get_human_ref, axis=1)

    print(data_df)
    exit()



    example_id_cols = ["text_id", "annotator_id", "criterion", "human_judgment"] 
    system_cols = ["constant"]
    human_judge_cols = list(criteria)
    print(human_judge_cols)
    sorted_cols = example_id_cols + system_cols + llm_answer_prob_columns + human_judge_cols
    misc_columns = sorted([column for column in data_df.columns if column not in sorted_cols])

    print(
        data_df[sorted_cols + misc_columns]
    )
    from sklearn import metrics
    for c, group in data_df[sorted_cols + misc_columns].groupby("criterion"):
        print(c)
        yhat = group["constant"].values
        y = group["human_judgment"]
        print(metrics.root_mean_squared_error(y, yhat))

    exit()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    data_df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
