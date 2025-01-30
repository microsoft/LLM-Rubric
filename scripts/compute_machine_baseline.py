from pathlib import Path
from typing_extensions import Annotated
import math

import typer
import numpy as np
import pandas as pd


def main(
    machine_path: Annotated[Path, typer.Option("--machine-evaluations")],
    human_path: Annotated[Path, typer.Option("--human-judgments")],
    output_path: Path = typer.Option(default=...),
    random_seed: int = 4076609,
    criteria: str = "Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q0",
):
    np.random.seed(random_seed)
    criteria = criteria.split(",")
    machine_df = pd.read_csv(machine_path, sep="\t")
    machine_df = machine_df[machine_df["criterion"].isin(criteria)]
    human_df = pd.read_csv(human_path, sep="\t")
    llm_answer_prob_columns = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]
    
    for idx, row in machine_df.iterrows():
        if row[llm_answer_prob_columns].sum() < 1.0:
            s = row[llm_answer_prob_columns].sum()
            for c in llm_answer_prob_columns:
                machine_df.at[idx, c] /= s

    scores = np.array([[1.0, 2.0, 3.0, 4.0]]).T
    machine_df["expected_llm"] = machine_df[llm_answer_prob_columns].values @ scores
    machine_df["argmax_llm"] = machine_df[llm_answer_prob_columns].values.argmax(axis=1) + 1
    machine_df["sample_llm"] = [np.random.choice([1.0, 2.0, 3.0, 4.0], p=row) for row in machine_df[llm_answer_prob_columns].values]

    data = []
    for _, row in human_df.iterrows():
        for criterion in criteria:
            if not row[criterion] in set([1.0, 2.0, 3.0, 4.0]):
                continue
            if math.isnan(row[criterion]):
                continue
            mpred_df = machine_df.loc[(machine_df["text_id"] == row["text_id"]) & (machine_df["criterion"] == criterion)]
            assert len(mpred_df) == 1
            data.append([
                row["text_id"],
                row["annotator_id"],
                row["dialogue_system"],
                criterion,
                float(mpred_df["sample_llm"].values[0]),
                float(mpred_df["argmax_llm"].values[0]),
                float(mpred_df["expected_llm"].values[0]),
                #ann_constants[(row["annotator_id"], criterion)],
            ])
    data_df = pd.DataFrame(data, columns=["text_id", "annotator_id", "dialogue_system", "criterion", "sample_llm", "argmax_llm", "expected_llm"])
    print(data_df)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    data_df.to_csv(output_path, sep="\t", index=False)

if __name__ == "__main__":
    typer.run(main)
