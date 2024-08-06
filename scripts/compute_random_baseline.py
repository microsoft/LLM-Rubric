from pathlib import Path
from typing_extensions import Annotated
import math

import typer
import numpy as np
import pandas as pd


def main(
    human_path: Annotated[Path, typer.Option("--human-judgments")],
    output_path: Path = typer.Option(default=...),
    random_seed: int = 4076609,
    criteria: str = "Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q0",
    values: str = "1,2,3,4"
) -> None:
    np.random.seed(random_seed)
    criteria = criteria.split(",")
    values = [float(x) for x in values.split(",")]
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
                np.random.choice(values),
            ])
    data_df = pd.DataFrame(data, columns=["text_id", "annotator_id", "dialogue_system", "criterion", "random"])
    print(data_df)
  
    output_path.parent.mkdir(exist_ok=True, parents=True)
    data_df.to_csv(output_path, sep="\t", index=False)

if __name__ == "__main__":
    typer.run(main)
