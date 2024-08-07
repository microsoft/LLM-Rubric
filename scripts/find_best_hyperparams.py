from pathlib import Path
from typing_extensions import Annotated
import json

import typer
import pandas as pd
from tqdm import tqdm


def main(
    hyperparam_dir: Annotated[Path, typer.Option("--hps")],
    output_path: Annotated[Path, typer.Option("--output")],
    num_folds: int = typer.Option(default=5),
    metric: str = typer.Option(default="log_loss_finetune"),
) -> None: 
    paths = list(hyperparam_dir.glob("*.json"))
    results = []
    for path in tqdm(paths, desc="Reading hyperparameters "):
        result = json.loads(path.read_text())
        results.append(result)
    df = pd.DataFrame(results)

    grouped_results = []
    for hp, df_hp in df.groupby("hyperparameters"):
        values = {"hp": hp}
        for fold, value in df_hp[["fold", metric]].values:
            values[fold] = value            
        grouped_results.append(values)

    grouped_df = pd.DataFrame(grouped_results).set_index("hp")
    grouped_df["mean"] = grouped_df.mean(axis=1)
    grouped_df.sort_values("mean", inplace=True)
    best_hp = grouped_df.index[0]
    print("Best hyperparameters:") 
    print(json.dumps(json.loads(best_hp), indent="  "))
    print(f"Metric mean {metric}: {grouped_df.at[best_hp, 'mean']}")
    print(df[df["hyperparameters"] == best_hp])
    with output_path.open("w") as fh:
        print(best_hp, file=fh)


if __name__ == "__main__":
    typer.run(main)
