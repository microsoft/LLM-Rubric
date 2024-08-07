import numpy as np
import pandas as pd

from llm_rubric.model.torch_impl import Dataset


def convert_criteria_to_feature_columns(criteria: list[str], num_answers: int) -> list[str]:
    return [f"{c}_{i}_prob" for c in criteria for i in range(1, num_answers + 1)]


def join_rowwise_predictions(
    df: pd.DataFrame,
    text_id_column: str,
    criterion_column: str,
    criteria: list[str],
    num_answers: int,
):
    join_data = []
    prob_columns = [f"answer{i}_prob" for i in range(1, num_answers + 1)]
    for text_id, df_group in df.groupby(text_id_column):
        datum = [text_id] 
        for c in criteria:
            df_criteria = df_group[df_group[criterion_column] == c]
            probs = df_criteria[prob_columns].values[0].tolist()
            datum.extend(probs)
        join_data.append(datum)

    join_df = pd.DataFrame(
        join_data,
        columns=[text_id_column] + convert_criteria_to_feature_columns(criteria, num_answers)
    )

    return join_df


def add_judge_ids(
    df: pd.DataFrame,
    judge_column: str,
    judge_id_column: str,
    judge_id_map = None,
):
    df[judge_column] = df[judge_column].apply(lambda x: str(x))
    if judge_id_map is None:
        judge_id_map = {
            str(judge): judge_id 
            for judge_id, judge in enumerate(df[judge_column].unique())
        }
    df[judge_id_column] = df[judge_column].apply(
        lambda x: judge_id_map[x]
    )
    return judge_id_map


def make_dataset(
    df: pd.DataFrame,
    input_columns: list[str],
    annotator_column: str,
    output_columns: list[str],
) -> Dataset:

    # TODO move this feature naming logic out.
    input_features = [f"{c}_{i}_prob" for c in input_columns for i in range(1,5)]
    return Dataset(
        machine_evaluations=df[input_features].values.astype(np.float32),
        human_judges=df[annotator_column].values.astype(np.int64)[:, None],
        human_judgments=df[output_columns].values.astype(np.float32),
    )
