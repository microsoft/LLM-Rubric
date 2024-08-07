import pandas as pd


def train_test_dataframe_iter(
    df: pd.DataFrame,
    num_folds: int = 5,
    fold_column: str = "fold",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold_column not in df.columns:
        df_cols = ", ".join([f"'{column}'" for column in df.columns])
        raise Exception(
            f"Fold column '{fold_column}' not in dataframe columns: {df_cols}"
        )
    folds = list(range(0, num_folds))
    if not df[fold_column].isin(folds).all():
        raise Exception(
            f"Found uexpected number of fold ids in column: '{fold_column}'"
        )

    for fold_num in range(num_folds):
        test_split_df = df[df[fold_column] == fold_num].copy()
        train_split_df = df[df[fold_column] != fold_num].copy()
        yield train_split_df, test_split_df


def make_fold_map(ids: list[str], k: int = 5):
    fold_map = {}
    for i, id_ in enumerate(ids):
        fold_map[id_] = i % k
    return fold_map



