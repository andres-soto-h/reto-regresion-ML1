"""
In this module we store prepare the sataset for machine learning experiments.
"""

import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    df = clean_dataset(df)
    y = df["y"]
    X = df.drop(columns=["y"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=X["age_range"]
    )
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    #split_mapping = {"train": (X, y), "test": (X, y)}
    return {k: split_mapping[k] for k in splits}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
            _fix_unhandled_nulls,
            _add_age_range,
            _add_bmi_range
        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper


def _fix_unhandled_nulls(df):
    df.dropna(inplace=True)
    return df

def transform_age(age):
    if age <=32:
        return "[18 to 32]"
    elif age<=48:
        return "[33 to 48]"
    else:
        return "[older than 48]"

#https://www.cdc.gov/obesity/adult/defining.html
def transform_bmi(bmi):
    if bmi <=18.5:
        return "Underweight"
    elif bmi<=24.9:
        return "Healthy weight"
    elif bmi<=29.9:
        return "Overweight"
    else:
        return "Obesity"

def _add_age_range(df):

    df['age_range'] = df['age'].apply(transform_age)
    df.drop(columns=["age"], axis=1)

    return df

def _add_bmi_range(df):

    df['bmi_range'] = df['bmi'].apply(transform_bmi)
    df.drop(columns=["bmi"], axis=1)
    return df


def get_categorical_column_names() -> t.List[str]:
    return (
        "sex,region"
    ).split(",")


def get_binary_column_names() -> t.List[str]:
    return "smoker".split(",")


def get_numeric_column_names() -> t.List[str]:
    return (
        "age,bmi,children"
    ).split(",")


def get_column_names() -> t.List[str]:
    return (
        "age,sex,bmi,children,smoker,region"
    ).split(",")


def get_categorical_variables_values_mapping() -> t.Dict[str, t.Sequence[str]]:
    return {
        "sex": ("male","female"),
        "smoker": ("yes","no"),
        "region": ("northeast","southeast","northwest","southwest"),
    }
