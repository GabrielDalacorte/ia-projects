import pandas as pd
from pandas import DataFrame


def is_happiness_tax(csv_money, csv_happiness) -> DataFrame:
    """
    :param csv_money:
    :param csv_happiness:
    :return: merge happiness to value -> boolean list
    """
    money_data_frame = pd.read_csv(csv_money)
    happiness_data_frame = pd.read_csv(csv_happiness)

    money_data_frame.rename(columns={'Country': 'country'}, inplace=True)
    happiness_data_frame.rename(columns={'fCountry': 'country'}, inplace=True)

    final_merge = pd.merge(money_data_frame, happiness_data_frame, on='country', how='inner')

    return final_merge.head(10)  # return 10 values


def linear_regression(pib):
    # in progress
    pass