import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LinearRegression


class HappinessTax:
    def __init__(self):
        self.__csv_money = pd.read_csv("data_frames/money.csv")
        self.__csv_happiness = pd.read_csv("data_frames/happiness.csv")
        self.__dff_money_and_happiness = ''
        self.linear_regression()

    def merge_happiness_tax(self) -> DataFrame:
        """
        :return: merge happiness to value -> boolean list
        """
        money_data_frame = self.__csv_money
        happiness_data_frame = self.__csv_happiness
        money_data_frame.rename(columns={'Country': 'country'}, inplace=True)
        happiness_data_frame.rename(columns={'fCountry': 'country'}, inplace=True)

        final_merge = pd.merge(money_data_frame, happiness_data_frame, on='country', how='inner')

        return final_merge

    def getter_happiness_tax(self) -> DataFrame:
        return self.__csv_happiness

    def getter_money_tax(self) -> DataFrame:
        return self.__csv_money

    def __str__(self):
        return self.merge_happiness_tax()

    def linear_regression(self) -> LinearRegression:
        """
        :return: Linear regression model and RMSE
        """
        dff_money_and_happiness = self.merge_happiness_tax()
        self.__dff_money_and_happiness = dff_money_and_happiness
        dff_money_and_happiness['GDP'] = dff_money_and_happiness['GDP'].apply(float)

        X = np.array(dff_money_and_happiness['GDP']).reshape(-1, 1)
        y = dff_money_and_happiness['Happiness']

        linear_regression_model = LinearRegression()
        linear_regression_model.fit(X, y)  # Finish IA

        prediction_y = linear_regression_model.predict(X)
        plt.scatter(X, y, color='blue')
        plt.plot(X, prediction_y, color='red')
        plt.show()

        dff_money_and_happiness['Residual'] = y - prediction_y
        ordered_df = dff_money_and_happiness.sort_values(by='Residual')

        print("3 countries worst happiness:")
        print(ordered_df[['country', 'GDP', 'Happiness', 'Residual']].head(3))

        print("\n3 countries best happiness:")
        print(ordered_df[['country', 'GDP', 'Happiness', 'Residual']].tail(3))

        return linear_regression_model
