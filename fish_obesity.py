import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt


class FishObesity:
    def __init__(self):
        self.fish_obesity_df = pd.read_csv('data_frames/fish obesity.csv')
        self.prepare()


    def prepare(self):
        """
        :return: merge happiness to value -> boolean list
        """

        fish_obesity_df = self.fish_obesity_df
        labels = fish_obesity_df['Obese']

        features = fish_obesity_df[['Weight', 'Height']]

        # Splitting the data into training and testing sets, in a ratio of 80% to 20%. Using train_test_split from
        # scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        classifier = SVC(random_state=42)
        classifier.fit(X_train.values, y_train)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_train['Weight'], X_train['Height'], c=y_train, cmap='viridis')

        # Plotting the contour line
        ax = plt.gca()
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        xx = np.linspace(x_lim[0], x_lim[1], 100)
        yy = np.linspace(y_lim[0], y_lim[1], 100)
        XX, YY = np.meshgrid(xx, yy)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = classifier.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        plt.xlabel('Weight')
        plt.ylabel('Height')
        plt.title('Fish Obesity Classification')
        plt.colorbar(label='Class')
        plt.show()

        is_obesity_new_fish = [[200, 163]]
        obesity_new_fish = classifier.predict(is_obesity_new_fish)
        print(f"Obesity: {True if obesity_new_fish[0] == 1 else False}")