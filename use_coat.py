from sklearn.linear_model import LogisticRegression


def use_coat(degrees: list) -> str:
    """
    [IA SUPERVISED] Function that checks whether to wear a coat
    :param degrees:
    :return: coat used -> boolean list
    """
    temperature = [[30], [12], [14], [18], [25], [5], [15], [27]]  # features
    coat = [False, True, True, True, False, True, True, False]  # labels

    classifier = LogisticRegression()
    classifier.fit(temperature, coat)
    should_wear_coat = degrees
    use_coat_classification = classifier.predict(should_wear_coat)
    return f"Use coat: {use_coat_classification}"
