import re
import json
import random
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


from tqdm import tqdm


random.seed(42)


def readJson():
    """
    {
        "fit": "fit",
        "user_id": "123612",
        "bust size": "36b",
        "item_id": "127865",
        "weight": "155lbs",
        "rating": "10",
        "rented for": "wedding",
        "review_text": "This dress was wonderful! I had originally planned to wear the BCBG MAXAZRIA Inevitable Beauty Gown, but the fit was not amazing (if you have meat on your thighs, steer clear of it). The dress I wore was a backup style and I was so happy I had it. It was so comfortable and flattering. It is also pretty forgiving. I received compliments all night from other quests and strangers on the street. I highly recommend choosing this dress if you have a black tie or black tie optional wedding to attend. ",
        "body type": "athletic",
        "review_summary": "I wore this to a beautiful black tie optional wedding in Boston. ",
        "category": "gown",
        "height": "5' 6\"",
        "size": 16,
        "age": "30",
        "review_date": "August 29, 2017",
    }
    """
    allReview = []
    with open("renttherunway_final_data.json", "rt") as f:
        for l in tqdm(f.readlines()):
            d = json.loads(l)
            if "height" in d and "weight" in d and "size" in d:
                allReview.append(json.loads(l))
    return allReview


def main():
    allReview = readJson()
    allUser = {}
    allItem = {}

    weight_pattern = re.compile(r"(\d+)lbs")
    height_pattern = re.compile(r"(\d+)' (\d+)\"")

    fit_dict = {
        "small": -1,
        "fit": 0,
        "large": 1,
    }

    for r in allReview:
        if r["user_id"] not in allUser:
            allUser[r["user_id"]] = len(allUser)
        if r["item_id"] not in allItem:
            allUser[r["item_id"]] = len(allItem)

    X = [
        (
            int(weight_pattern.search(r["weight"]).group(1)) * 0.4535923,
            (
                int(height_pattern.search(r["height"]).group(1)) * 12
                + int(height_pattern.search(r["height"]).group(2))
            )
            * 0.0254,
            r['size']
        )
        for r in allReview
    ]
    Y = [fit_dict[r["fit"]] for r in allReview]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42
    )
    clf = LogisticRegression(multi_class="auto", class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"(height, weight, size) f1: {metrics.f1_score(y_test, y_pred, average=None)}")

    height, weight, size = zip(*X_train)
    BMI_train = np.stack([np.array(height) / (np.array(weight) ** 2), np.array(size)]).transpose()
    print(BMI_train)

    height, weight, size = zip(*X_test)
    BMI_test = np.stack([np.array(height) / (np.array(weight) ** 2), np.array(size)]).transpose()

    clf.fit(BMI_train, y_train)
    y_pred = clf.predict(BMI_test)
    print(f"(BMI, size) f1: {metrics.f1_score(y_test, y_pred, average=None)}")


if __name__ == "__main__":
    main()
