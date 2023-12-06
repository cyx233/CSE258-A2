from cProfile import label
from collections import Counter, defaultdict
import heapq
import re
import seaborn as sns
import json
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import train_test_split
from scipy import sparse


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
            allReview.append(json.loads(l))
    return allReview


def main():
    allReview = readJson()
    allUser = {}
    allItem = {}
    fit_dict = {
        "small": -1,
        "fit": 0,
        "large": 1,
    }

    for r in allReview:
        if r["user_id"] not in allUser:
            allUser[r["user_id"]] = len(allUser)
        if r["item_id"] not in allItem:
            allItem[r["item_id"]] = len(allItem)

    X = [(r["user_id"], r["item_id"]) for r in allReview]
    Y = [fit_dict[r["fit"]] for r in allReview]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    rows = []
    cols = []
    values = [1] * 2 * len(X_train)
    for i, r in enumerate(X_train):
        rows += [i, i]
        cols += [allUser[r[0]], allItem[r[1]]]
    feat_train = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(X_train), len(allUser) + len(allItem))
    )

    clf = LogisticRegression(multi_class="auto", class_weight="balanced", max_iter=1000)
    clf.fit(feat_train, y_train)

    rows = []
    cols = []
    values = [1] * 2 * len(X_test)
    for i, r in enumerate(X_test):
        rows += [i, i]
        cols += [allUser[r[0]], allItem[r[1]]]
    feat_test = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(X_test), len(allUser) + len(allItem))
    )
    y_pred = clf.predict(feat_test)
    print(f"one_hot f1: {metrics.f1_score(y_test, y_pred, average=None)}")


if __name__ == "__main__":
    main()
