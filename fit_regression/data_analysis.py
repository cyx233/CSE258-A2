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
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE


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
    allUser = set()
    allItem = set()
    for r in allReview:
        allUser.add(r["user_id"])
        allItem.add(r["item_id"])
    print(
        f"All Users: {len(allUser)}, All Item: {len(allItem)}, All Review: {len(allReview)}"
    )

    allWeight = []
    allHeight = []
    allSize = []

    weight_pattern = re.compile(r"(\d+)lbs")
    height_pattern = re.compile(r"(\d+)' (\d+)\"")

    for r in allReview:
        if "height" in r and "weight" in r and "size" in r:
            weight = int(weight_pattern.search(r["weight"]).group(1))
            height = int(height_pattern.search(r["height"]).group(1)) * 12 + int(
                height_pattern.search(r["height"]).group(2)
            )
            weight *= 0.4535923
            height *= 0.0254
            allWeight.append(weight)
            allHeight.append(height)
            allSize.append(r["size"])
    plt.hist(allWeight)
    plt.xlabel("weight (kg)")
    plt.savefig("weight_hist.png")
    plt.close()

    plt.hist(allHeight)
    plt.xlabel("height (meter)")
    plt.savefig("height_hist.png")
    plt.close()

    plt.hist(allSize)
    plt.savefig("size_hist.png")
    plt.close()

    sample_fit = []
    sample_large = []
    sample_small = []
    for r in allReview:
        if "height" in r and "weight" in r and "size" in r:
            weight = int(weight_pattern.search(r["weight"]).group(1))
            height = int(height_pattern.search(r["height"]).group(1)) * 12 + int(
                height_pattern.search(r["height"]).group(2)
            )
            weight *= 0.4535923
            height *= 0.0254
            if r["fit"] == "fit":
                sample_fit.append((weight, height, r["size"]))
            elif r['fit'] == 'small':
                sample_small.append((weight, height, r["size"]))
            else:
                sample_large.append((weight, height, r["size"]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    X, Y, Z = zip(*sample_fit)
    ax.scatter(X, Y, Z, s=1, c="g", label="fit")
    X, Y, Z = zip(*sample_small)
    ax.scatter(X, Y, Z, s=1, c="r", label="small")
    X, Y, Z = zip(*sample_large)
    ax.scatter(X, Y, Z, s=1, c="b", label="large")

    plt.legend()
    ax.set_xlabel("weight (kg)")
    ax.set_ylabel("height (meter)")
    ax.set_zlabel("size")

    plt.savefig("fit_distribute.png")
    plt.close()

    X, Y, Z = zip(*sample_fit)
    BMI = np.array(X) / (np.array(Y) ** 2)
    plt.scatter(BMI, Z, s=1, c="g", label="fit")
    X, Y, Z = zip(*sample_large)
    BMI = np.array(X) / (np.array(Y) ** 2)
    plt.scatter(BMI, Z, s=1, c="r", label="large")
    X, Y, Z = zip(*sample_small)
    BMI = np.array(X) / (np.array(Y) ** 2)
    plt.scatter(BMI, Z, s=1, c="b", label="small")

    plt.legend()
    plt.xlabel("BMI")
    plt.ylabel("size")

    plt.savefig("fit_distribute_BMI.png")
    plt.close()

    samples_fit = []
    samples_rating = []
    category_index = {}
    for r in allReview:
        if r['category'] not in category_index:
            category_index[r['category']] = len(category_index)
        index = category_index[r['category']]
        if r["fit"] == "fit":
            samples_fit.append((index, r["size"], 0))
        elif r['fit'] == 'small':
            samples_fit.append((index, r["size"], -1))
        else:
            samples_fit.append((index, r["size"], 1))
        if r['rating']:
            samples_rating.append((index, r["size"], int(r['rating'])))

    fig = plt.figure(figsize=(10,5))

    ax = fig.add_subplot(121, projection="3d")
    X, Y, Z = zip(*samples_rating)
    ax.scatter(X, Y, Z, s=1)

    ax.set_xlabel("Category")
    ax.set_ylabel("Size")
    ax.set_title("Rating")

    ax = fig.add_subplot(122, projection="3d")
    X, Y, Z = zip(*samples_fit)
    ax.scatter(X, Y, Z, s=1)

    ax.set_xlabel("Category")
    ax.set_ylabel("Size")
    ax.set_zticks([-1, 0, 1])
    ax.set_zticklabels(['small', 'fit', 'large'])
    ax.set_title("Fit")

    fig.subplots_adjust(left=0, right=0.95, top=1, bottom=0)
    plt.savefig("item_features.png")
    plt.close()

if __name__ == "__main__":
    main()
