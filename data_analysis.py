from collections import Counter, defaultdict
import heapq
import seaborn as sns
import json
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity


from tqdm import tqdm


random.seed(42)


def process_games():
    """
    item data sample
    {
        "publisher": "Kotoshiro",
        "genres": ["Action", "Casual", "Indie", "Simulation", "Strategy"],
        "app_name": "Lost Summoner Kitty",
        "title": "Lost Summoner Kitty",
        "url": "http://store.steampowered.com/app/761140/Lost_Summoner_Kitty/",
        "release_date": "2018-01-04",
        "tags": ["Strategy", "Action", "Indie", "Casual", "Simulation"],
        "discount_price": 4.49,
        "reviews_url": "http://steamcommunity.com/app/761140/reviews/?browsefilter=mostrecent&p=1",
        "specs": ["Single-player"],
        "price": 4.99,
        "early_access": False,
        "id": "761140",
        "developer": "Kotoshiro",
    }
    """
    allGame = {}
    with open("steam_games.json", "rt") as f:
        for l in tqdm(f.readlines()):
            d = eval(l)
            if "app_name" in d:
                if "id" in d and "genres" in d:
                    allGame[d["id"]] = {
                        "index": len(allGame),
                        "title": d["app_name"],
                        "genres": d["genres"],
                    }
    with open("processed_games.json", "w") as f:
        json.dump(allGame, f)


def process_reviews():
    """
    review data sample
    {
        "username": "₮ʜᴇ Wᴀʀᴛᴏɴ",
        "hours": 51.1,
        "products": 769,
        "product_id": "328100",
        "page_order": 0,
        "date": "2017-12-27",
        "text": "looks like a facebook game",
        "early_access": False,
        "page": 1,
    }
    """
    allReview = []
    with open("steam_reviews.json", "rt") as f:
        for l in tqdm(f.readlines()):
            d = eval(l)
            allReview.append((d["username"], d["product_id"], d["text"]))
    with open("processed_reviews.json", "w") as f:
        json.dump(allReview, f)


def main():
    with open("processed_reviews.json", "r") as f:
        allReviews = json.load(f)

    with open("processed_games.json", "r") as f:
        allGames = json.load(f)

    allReviews = list(filter(lambda x: x[1] in allGames, allReviews))
    game_to_review = defaultdict(list)
    for username, gid, text in allReviews:
        game_to_review[gid].append((username, text))

    allGames = {
        k: v for k, v in filter(lambda x: x[0] in game_to_review, allGames.items())
    }
    print(f"all reviews: {len(allReviews)}, all games: {len(allGames)}")

    genre_to_games = defaultdict(set)
    for id in allGames:
        for t in allGames[id]["genres"]:
            genre_to_games[t].add(id)

    genres_cnt = Counter({t: len(genre_to_games[t]) for t in genre_to_games})
    print(f"unique genres: {len(genres_cnt)}")
    X, Y = zip(*genres_cnt.most_common(30))
    plt.bar(X, Y)
    plt.subplots_adjust(bottom=0.4)
    plt.xticks(rotation=90)
    plt.title(f"Top 30 Genres for {len(allGames)} Games")
    plt.savefig("genres_cnt.png")
    plt.close()

    X = ["Action Only", "Casual Only", "Both", "None"]
    both = genre_to_games["Action"] & genre_to_games["Casual"]
    none = set(allGames.keys()) - genre_to_games["Action"] - genre_to_games["Casual"]
    action_only = genre_to_games["Action"] - genre_to_games["Casual"]
    casual_only = genre_to_games["Casual"] - genre_to_games["Action"]
    Y = [len(action_only), len(casual_only), len(both), len(none)]
    plt.bar(X, Y)
    plt.title(f"Action/Casual Games Count")
    plt.savefig("action_casual.png")
    plt.close()

    sample_K = []
    sample_games = []
    for data in [action_only, casual_only, both, none]:
        good_data = list(filter(lambda x: len(game_to_review[x]) >= 10, data))
        sample_games += good_data[:500]
        sample_K.append(len(good_data[:500]))

    gid_index = {}
    game_players = {}
    for gid in sample_games:
        if gid not in gid_index:
            gid_index[gid] = len(gid_index)
        game_players[gid] = set([i[0] for i in game_to_review[gid]])

    def jaccard_simularity(s1, s2):
        num = len(s1 & s2)
        den = len(s1 | s2)
        return num / den if den > 0 else 0

    def kNN(k):
        k_neighbor = defaultdict(set)
        for g1 in tqdm(sample_games):
            neighbors = []
            for g2 in sample_games:
                similarity = jaccard_simularity(game_players[g1], game_players[g2])
                if similarity > 0:
                    if len(neighbors) < k:
                        heapq.heappush(neighbors, (similarity, g2))
                    else:
                        heapq.heappushpop(neighbors, (similarity, g2))
            k_neighbor[g1] = set([i[1] for i in neighbors])
        return k_neighbor

    def pagerank(focus_genre):
        X = np.zeros(len(sample_games))
        games_k_neighbor = kNN(40)
        for gid in sample_games:
            for neighbor in games_k_neighbor[gid]:
                if neighbor in genre_to_games[focus_genre]:
                    X[gid_index[gid]] += 1 / len(game_players[neighbor])
        return X

    Action_X = pagerank("Action")
    Casual_X = pagerank("Casual")

    colors = ["red", "blue", "green", "black"]
    labels = ["Action Only", "Casual Only", "Both", "None"]

    start = 0
    end = 0
    for i in range(len(labels)):
        start = end
        end += sample_K[i]
        plt.scatter(
            Action_X[start:end],
            Casual_X[start:end],
            c=colors[i],
            label=labels[i],
            s=1,
        )
    
    plt.xscale("log")
    plt.yscale("log")

    plt.legend()
    plt.xlabel("Action Score")
    plt.ylabel("Casual Score")

    plt.title("Action/Casual Games Visualization")
    plt.savefig("visualization.png")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_thetamin(0.0)
    ax.set_thetamax(180.0)
    ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow("True")

    r = np.sqrt(Action_X**2 + Casual_X**2)
    theta = np.arctan2(Casual_X, Action_X) * 2

    start = 0
    end = 0
    for i in range(len(labels)):
        start = end
        end += sample_K[i]
        plt.scatter(
            theta[start:end],
            r[start:end],
            c=colors[i],
            label=labels[i],
            s=1,
        )

    plt.yscale("log")

    plt.legend()

    plt.title("Action/Casual Games Visualization")
    plt.savefig("visualization_angle.png")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists("processed_games.json"):
        process_games()
    if not os.path.exists("processed_reviews.json"):
        process_reviews()
    main()
