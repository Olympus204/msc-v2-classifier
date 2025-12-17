import json
import math
import pickle
from pathlib import Path
import networkx as nx

BASE = Path("data/graphs")

def load_cooc(path):
    with open(path) as f:
        raw = json.load(f)
    return {
        tuple(k.split("|")): v
        for k, v in raw.items()
    }


def build_graph(cooc, min_weight=5):
    G = nx.Graph()

    for (a, b), w in cooc.items():
        if w >= min_weight:
            G.add_edge(a, b, weight=math.log1p(w))

    return G


if __name__ == "__main__":
    for level in ["root", "mid", "full"]:
        cooc = load_cooc(BASE / f"cooc_{level}.json")
        G = build_graph(cooc)

        with open(BASE / f"graph_{level}.gpickle", "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
