import json
import pickle
import pathlib
import networkx as nx
from node2vec import Node2Vec

BASE = pathlib.Path("data/graphs")
BASE.mkdir(parents=True, exist_ok=True)

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def embed_graph(graph_path, dim=64):
    G = load_graph(graph_path)

    n2v = Node2Vec(
        G,
        dimensions=dim,
        walk_length=30,
        num_walks=200,
        workers=4,
        weight_key="weight"
    )

    model = n2v.fit(window=10, min_count=1)
    return {
        node: model.wv[node].tolist()
        for node in G.nodes
    }


if __name__ == "__main__":
    all_embeddings = {}

    for level in ["root", "mid", "full"]:
        print(f"Embedding {level} graph...")
        graph_path = BASE / f"graph_{level}.gpickle"
        all_embeddings[level] = embed_graph(graph_path)

    out_path = BASE / "graph_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(all_embeddings, f, indent=2)

    print(f"Saved embeddings to {out_path}")
