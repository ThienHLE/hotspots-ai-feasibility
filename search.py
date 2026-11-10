"""Part 1 starter: rank local spots using Ollama embeddings."""

import argparse
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def cosine_similarity(left, right):
    if not left or len(left) != len(right):
        raise ValueError("Embedding vectors must have matching, nonzero dimensions.")
    denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def embed(texts, model):
    payload = json.dumps({"model": model, "input": texts, "truncate": False}).encode("utf-8")
    request = Request("http://localhost:11434/api/embed", data=payload,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        vectors = json.load(response)["embeddings"]
    if len(vectors) != len(texts):
        raise ValueError("Ollama returned an unexpected number of embeddings.")
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Describe the kind of spot you want.")
    parser.add_argument("--model", default="all-minilm")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "spots.json")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("Query must not be empty.")

    try:
        spots = json.loads(args.data.read_text(encoding="utf-8"))
        if not isinstance(spots, list):
            raise ValueError("Dataset must be a JSON array of spots.")
        if not spots:
            print("Add spots to data/spots.json first. See README.md for the format.")
            return 0
        texts = []
        ids = set()
        for spot in spots:
            if not isinstance(spot, dict):
                raise ValueError("Each spot must be a JSON object.")
            for field in ("id", "name", "description"):
                if not isinstance(spot.get(field), str) or not spot[field].strip():
                    raise ValueError(f"Each spot needs a nonempty {field} string.")
            categories = spot.get("categories")
            if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories):
                raise ValueError("Each spot needs a categories array of strings.")
            if spot["id"] in ids:
                raise ValueError("Spot IDs must be unique.")
            ids.add(spot["id"])
            texts.append(f"{spot['name']}. {', '.join(categories)}. {spot['description']}")
        vectors = embed(texts + [args.query], args.model)
        ranked = sorted(zip(spots, vectors[:-1]),
                        key=lambda item: cosine_similarity(item[1], vectors[-1]), reverse=True)
        for rank, (spot, vector) in enumerate(ranked[:3], start=1):
            score = cosine_similarity(vector, vectors[-1])
            print(f"{rank}. {spot['name']} [{spot['id']}] | similarity: {score:.4f}")
            print(f"   {spot['description']}")
    except HTTPError as error:
        print(f"Ollama error {error.code}: {error.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError):
        print("Could not reach Ollama. Start it and run: ollama pull " + args.model, file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Search failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
