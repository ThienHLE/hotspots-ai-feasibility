# HotSpots AI Feasibility

Small experiments to check whether semantic search and a local LLM can help users find suitable spots for HotSpots. The initial setup covers Part 1: semantic search. Part 2 will test an LLM separately with a fixed set of spots.

## Part 1 setup

Requires Python 3.10+ and [Ollama](https://ollama.com/download). No Python packages are needed.

1. Clone this repo and open a terminal in its folder.
2. Start Ollama (open the app, or run `ollama serve` if it is not already running).
3. Download the starter embedding model:

   ```sh
   ollama pull all-minilm
   ```

4. Add spots to `data/spots.json`, using this format:

   ```json
   [
     {
       "id": "spot-001",
       "name": "Example spot",
       "categories": ["outdoors"],
       "description": "Replace this example with an accurate description of a real Fort Smith spot."
     }
   ]
   ```

5. Run a search:

   ```sh
   python search.py "a quiet place to study"
   ```

The script embeds each spot's name, categories, and description and the user's query through Ollama, then prints the top three spots ranked by cosine similarity. It uses a pretrained embedding model; no model training is included. The dataset starts empty so example places are not mistaken for real listings. Keep descriptions short; inputs that exceed the model's context window return an error.

## Experiment plan

Collect 30–50 Fort Smith spots and prepare 10–20 queries. Before running searches, manually identify relevant spot IDs for each query. Compare the top three semantic results with keyword search on the same data, including queries that use different wording from the descriptions.

Target: at least 80% of queries have a relevant spot in the top three, and semantic search outperforms keyword search on those meaning-based queries. Record the query, expected IDs, returned IDs, and whether a relevant spot appeared in the top three. Keyword search and evaluation are next steps; this commit only provides the semantic search starter. No experiment results have been collected yet.

The starter model can be changed with `--model MODEL_NAME`. Embeddings are recomputed on each run for now. Similarity scores are rankings, not probabilities or guarantees of suitability.

Reference: [Ollama embeddings API](https://docs.ollama.com/api/embed).
