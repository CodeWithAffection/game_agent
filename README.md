
A RAG agent built with LangChain and Claude that answers questions about games. It indexes markdown files from `knowledge-base/` into ChromaDB and retrieves relevant context before passing your query to the model.

## Setup

```bash
git clone https://github.com/CodeWithAffection/game_agent.git
cd game_agent
pip install -r requirements.txt
cp .env.example .env
```

Fill in `ANTHROPIC_API_KEY` in `.env`. Everything else has sensible defaults.

## Usage

Index the knowledge base first (re-run whenever you update `knowledge-base/`):

```bash
python ingest.py
```

Then run the agent. Set your query via the `MESSAGE` variable in `.env`:

```bash
python rag.py
```
