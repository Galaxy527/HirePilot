"""One-shot: reindex all ready resumes with new chunking policy."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app
from app.services.rag import get_rag

app = create_app()
with app.app_context():
    n = get_rag().reindex_all_from_db()
    print(f"reindexed_chunks={n}")
