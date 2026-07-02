# =================================================================================================================
# Test vector pipeline
# =================================================================================================================

# from streamlit_app.services.rag_service import (
#     get_vector_pipeline
# )

# pipe = get_vector_pipeline()

# print("Loaded successfully")

# Test database connection

# from streamlit_app.database.db import (
#     health_check
# )

# print(
#     health_check()
# )

# =================================================================================================================
# Initialise Schema
# =================================================================================================================

# from streamlit_app.database.schema import (
#     initialize_schema
# )

# initialize_schema()

# =================================================================================================================
# Test tables
# =================================================================================================================

# from streamlit_app.database.db import (
#     get_connection
# )

# conn = get_connection()

# cursor = conn.cursor()

# cursor.execute(
#     "SELECT name FROM sqlite_master WHERE type='table';"
# )

# print(cursor.fetchall())

# =================================================================================================================
# Test repository
# =================================================================================================================

# from streamlit_app.database.repository import (
#     save_conversation,
#     get_conversations
# )

# chat_id = save_conversation(
#     session_id="test",
#     method="Hybrid",
#     model_name="mistral-medium-latest",
#     prompt="What was Amazon revenue?",
#     response="Amazon revenue was ...",
# )

# print(chat_id)

# print(
#     get_conversations()
# )


# =================================================================================================================
# Memory Test
# =================================================================================================================

# from streamlit_app.database.repository import (
#     get_conversations
# )

# for row in get_conversations():

#     print(row["method"])
#     print(row["prompt"])
#     print(row["total_latency"])

# =================================================================================================================
# Initialise Schema
# =================================================================================================================

# from streamlit_app.database.schema import (
#     initialize_schema
# )

# initialize_schema()


# =================================================================================================================
# Test Schema
# =================================================================================================================

# from streamlit_app.database.db import (
#     get_connection
# )

# conn = get_connection()

# cursor = conn.cursor()

# cursor.execute(
#     "SELECT name FROM sqlite_master WHERE type='table';"
# )

# for table in cursor.fetchall():
#     print(table)

# conn.close()


# =================================================================================================================
# Test Repository
# =================================================================================================================

# from streamlit_app.database.repository import (
#     save_evaluation,
#     get_evaluations
# )

# import uuid

# save_evaluation(
#     evaluation_id=str(uuid.uuid4()),

#     method="Hybrid",

#     evaluator="Test",

#     total_questions=10,

#     avg_judge_score=4.5,
#     pass_rate=90,

#     company_accuracy=100,

#     faithfulness=0.88,
#     answer_relevancy=0.91,

#     context_precision=0.84,
#     context_recall=0.87,

#     contextual_relevancy=0.89,

#     overall_score=0.88
# )

# print(
#     get_evaluations()
# )


# from streamlit_app.database.repository import (
#     get_conversations
# )

# conversations = get_conversations()

# print(
#     f"Count: {len(conversations)}"
# )

# for c in conversations[:3]:
#     print(c)


from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data_loader import run_preprocessing_pipeline
from vectorless_rag.indexer import build_bm25_index

print("=" * 60)
print("REBUILDING BM25")
print("=" * 60)

print("\nRunning preprocessing pipeline...")
data = run_preprocessing_pipeline()

print(f"Parents : {len(data['parents'])}")
print(f"Children: {len(data['children'])}")

print("\nBuilding BM25 index...")
build_bm25_index(data)

print("\n✅ BM25 rebuild complete")