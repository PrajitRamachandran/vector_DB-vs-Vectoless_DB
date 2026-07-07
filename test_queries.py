import json
from utils.query_processor import detect_company

class MockChunks:
    def __init__(self):
        self.parents = [
            {"company": "TESLA"},
            {"company": "APPLE"},
            {"company": "NVIDIA"}
        ]

# Monkeypatch the get_known_companies logic so we can test the regex
import utils.query_processor
utils.query_processor._COMPANIES_CACHE = ["TESLA", "APPLE", "NVIDIA"]

questions = [
    "What was Tesla's total revenue in Q1 2026?",
    "What was NVIDIA's total revenue in Fiscal 2026 Q4?",
    "What were Apple's total net sales for the three months ended December 27, 2025?",
    "Tell me about some random startup."
]

for q in questions:
    print("-" * 40)
    company = detect_company(q)
    print(f"Result for '{q}': {company}")
