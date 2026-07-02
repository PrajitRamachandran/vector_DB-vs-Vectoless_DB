import json
from pathlib import Path
from tqdm import tqdm

from vector_rag.pipeline import VectorRAGPipeline
from vector_rag.retriever import retrieve
from vector_rag.indexer import get_chroma_collection

OUTPUT_DIR = Path(
    "finetuning/training/generated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

QUESTIONS_PATH = Path(
    "evaluation/test_questions_test.json"
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "verified_train_pairs.jsonl"
)


def load_questions():

    with open(
        QUESTIONS_PATH,
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data["questions"]


def verify_chunks_json(rag):

    print("\n" + "=" * 80)
    print("VERIFYING PARENT LOOKUP")
    print("=" * 80)

    parent_ids = set(
        rag.parent_lookup.keys()
    )

    print(
        "Parent Count:",
        len(parent_ids)
    )

    return parent_ids


def verify_chroma():

    print("\n" + "=" * 80)
    print("VERIFYING CHROMADB")
    print("=" * 80)

    collection = get_chroma_collection()

    count = collection.count()

    print(
        "Chroma Vector Count:",
        count
    )

    return count


def save_jsonl(dataset):

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for row in dataset:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"\nSaved: {OUTPUT_PATH.resolve()}"
    )


def generate_pairs():

    rag = VectorRAGPipeline()

    verify_chunks_json(rag)

    verify_chroma()

    questions = load_questions()

    dataset = []

    mismatches = []

    print("\n" + "=" * 80)
    print("GENERATING PAIRS")
    print("=" * 80)

    for item in tqdm(questions):

        question = item["question"]

        expected_company = (
            item["company"]
            .upper()
            .replace("-", "")
            .replace(" ", "")
        )

        result = retrieve(
            query=question,
            collection=rag.collection,
            parent_lookup=rag.parent_lookup,
            top_k=20
        )

        chunks = result["chunks"]

        if not chunks:

            print(
                f"\n❌ No chunks for:\n{question}"
            )

            continue

        top = chunks[0]

        retrieved_company = (
            top["metadata"]
            .get("company", "")
            .upper()
            .replace("-", "")
            .replace(" ", "")
        )

        parent_id = (
            top["metadata"]
            .get("parent_id")
        )

        parent_company = None

        if parent_id in rag.parent_lookup:

            parent_company = (
                rag.parent_lookup[parent_id]
                .get("company")
            )

        print("\n" + "-" * 80)

        print("QUESTION:")
        print(question)

        print("\nEXPECTED:")
        print(item["company"])

        print("\nRETRIEVED:")
        print(retrieved_company)

        print("\nPARENT COMPANY:")
        print(parent_company)

        print("\nPARENT ID:")
        print(parent_id)

        print("\nCHILD CHUNK:")
        print(
            top["child_text"][:400]
        )

        print("\nPARENT CHUNK:")
        print(
            top["text"][:400]
        )

        if (
            expected_company
            !=
            retrieved_company
        ):

            mismatches.append(
                {
                    "question": question,
                    "expected": item["company"],
                    "retrieved": retrieved_company
                }
            )

            print(
                "\n❌ COMPANY MISMATCH"
            )

        else:

            print(
                "\n✅ COMPANY MATCH"
            )

        dataset.append(
            {
                "question":
                    question,

                "positive_chunk":
                    top["child_text"],

                "parent_chunk":
                    top["text"],

                "company":
                    item["company"],

                "retrieved_company":
                    retrieved_company,

                "parent_company":
                    parent_company,

                "parent_id":
                    parent_id,

                "category":
                    item["category"],

                "score":
                    top.get(
                        "rerank_score",
                        top.get(
                            "score",
                            0.0
                        )
                    )
            }
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "Questions:",
        len(questions)
    )

    print(
        "Generated Pairs:",
        len(dataset)
    )

    print(
        "Company Mismatches:",
        len(mismatches)
    )

    if mismatches:

        print("\nMISMATCHES")

        for x in mismatches:

            print(
                f"{x['expected']} -> "
                f"{x['retrieved']}"
            )

            print(
                x["question"]
            )

            print()

    save_jsonl(dataset)

    return dataset


if __name__ == "__main__":

    generate_pairs()