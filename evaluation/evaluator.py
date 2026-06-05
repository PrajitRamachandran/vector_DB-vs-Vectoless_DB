# evaluation/evaluator.py
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.rate_limiter import RateLimiter

_HERE = Path(__file__).parent
QUESTIONS_PATH = _HERE / "test_questions.json"
RESULTS_DIR = _HERE / "results"


def load_judge():
    """
    Creates a Mistral judge client for evaluation.
    """
    if not config.MISTRAL_JUDGE_API_KEY:
        raise ValueError(
            "MISTRAL_JUDGE_API_KEY not found in .env. "
            "Add your evaluation key before running the benchmark."
        )

    client = OpenAI(
        api_key=config.MISTRAL_JUDGE_API_KEY,
        base_url=config.MISTRAL_BASE_URL,
    )
    print(f"Mistral judge ready - model: {config.JUDGE_MODEL_ID}")
    return client


def _extract_text_content(content) -> str:
    """
    Normalizes OpenAI/Mistral content formats into plain text.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(str(text))
            elif hasattr(item, "text"):
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))
        return " ".join(parts).strip()

    return str(content).strip()


def _parse_judge_response(full_text: str) -> Optional[dict]:
    """
    Best-effort extraction of {"score": int, "reason": str} from model output.
    """
    text = (full_text or "").strip()
    if not text:
        return None

    # 1) Direct JSON.
    try:
        result = json.loads(text)
        return {
            "score": int(result["score"]),
            "reason": str(result["reason"]),
        }
    except Exception:
        pass

    # 2) JSON blob embedded in extra text.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            return {
                "score": int(result["score"]),
                "reason": str(result["reason"]),
            }
        except Exception:
            pass

    # 3) Regex fallback for truncated/partially formatted responses.
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text, re.DOTALL)

    if score_match:
        reason = "Judge reason truncated."
        if reason_match:
            partial_reason = reason_match.group(1).strip()
            if partial_reason:
                reason = partial_reason
        return {
            "score": int(score_match.group(1)),
            "reason": reason,
        }

    return None


def score_answer(judge_client, question: str, answer: str, context: str) -> dict:
    """
    Uses a higher-capability Mistral judge model to score an answer from 1 to 5.
    """
    prompt = f"""You are an expert financial analyst evaluating an answer about a company 10-K annual report.

Question: {question}

Retrieved Context:
{context[:1500]}

Answer given: {answer}

Scoring rubric:
5 = Correct and specific - exact figures or facts supported by context
4 = Mostly correct - right conclusion with minor omission or imprecision
3 = Partially correct - some relevant truth but incomplete, vague, or weakly supported
2 = Mostly wrong - some overlap in topic but materially incorrect company, facts, or figures
1 = Completely wrong, unsupported, or hallucinated

Keep the reason under 18 words.

Return ONLY valid JSON in this exact schema:
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        response = judge_client.chat.completions.create(
            model=config.JUDGE_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict financial QA judge. "
                        "Return only compact JSON with no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=config.JUDGE_TEMPERATURE,
            max_tokens=config.JUDGE_MAX_TOKENS,
        )

        print("\n===== DEBUG =====")
        print("MODEL:", config.JUDGE_MODEL_ID)

        msg = response.choices[0].message

        print("MESSAGE:", msg)
        print("CONTENT TYPE:", type(msg.content))
        print("CONTENT:", repr(msg.content))

        if hasattr(msg, "reasoning_content"):
            print("REASONING:", repr(msg.reasoning_content))

        print("=================\n")

        full_text = _extract_text_content(response.choices[0].message.content)
        parsed = _parse_judge_response(full_text)

        if parsed is not None:
            return parsed

        print(
            f"\n   Warning: Could not extract judge JSON. "
            f"Raw response: {repr(full_text[:300])}"
        )
        return {"score": 0, "reason": f"Parse failed - raw: {full_text[:80]}"}

    except Exception as e:
        return {"score": 0, "reason": f"Scoring failed: {str(e)}"}


def compute_retrieval_metrics(question_company: str, retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "company_accuracy": 0.0,
            "avg_score": 0.0,
            "correct_chunks": 0,
            "total_chunks": 0,
        }

    companies = [c.get("metadata", {}).get("company", "") for c in retrieved_chunks]
    correct = sum(1 for c in companies if c == question_company)
    total = len(retrieved_chunks)

    scores = []
    for c in retrieved_chunks:
        score = c.get("rerank_score", c.get("score", 0))
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            scores.append(0.0)

    avg_score = sum(scores) / total if total else 0.0

    return {
        "company_accuracy": round(correct / total, 3) if total else 0.0,
        "avg_score": round(avg_score, 4),
        "correct_chunks": correct,
        "total_chunks": total,
    }


def run_evaluation(vector_pipeline, vectorless_pipeline) -> pd.DataFrame:
    """
    Runs both pipelines on every question with rate limiting.

    Per question flow:
    1. [Mistral generation limiter] -> vector_pipeline.ask()
    2. [Mistral generation limiter] -> vectorless_pipeline.ask()
    3. [Mistral judge limiter] -> score_answer() for vector result
    4. [Mistral judge limiter] -> score_answer() for vectorless result
    """
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_PATH}")

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
        questions = data.get("questions", [])

    judge_model = load_judge()

    generation_limiter = RateLimiter(max_rpm=config.MISTRAL_RPM, name="Mistral generation")
    judge_limiter = RateLimiter(max_rpm=config.MISTRAL_JUDGE_RPM, name="Mistral judge")

    records = []
    total = len(questions)

    print(f"\n{'=' * 55}")
    print(f"   PHASE 5 - EVALUATION ({total} questions x 2 methods)")
    print(f"   Generation limit: {config.MISTRAL_RPM} RPM")
    print(f"   Judge limit     : {config.MISTRAL_JUDGE_RPM} RPM")
    print(
        "   Est. time       : ~"
        f"{int(total * max(60 / config.MISTRAL_RPM * 2, 60 / config.MISTRAL_JUDGE_RPM * 2))} seconds"
    )
    print(f"{'=' * 55}\n")

    from llm import format_context

    for i, q in enumerate(tqdm(questions, desc="Questions"), 1):
        qid = q.get("id", i)
        question = q.get("question", "")
        company = q.get("company", "")
        category = q.get("category", "")

        print(f"\n[{i}/{total}] {qid} - {company}")

        generation_limiter.wait()
        try:
            vec_result = vector_pipeline.ask(question)
        except Exception as e:
            print(f"   Vector pipeline error: {e}")
            vec_result = {
                "answer": f"ERROR: {e}",
                "retrieved": [],
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": 0,
            }

        generation_limiter.wait()
        try:
            vl_result = vectorless_pipeline.ask(question)
        except Exception as e:
            print(f"   Vectorless pipeline error: {e}")
            vl_result = {
                "answer": f"ERROR: {e}",
                "retrieved": [],
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": 0,
            }

        vec_context = format_context(vec_result.get("retrieved", []))
        judge_limiter.wait()
        vec_judge = score_answer(judge_model, question, vec_result.get("answer", ""), vec_context)

        vl_context = format_context(vl_result.get("retrieved", []))
        judge_limiter.wait()
        vl_judge = score_answer(judge_model, question, vl_result.get("answer", ""), vl_context)

        print(f"   Vector     score: {vec_judge['score']}/5 - {vec_judge['reason'][:60]}")
        print(f"   Vectorless score: {vl_judge['score']}/5 - {vl_judge['reason'][:60]}")

        for method, result, judge in [
            ("vector", vec_result, vec_judge),
            ("vectorless", vl_result, vl_judge),
        ]:
            ret_metrics = compute_retrieval_metrics(company, result.get("retrieved", []))
            records.append(
                {
                    "id": qid,
                    "company": company,
                    "category": category,
                    "method": method,
                    "question": question,
                    "answer": result.get("answer", ""),
                    "judge_score": judge["score"],
                    "judge_reason": judge["reason"],
                    "pass": judge["score"] >= 3,
                    "company_accuracy": ret_metrics["company_accuracy"],
                    "avg_chunk_score": ret_metrics["avg_score"],
                    "correct_chunks": ret_metrics["correct_chunks"],
                    "retrieval_time": result.get("retrieval_time", 0),
                    "generation_time": result.get("generation_time", 0),
                    "total_time": result.get("total_time", 0),
                }
            )

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "full_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nResults saved -> {out_path}")
    return df


def print_summary(df: pd.DataFrame):
    if df.empty:
        print("No results to summarize.")
        return

    print(f"\n{'=' * 55}")
    print("   EVALUATION SUMMARY")
    print(f"{'=' * 55}")

    for method, label in [("vector", "Vector RAG"), ("vectorless", "Vectorless RAG")]:
        m = df[df["method"] == method]
        if m.empty:
            print(f"\n{label}: no rows")
            continue

        print(f"\n{'-' * 55}")
        print(f"  {label}")
        print(f"{'-' * 55}")
        print(f"  Avg judge score    : {m['judge_score'].mean():.2f} / 5")
        print(f"  Pass rate  (>=3)   : {m['pass'].mean() * 100:.1f}%")
        print(f"  Company accuracy   : {m['company_accuracy'].mean() * 100:.1f}%")
        print(f"  Avg retrieval time : {m['retrieval_time'].mean():.4f}s")
        print(f"  Avg generation time: {m['generation_time'].mean():.2f}s")
        print(f"  Avg total latency  : {m['total_time'].mean():.2f}s")

    print(f"\n{'-' * 55}")
    print("  Score by category")
    print(f"{'-' * 55}")
    pivot = df.groupby(["category", "method"])["judge_score"].mean().unstack()
    print(pivot.round(2).to_string())

    print(f"\n{'-' * 55}")
    print("  Score by company")
    print(f"{'-' * 55}")
    pivot2 = df.groupby(["company", "method"])["judge_score"].mean().unstack()
    print(pivot2.round(2).to_string())
    print(f"\n{'=' * 55}\n")
