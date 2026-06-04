# evaluation/evaluator.py
import sys
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import types
sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.rate_limiter import RateLimiter

# NEW — always resolves correctly regardless of where it's called from
_HERE          = Path(__file__).parent                  # → evaluation/
QUESTIONS_PATH = _HERE / "test_questions.json"          # → evaluation/test_questions.json
RESULTS_DIR    = _HERE / "results"                      # → evaluation/results/


# ─────────────────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────────────────

def load_judge():
    """
    Creates a Gemini 2.5 Flash client using the new google.genai SDK.
    """
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in .env\n"
            "Get a free key at aistudio.google.com"
        )
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    print(f"✅ Gemini judge ready — model: {config.GEMINI_MODEL_ID}")
    return client


def score_answer(judge_client, question: str,
                 answer: str, context: str) -> dict:
    """
    Gemini 2.5 Flash fix:
    - thinking_budget=0  → disables thinking mode (we don't need chain-of-thought
                           for a 1-5 scoring task — it just adds noise and splits
                           the response into parts that break JSON parsing)
    - regex fallback     → extracts JSON even if model adds preamble text
    """
    prompt = f"""You are an expert financial analyst evaluating an answer
about a company 10-K annual report.

Question: {question}

Retrieved Context:
{context[:1500]}

Answer given: {answer}

Score the answer from 1 to 5:
5 = Correct and specific — exact figures from context
4 = Mostly correct — right topic, minor detail missing
3 = Partially correct — vague or incomplete
2 = Mostly wrong — right topic but wrong company or figures
1 = Completely wrong or hallucinated

Respond with ONLY this JSON, no other text:
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        response = judge_client.models.generate_content(
            model    = config.GEMINI_MODEL_ID,
            contents = prompt,
            config   = types.GenerateContentConfig(
                temperature    = 0.0,
                max_output_tokens = 150,
                thinking_config= types.ThinkingConfig(
                    thinking_budget=0   # disable thinking — keeps response clean
                )
            )
        )

        # Collect text from ALL response parts
        # (thinking mode splits into multiple parts — grab everything)
        full_text = ""
        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    full_text += part.text
        except Exception:
            full_text = response.text or ""

        full_text = full_text.strip()

        # ── Attempt 1: direct JSON parse ──────────────────
        try:
            result = json.loads(full_text)
            return {
                "score" : int(result["score"]),
                "reason": str(result["reason"])
            }
        except json.JSONDecodeError:
            pass

        # ── Attempt 2: regex extract {…} from anywhere ───
        import re
        match = re.search(r'\{[^{}]+\}', full_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {
                "score" : int(result["score"]),
                "reason": str(result["reason"])
            }

        # ── Attempt 3: nothing worked — show raw for debug ─
        print(f"\n   ⚠️  Could not extract JSON. Full response:\n   {repr(full_text[:300])}")
        return {"score": 0, "reason": f"Parse failed — raw: {full_text[:80]}"}

    except Exception as e:
        return {"score": 0, "reason": f"Scoring failed: {str(e)}"}

# ─────────────────────────────────────────────────────────
# RETRIEVAL METRICS
# ─────────────────────────────────────────────────────────

def compute_retrieval_metrics(question_company: str,
                               retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "company_accuracy": 0.0,
            "avg_score"       : 0.0,
            "correct_chunks"  : 0,
            "total_chunks"    : 0
        }

    companies = [
        c.get("metadata", {}).get("company", "")
        for c in retrieved_chunks
    ]
    correct   = sum(1 for c in companies if c == question_company)
    total     = len(retrieved_chunks)
    avg_score = sum(
        c.get("rerank_score", c.get("score", 0))
        for c in retrieved_chunks
    ) / total

    return {
        "company_accuracy": round(correct / total, 3),
        "avg_score"       : round(avg_score, 4),
        "correct_chunks"  : correct,
        "total_chunks"    : total
    }


# ─────────────────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────────────────

def run_evaluation(vector_pipeline, vectorless_pipeline) -> pd.DataFrame:
    """
    Runs both pipelines on every question with rate limiting.

    Rate limiters are created here and only here — they have
    zero effect on regular pipeline.ask() calls outside evaluation.

    Per question flow:
    1. [Groq limiter] → vector_pipeline.ask()
    2. [Groq limiter] → vectorless_pipeline.ask()
    3. [Gemini limiter] → score_answer() for vector result
    4. [Gemini limiter] → score_answer() for vectorless result
    """
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    judge_model   = load_judge()

    # Rate limiters — only alive inside this function
    groq_limiter   = RateLimiter(max_rpm=config.GROQ_RPM,   name="Groq")
    gemini_limiter = RateLimiter(max_rpm=config.GEMINI_RPM, name="Gemini")

    records = []
    total   = len(questions)

    print(f"\n{'='*55}")
    print(f"   PHASE 5 — EVALUATION ({total} questions × 2 methods)")
    print(f"   Groq limit : {config.GROQ_RPM} RPM")
    print(f"   Gemini limit: {config.GEMINI_RPM} RPM")
    print(f"   Est. time  : ~{int(total * max(60/config.GROQ_RPM*2, 60/config.GEMINI_RPM*2))} seconds")
    print(f"{'='*55}\n")

    for i, q in enumerate(tqdm(questions, desc="Questions"), 1):
        qid      = q["id"]
        question = q["question"]
        company  = q["company"]
        category = q["category"]

        print(f"\n[{i}/{total}] {qid} — {company}")

        # ── Step 1: Vector RAG answer (Groq) ──────────────
        groq_limiter.wait()
        try:
            vec_result = vector_pipeline.ask(question)
        except Exception as e:
            print(f"   ❌ Vector pipeline error: {e}")
            vec_result = {
                "answer": f"ERROR: {e}", "retrieved": [],
                "retrieval_time": 0, "generation_time": 0, "total_time": 0
            }

        # ── Step 2: Vectorless RAG answer (Groq) ──────────
        groq_limiter.wait()
        try:
            vl_result = vectorless_pipeline.ask(question)
        except Exception as e:
            print(f"   ❌ Vectorless pipeline error: {e}")
            vl_result = {
                "answer": f"ERROR: {e}", "retrieved": [],
                "retrieval_time": 0, "generation_time": 0, "total_time": 0
            }

        # ── Step 3: Judge vector answer (Gemini) ──────────
        from llm import format_context
        vec_context = format_context(vec_result["retrieved"])

        gemini_limiter.wait()
        vec_judge = score_answer(
            judge_model, question,
            vec_result["answer"], vec_context
        )

        # ── Step 4: Judge vectorless answer (Gemini) ──────
        vl_context = format_context(vl_result["retrieved"])

        gemini_limiter.wait()
        vl_judge = score_answer(
            judge_model, question,
            vl_result["answer"], vl_context
        )

        print(f"   🔵 Vector     score: {vec_judge['score']}/5 — {vec_judge['reason'][:60]}")
        print(f"   🟢 Vectorless score: {vl_judge['score']}/5 — {vl_judge['reason'][:60]}")

        # ── Record both results ────────────────────────────
        for method, result, judge, context_str in [
            ("vector",     vec_result, vec_judge, vec_context),
            ("vectorless", vl_result,  vl_judge,  vl_context),
        ]:
            ret_metrics = compute_retrieval_metrics(
                company, result["retrieved"]
            )
            records.append({
                "id"              : qid,
                "company"         : company,
                "category"        : category,
                "method"          : method,
                "question"        : question,
                "answer"          : result["answer"],
                "judge_score"     : judge["score"],
                "judge_reason"    : judge["reason"],
                "pass"            : judge["score"] >= 3,
                "company_accuracy": ret_metrics["company_accuracy"],
                "avg_chunk_score" : ret_metrics["avg_score"],
                "correct_chunks"  : ret_metrics["correct_chunks"],
                "retrieval_time"  : result.get("retrieval_time", 0),
                "generation_time" : result.get("generation_time", 0),
                "total_time"      : result.get("total_time", 0),
            })

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "full_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n💾 Results saved → {out_path}")
    return df


# ─────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("   EVALUATION SUMMARY")
    print(f"{'='*55}")

    for method, label in [("vector", "Vector RAG"), ("vectorless", "Vectorless RAG")]:
        m = df[df["method"] == method]
        print(f"\n{'─'*55}")
        print(f"  {label}")
        print(f"{'─'*55}")
        print(f"  Avg judge score    : {m['judge_score'].mean():.2f} / 5")
        print(f"  Pass rate  (≥3)    : {m['pass'].mean()*100:.1f}%")
        print(f"  Company accuracy   : {m['company_accuracy'].mean()*100:.1f}%")
        print(f"  Avg retrieval time : {m['retrieval_time'].mean():.4f}s")
        print(f"  Avg generation time: {m['generation_time'].mean():.2f}s")
        print(f"  Avg total latency  : {m['total_time'].mean():.2f}s")

    print(f"\n{'─'*55}")
    print("  Score by category")
    print(f"{'─'*55}")
    pivot = df.groupby(["category", "method"])["judge_score"].mean().unstack()
    print(pivot.round(2).to_string())

    print(f"\n{'─'*55}")
    print("  Score by company")
    print(f"{'─'*55}")
    pivot2 = df.groupby(["company", "method"])["judge_score"].mean().unstack()
    print(pivot2.round(2).to_string())
    print(f"\n{'='*55}\n")