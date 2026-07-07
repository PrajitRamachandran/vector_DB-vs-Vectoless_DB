"""
Chat Interface

Financial RAG Benchmark

Supports:
- Vector RAG
- Vectorless RAG
- Hybrid RAG
- Random RAG
- Auto RAG

This page is intentionally a thin orchestration layer: all
styling, retrieval-diagram, chunk-rendering, admin-control,
export, and history-browser logic lives in
`streamlit_app/components/chat/*`, all persistence lives in
`streamlit_app/database/repository.py`, and all retrieval /
LLM logic lives in `streamlit_app/services/rag_service.py`.
"""

import time
import uuid
from datetime import datetime

import streamlit as st

import config

from streamlit_app.auth.protect_page import require_login

require_login()

from streamlit_app.auth.session_manager import is_admin

from streamlit_app.services.rag_service import (
    ask_question,
    clear_pipeline_cache,
)
from streamlit_app.services.analytics_service import (
    confidence_level,
    generate_title,
)

from streamlit_app.database.repository import (
    save_conversation,
    save_retrieved_chunks,
    save_log,
    get_chunks_for_chat,
    get_session_messages,
    get_session_stats,
    clear_session,
)

from streamlit_app.components.chat.styles import (
    inject_chat_styles,
    method_badge_html,
    confidence_html,
    risk_html,
)
from streamlit_app.components.chat.welcome import render_welcome_screen
from streamlit_app.components.chat.chunks import render_chunk_panel
from streamlit_app.components.chat.diagrams import (
    render_workflow_diagram,
    render_stage_tracker,
)
from streamlit_app.components.chat.admin import (
    render_admin_controls,
    render_query_templates,
)
from streamlit_app.components.chat.session import (
    render_session_summary_card,
    render_health_panel,
)
from streamlit_app.components.chat.history import render_history_browser
from streamlit_app.components.chat.export_panel import render_export_panel
from streamlit_app.components.chat.message_actions import (
    render_message_actions,
    render_citations,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="wide"
)

inject_chat_styles()

# ============================================================
# SESSION IDENTITY (persists across a hard refresh via the URL,
# not just Streamlit's in-memory session_state)
# ============================================================

_query_params = st.query_params

if "sid" in _query_params:
    st.session_state.session_id = _query_params["sid"]
elif "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    _query_params["sid"] = st.session_state.session_id
else:
    _query_params["sid"] = st.session_state.session_id

# ============================================================
# SESSION STATE INIT + AUTO-RECOVERY AFTER REFRESH
# ============================================================

if "chat_history" not in st.session_state:

    recovered = []

    try:
        past_rows = get_session_messages(st.session_state.session_id)
    except Exception:
        past_rows = []

    for row in past_rows:
        recovered.append({
            "role": "user",
            "content": row.get("prompt", ""),
            "timestamp": row.get("timestamp"),
        })

        recovered.append({
            "role": "assistant",
            "content": row.get("response", ""),
            "chat_id": row.get("chat_id"),
            "method": row.get("method"),
            "timestamp": row.get("timestamp"),
            "confidence": row.get("confidence_score") or 0,
            "confidence_level": confidence_level(row.get("confidence_score") or 0),
            "hallucination_risk": row.get("hallucination_risk") or "unknown",
            "query_type": row.get("query_type") or "unknown",
            "company_filter": row.get("company_filter"),
            "retrieved": get_chunks_for_chat(row.get("chat_id")) if row.get("chat_id") else [],
            "rag_result": {
                "retrieval_time": row.get("retrieval_latency"),
                "rerank_latency": row.get("rerank_latency"),
                "generation_time": row.get("generation_latency"),
                "total_time": row.get("total_latency"),
                "vector_candidates": row.get("vector_candidates"),
                "bm25_candidates": row.get("bm25_candidates"),
                "fused_candidates": row.get("fused_candidates"),
            },
            "question": row.get("prompt", ""),
            "status": row.get("status", "SUCCESS"),
            "error": row.get("error_message"),
        })

    st.session_state.chat_history = recovered

if "_pending_question" not in st.session_state:
    st.session_state._pending_question = None

# ============================================================
# HEADER
# ============================================================

st.title("💬 Financial RAG Chat")
st.caption("Ask questions about company 10-K reports.")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("Retrieval Method")

    retrieval_method = st.radio(
        label="Choose Method",
        options=["Hybrid", "Vector", "Vectorless", "Random", "Auto"],
        index=0
    )

    compare_mode = st.toggle(
        "⚔️ Compare mode (A/B two methods)",
        value=False,
        help="Runs the question through two retrieval methods and "
             "shows the answers side by side."
    )

    compare_method_b = None
    if compare_mode:
        compare_method_b = st.selectbox(
            "Compare against",
            [m for m in ["Hybrid", "Vector", "Vectorless", "Random", "Auto"]
             if m != retrieval_method],
            key="compare_method_b"
        )

    st.divider()

    benchmark_mode = st.toggle(
        "🧪 Benchmark mode",
        value=False,
        help="Tags saved conversations for evaluation instead of "
             "normal usage tracking."
    )
    benchmark_tag = None
    if benchmark_mode:
        benchmark_tag = st.text_input("Benchmark tag", value="untagged")

    debug_mode = st.toggle(
        "🐞 Debug mode",
        value=False,
        help="Shows advanced retrieval diagnostics (query "
             "classification, candidate counts, confidence, "
             "coverage stats). Hidden by default for normal users."
    )

    st.divider()

    admin_cfg = {"overrides": {}, "model_name": config.LLM_MODEL_ID}

    if is_admin():
        with st.expander("🛠 Admin Controls", expanded=False):
            admin_cfg = render_admin_controls()

            st.markdown("**Audit Log**")
            if st.button("View recent system logs", key="view_logs_btn"):
                from streamlit_app.database.repository import get_logs
                for log in get_logs(limit=20):
                    st.caption(f"[{log['timestamp']}] {log['level']} · {log['module']} · {log['message']}")

    with st.expander("📋 Query Templates", expanded=False):
        template_text = render_query_templates()
        if template_text:
            st.session_state._pending_question = template_text

    st.divider()
    st.subheader("Utilities")

    if st.button("♻ Reload Pipelines", use_container_width=True):
        if st.session_state.get("_confirm_reload"):
            clear_pipeline_cache()
            st.session_state["_confirm_reload"] = False
            st.success("Pipeline cache cleared.")
        else:
            st.session_state["_confirm_reload"] = True
            st.warning("Click again to confirm reloading all pipelines.")

    if st.button("🗑 Clear Chat", use_container_width=True):
        if st.session_state.get("_confirm_clear"):
            clear_session(st.session_state.session_id)
            st.session_state.chat_history = []
            st.session_state["_confirm_clear"] = False
            st.rerun()
        else:
            st.session_state["_confirm_clear"] = True
            st.warning("Click again to confirm clearing this chat.")

    st.divider()

    with st.expander("📊 Session Summary", expanded=True):
        stats = get_session_stats(st.session_state.session_id)
        render_session_summary_card(stats)

    with st.expander("📤 Export", expanded=False):
        export_rows = [
            {
                "timestamp": m.get("timestamp"),
                "title": generate_title(m.get("question", "")),
                "method": m.get("method"),
                "model_name": admin_cfg["model_name"],
                "prompt": m.get("question"),
                "response": m.get("content"),
                "company_filter": m.get("company_filter"),
                "total_latency": (m.get("rag_result") or {}).get("total_time"),
                "status": m.get("status", "SUCCESS"),
            }
            for m in st.session_state.chat_history
            if m["role"] == "assistant"
        ]
        render_export_panel(export_rows)

    with st.expander("🗂 History", expanded=False):
        reopen_id = render_history_browser(
            st.session_state.get("user_id"), is_admin()
        )
        if reopen_id:
            from streamlit_app.database.repository import get_chat
            past = get_chat(reopen_id)
            if past["conversation"]:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": past["conversation"]["prompt"],
                    "timestamp": past["conversation"]["timestamp"],
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": past["conversation"]["response"],
                    "chat_id": reopen_id,
                    "method": past["conversation"]["method"],
                    "timestamp": past["conversation"]["timestamp"],
                    "confidence": past["conversation"].get("confidence_score") or 0,
                    "confidence_level": confidence_level(past["conversation"].get("confidence_score") or 0),
                    "hallucination_risk": past["conversation"].get("hallucination_risk") or "unknown",
                    "query_type": past["conversation"].get("query_type") or "unknown",
                    "company_filter": past["conversation"].get("company_filter"),
                    "retrieved": past["chunks"],
                    "rag_result": {
                        "retrieval_time": past["conversation"].get("retrieval_latency"),
                        "rerank_latency": past["conversation"].get("rerank_latency"),
                        "generation_time": past["conversation"].get("generation_latency"),
                        "total_time": past["conversation"].get("total_latency"),
                    },
                    "question": past["conversation"]["prompt"],
                    "status": past["conversation"].get("status", "SUCCESS"),
                })
                st.rerun()

    with st.expander("🩺 System Health", expanded=False):
        render_health_panel()

    with st.expander("🔗 Navigation", expanded=False):
        if st.button("📊 Analytics Dashboard", use_container_width=True):
            try:
                st.switch_page("pages/01_dashboard.py")
            except Exception:
                st.info("Dashboard page not found.")
        if st.button("🧪 Evaluation Results", use_container_width=True):
            try:
                st.switch_page("pages/evaluation.py")
            except Exception:
                st.info("Evaluation page not found.")

    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        st.caption("**Enter** — send message")
        st.caption("**Ctrl/Cmd + K** — focus the message box")

# Focus-the-input keyboard shortcut (front-end only, no
# round-trip to the server required).
st.components.v1.html(
    """
    <script>
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
            const parentDoc = window.parent.document;
            const box = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (box) { box.focus(); e.preventDefault(); }
        }
    });
    </script>
    """,
    height=0
)

# ============================================================
# QUESTION INPUT
# ============================================================

typed_question = st.chat_input("Ask a financial question...")

question = st.session_state._pending_question or typed_question
st.session_state._pending_question = None

# ============================================================
# WELCOME SCREEN (no history yet)
# ============================================================

if not st.session_state.chat_history and not question:
    starter_clicked = render_welcome_screen(config.STARTER_QUESTIONS)
    if starter_clicked:
        question = starter_clicked

# ============================================================
# DISPLAY HISTORY
# ============================================================

for idx, msg in enumerate(st.session_state.chat_history):

    with st.chat_message(msg["role"]):

        st.markdown(
            f'<div class="msg-container">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

        ts = msg.get("timestamp")
        if ts:
            st.markdown(
                f'<div class="msg-timestamp">🕒 {ts}</div>',
                unsafe_allow_html=True
            )

        if msg["role"] == "assistant" and msg.get("status", "SUCCESS") == "SUCCESS":

            method_label = msg.get("method") or retrieval_method
            badge_cols = st.columns([3, 2, 2])

            with badge_cols[0]:
                st.markdown(method_badge_html(method_label), unsafe_allow_html=True)
            with badge_cols[1]:
                st.markdown(
                    confidence_html(msg.get("confidence", 0), msg.get("confidence_level", "low")),
                    unsafe_allow_html=True
                )
            with badge_cols[2]:
                if debug_mode or is_admin():
                    st.markdown(risk_html(msg.get("hallucination_risk", "unknown")), unsafe_allow_html=True)

            if debug_mode or is_admin():
                q_type = msg.get("query_type", "unknown")
                st.caption(
                    f"Query type: {config.QUERY_TYPE_LABELS.get(q_type, q_type)}"
                    + (f" · Company detected: {msg['company_filter']}" if msg.get("company_filter") else "")
                )

                rr = msg.get("rag_result") or {}
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Retrieval", f"{(rr.get('retrieval_time') or 0):.2f}s")
                with m2:
                    st.metric("Rerank", f"{(rr.get('rerank_latency') or rr.get('rerank_time') or 0):.2f}s")
                with m3:
                    st.metric("Generation", f"{(rr.get('generation_time') or 0):.2f}s")
                with m4:
                    st.metric("Total", f"{(rr.get('total_time') or 0):.2f}s")

                if any(rr.get(k) is not None for k in ("vector_candidates", "bm25_candidates", "fused_candidates")):
                    st.caption(
                        f"Vector candidates: {rr.get('vector_candidates', '-')} · "
                        f"BM25 candidates: {rr.get('bm25_candidates', '-')} · "
                        f"Fused candidates: {rr.get('fused_candidates', '-')}"
                    )

            key_prefix = f"msg_{idx}_{msg.get('chat_id', idx)}"

            render_citations(msg.get("retrieved", []), key_prefix)

            answer_terms = [w for w in (msg.get("question") or "").split() if len(w) > 3]
            render_chunk_panel(msg.get("retrieved", []), key_prefix, answer_terms)

            if msg.get("chat_id"):
                intents = render_message_actions(
                    msg["chat_id"], msg["content"], msg.get("question", ""), key_prefix
                )
                if intents["regenerate"]:
                    st.session_state._pending_question = msg.get("question", "")
                    st.rerun()
                if intents["edit_resend"]:
                    st.session_state._pending_question = intents["edit_resend"]
                    st.rerun()

        elif msg["role"] == "assistant":
            st.error(msg.get("error", "Something went wrong."))
            if st.button("🔁 Retry", key=f"retry_{idx}"):
                st.session_state._pending_question = msg.get("question", "")
                st.rerun()


# ============================================================
# CORE ANSWER GENERATION (shared by normal + compare mode)
# ============================================================

def _run_pipeline(question: str, method: str, overrides: dict, model_name: str):

    return ask_question(
        question=question,
        method=method.lower(),
        overrides=overrides,
        model_name=model_name,
    )


def _method_to_store(retrieval_method: str, rag_result: dict) -> str:

    if retrieval_method == "Random":
        return f"Random({rag_result.get('random_selected_method', 'unknown')})"

    if retrieval_method == "Auto":
        return f"Auto({(rag_result.get('auto_selected_method') or 'unknown')})"

    return retrieval_method


def _persist_and_render(question: str, retrieval_method: str, result: dict, elapsed: float):

    if not result["success"]:
        save_conversation(
            session_id=st.session_state.session_id,
            user_id=st.session_state.user_id,
            method=retrieval_method,
            model_name=admin_cfg["model_name"],
            prompt=question,
            response="",
            status="FAILED",
            error_message=result["error"],
            is_benchmark=benchmark_mode,
            benchmark_tag=benchmark_tag,
        )
        st.error(result["error"])
        with st.expander("Traceback"):
            st.code(result["traceback"])
        if st.button("🔁 Retry", key=f"retry_live_{len(st.session_state.chat_history)}"):
            st.session_state._pending_question = question
            st.rerun()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "",
            "status": "FAILED",
            "error": result["error"],
            "question": question,
        })
        return

    rag_result = result["result"]
    intent = rag_result.get("intent", "unknown")
    answer = rag_result.get("answer", "No answer generated.")

    method_to_store = _method_to_store(retrieval_method, rag_result)
    if intent == "chat":
        method_to_store = "CHAT"
    elif intent == "general_knowledge":
        method_to_store = "GENERAL_KNOWLEDGE"

    confidence = rag_result.get("confidence", 0)
    conf_level = confidence_level(confidence)
    hallucination_risk = rag_result.get("hallucination_risk", "unknown")
    query_type = rag_result.get("query_type", "unknown")
    title = generate_title(question)

    chat_id = save_conversation(
        session_id=st.session_state.session_id,
        user_id=st.session_state.user_id,
        method=method_to_store,
        model_name=admin_cfg["model_name"],
        prompt=question,
        response=answer,
        company_filter=rag_result.get("company_filter"),
        retrieval_latency=rag_result.get("retrieval_time"),
        rerank_latency=rag_result.get("rerank_latency") or rag_result.get("rerank_time"),
        generation_latency=rag_result.get("generation_time"),
        total_latency=rag_result.get("total_time"),
        vector_candidates=rag_result.get("vector_candidates"),
        bm25_candidates=rag_result.get("bm25_candidates"),
        fused_candidates=rag_result.get("fused_candidates"),
        status="SUCCESS",
        title=title,
        query_type=query_type,
        confidence_score=confidence,
        hallucination_risk=hallucination_risk,
        tokens_prompt=rag_result.get("tokens_prompt"),
        tokens_completion=rag_result.get("tokens_completion"),
        estimated_cost_usd=rag_result.get("estimated_cost_usd"),
        is_benchmark=benchmark_mode,
        benchmark_tag=benchmark_tag,
    )

    retrieved_chunks = rag_result.get("retrieved", [])
    if retrieved_chunks:
        save_retrieved_chunks(chat_id, retrieved_chunks)

    save_log("INFO", "chat", f"Answered via {method_to_store} in {elapsed}s")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "chat_id": chat_id,
        "method": method_to_store,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "confidence": confidence,
        "confidence_level": conf_level,
        "hallucination_risk": hallucination_risk,
        "query_type": query_type,
        "company_filter": rag_result.get("company_filter"),
        "retrieved": retrieved_chunks,
        "rag_result": rag_result,
        "question": question,
        "status": "SUCCESS",
    })


# ============================================================
# ASK QUESTION
# ============================================================

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        stage_box = st.empty()
        render_stage_tracker(stage_box, "routing")
        time.sleep(0.15)

        render_workflow_diagram(retrieval_method.lower())

        render_stage_tracker(stage_box, "retrieving")

        start_time = time.time()

        if compare_mode and compare_method_b:

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"**{retrieval_method}**")
                result_a = _run_pipeline(
                    question, retrieval_method, admin_cfg["overrides"], admin_cfg["model_name"]
                )
                render_stage_tracker(stage_box, "generating")
                if result_a["success"]:
                    st.markdown(result_a["result"].get("answer", ""))
                else:
                    st.error(result_a["error"])

            with col_b:
                st.markdown(f"**{compare_method_b}**")
                result_b = _run_pipeline(
                    question, compare_method_b, admin_cfg["overrides"], admin_cfg["model_name"]
                )
                if result_b["success"]:
                    st.markdown(result_b["result"].get("answer", ""))
                else:
                    st.error(result_b["error"])

            render_stage_tracker(stage_box, "done")
            elapsed = round(time.time() - start_time, 2)
            st.caption(f"Compared in {elapsed}s")

            # Persist the primary method's answer as the canonical
            # conversation record; the comparison is exploratory.
            _persist_and_render(question, retrieval_method, result_a, elapsed)

        else:

            render_stage_tracker(stage_box, "reranking")

            result = _run_pipeline(
                question, retrieval_method, admin_cfg["overrides"], admin_cfg["model_name"]
            )

            render_stage_tracker(stage_box, "generating")

            elapsed = round(time.time() - start_time, 2)

            stage_box.empty()

            if result["success"]:

                rag_result = result["result"]
                answer = rag_result.get("answer", "No answer generated.")

                # Simulated typewriter effect. The underlying LLM
                # call already returned the full answer (this app's
                # generation layer isn't token-streaming), so this
                # animates the reveal rather than truly streaming
                # tokens from the model.
                placeholder = st.empty()
                rendered = ""
                chunk_size = max(1, len(answer) // 60)
                for i in range(0, len(answer), chunk_size):
                    rendered += answer[i:i + chunk_size]
                    placeholder.markdown(rendered + "▌")
                    time.sleep(0.01)
                placeholder.markdown(answer)

                st.caption(f"Response time: {elapsed}s")

                method_label = _method_to_store(retrieval_method, rag_result)
                if rag_result.get("intent") == "chat":
                    st.success("💬 Chat Mode")
                elif rag_result.get("intent") == "general_knowledge":
                    st.success("🧠 General Knowledge Mode")
                elif retrieval_method == "Random":
                    st.success(f"🎲 Random Mode → {rag_result.get('random_selected_method', 'unknown').upper()}")
                elif retrieval_method == "Auto":
                    selected = (rag_result.get("auto_selected_method") or "unknown").upper()
                    st.success(f"🤖 Auto Mode → {selected} ({rag_result.get('query_type', 'unknown')})")
                    if debug_mode:
                        st.info(
                            f"Auto-RAG routed this as a **{rag_result.get('query_type', 'unknown')}** "
                            f"question and selected **{selected}** retrieval as the best fit."
                        )
                else:
                    st.success(f"📚 Retrieval Mode → {retrieval_method}")

                with st.expander("Raw Pipeline Output"):
                    st.json(rag_result)

            _persist_and_render(question, retrieval_method, result, elapsed)

    st.rerun()