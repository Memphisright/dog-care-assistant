import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.services.embedding_service import create_embedding_provider
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.rag_evaluation import (
    DeterministicOfflineLLM,
    RecordingEmbeddingProvider,
    blocked_report,
    evaluate_cases,
    format_console_summary,
    load_evaluation_cases,
    write_report,
)
from app.services.rag_service import RagService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the existing RAG retrieval and source policy offline."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "tests" / "data" / "rag_eval_cases.jsonl",
        help="JSONL evaluation data path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "rag_eval_report.json",
        help="JSON report output path.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Defaults to configured RAG top_k.")
    args = parser.parse_args()
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be at least 1")

    settings = get_settings()
    top_k = args.top_k or settings.rag_default_top_k
    cases = load_evaluation_cases(args.cases)
    knowledge_base_service = KnowledgeBaseService(
        index_path=settings.rag_index_path,
        score_threshold=settings.rag_score_threshold,
    )
    knowledge_base_service.ensure_ready()
    embedding_provider = RecordingEmbeddingProvider(
        create_embedding_provider(
            mode=settings.rag_embedding_mode,
            model_name=settings.rag_embedding_model,
        )
    )
    offline_llm = DeterministicOfflineLLM()
    rag_service = RagService(
        embedding_provider=embedding_provider,
        knowledge_base_service=knowledge_base_service,
        llm_service=offline_llm,
        answer_score_threshold=settings.rag_answer_score_threshold,
        source_score_threshold=settings.rag_source_score_threshold,
        symptom_answer_score_threshold=settings.rag_symptom_answer_score_threshold,
        symptom_source_score_threshold=settings.rag_symptom_source_score_threshold,
    )
    import asyncio

    try:
        report = asyncio.run(
            evaluate_cases(
                cases=cases,
                rag_service=rag_service,
                knowledge_base_service=knowledge_base_service,
                embedding_provider=embedding_provider,
                offline_llm=offline_llm,
                top_k=top_k,
            )
        )
    except AppError as exc:
        report = blocked_report(
            top_k=top_k,
            dataset_case_count=len(cases),
            reason=(
                f"{exc.code}: the configured local embedding model is unavailable. "
                "Cache or install the configured model, then rerun the evaluation."
            ),
        )
        report["configuration"] = {
            "cases_path": str(args.cases),
            "index_path": settings.rag_index_path,
            "embedding_model": settings.rag_embedding_model,
        }
        write_report(report, args.report)
        print(f"Evaluation blocked: {report['blocker']}", file=sys.stderr)
        print(f"report: {args.report}")
        raise SystemExit(2) from exc
    report["configuration"] = {
        "cases_path": str(args.cases),
        "index_path": settings.rag_index_path,
        "embedding_model": settings.rag_embedding_model,
    }
    write_report(report, args.report)
    print(format_console_summary(report))
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
