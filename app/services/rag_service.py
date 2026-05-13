import json
from collections.abc import Sequence

from app.schemas.rag import RagSource
from app.services.embedding_service import EmbeddingProvider
from app.services.knowledge_base_service import KnowledgeBaseService, RetrievalResult


class RagService:
    _SYMPTOM_KEYWORDS = (
        "发烧",
        "发热",
        "高烧",
        "退烧",
        "呕吐",
        "吐",
        "拉稀",
        "腹泻",
        "便血",
        "尿血",
        "掉毛",
        "脱毛",
        "睡觉不规律",
        "睡觉不太规律",
        "睡眠不规律",
        "作息不规律",
        "精神不好",
        "精神不太好",
        "精神差",
        "没精神",
        "不吃饭",
        "食欲不好",
        "食欲下降",
        "咳嗽",
        "瘙痒",
        "红肿",
        "发抖",
        "呼吸急促",
        "喘",
        "抽搐",
        "中毒",
        "骨折",
        "外伤",
        "感染",
        "发炎",
        "炎症",
        "皮肤病",
        "缺维生素",
        "缺钙",
        "缺铁",
        "贫血",
        "用药",
        "剂量",
        "抗生素",
        "治疗",
        "诊断",
        "急救",
        "急症",
        "药物",
    )
    _SYMPTOM_INTENT_PATTERNS = (
        "是什么问题",
        "什么问题",
        "什么原因",
        "是不是缺",
        "是不是生病",
        "是不是病了",
        "怎么处理",
        "怎么回事",
        "要不要紧",
        "正常吗",
    )
    _STRICT_DIAGNOSTIC_PATTERNS = (
        "是什么问题",
        "什么问题",
        "什么原因",
        "是不是缺",
        "是不是生病",
        "是不是病了",
        "怎么处理",
    )
    _ANSWER_SCORE_GAP = 0.08
    _ANSWER_SECONDARY_FLOOR = 0.36
    _SOURCE_SCORE_GAP = 0.06
    _SOURCE_SECONDARY_FLOOR = 0.46
    _SYMPTOM_SOURCE_SECONDARY_FLOOR = 0.58

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        knowledge_base_service: KnowledgeBaseService,
        llm_service,
        answer_score_threshold: float,
        source_score_threshold: float,
        symptom_answer_score_threshold: float,
        symptom_source_score_threshold: float,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._knowledge_base_service = knowledge_base_service
        self._llm_service = llm_service
        self._answer_score_threshold = answer_score_threshold
        self._source_score_threshold = source_score_threshold
        self._symptom_answer_score_threshold = symptom_answer_score_threshold
        self._symptom_source_score_threshold = symptom_source_score_threshold

    async def query(self, *, user_message: str, top_k: int) -> tuple[str, list[RagSource]]:
        self._knowledge_base_service.ensure_ready()
        normalized_question = user_message.strip()
        if self._is_strict_diagnostic_query(normalized_question):
            return self._fallback_answer(symptom_like=True), []

        is_symptom_query = self._is_symptom_or_abnormality_query(normalized_question)

        query_vector = await self._embedding_provider.embed_query(normalized_question)
        hits = self._knowledge_base_service.retrieve(query_vector=query_vector, top_k=top_k)
        answer_hits = self._select_answer_hits(hits, is_symptom_query=is_symptom_query)
        if not answer_hits:
            return self._fallback_answer(symptom_like=is_symptom_query), []

        source_hits = self._select_source_hits(answer_hits, is_symptom_query=is_symptom_query)

        answer, _ = await self._llm_service.complete_text(
            messages=self._build_answer_messages(
                user_message=normalized_question,
                hits=answer_hits,
                symptom_like=is_symptom_query,
            ),
            temperature=0.2,
        )
        cleaned_answer = answer.strip() or self._fallback_answer(symptom_like=is_symptom_query)
        return cleaned_answer, [self._to_source(hit) for hit in source_hits]

    @classmethod
    def _is_symptom_or_abnormality_query(cls, user_message: str) -> bool:
        lowered = user_message.lower()
        if any(keyword in user_message or keyword in lowered for keyword in cls._SYMPTOM_KEYWORDS):
            return True
        return any(pattern in user_message for pattern in cls._SYMPTOM_INTENT_PATTERNS)

    @classmethod
    def _is_strict_diagnostic_query(cls, user_message: str) -> bool:
        return any(pattern in user_message for pattern in cls._STRICT_DIAGNOSTIC_PATTERNS)

    def _select_answer_hits(
        self,
        hits: Sequence[RetrievalResult],
        *,
        is_symptom_query: bool,
    ) -> list[RetrievalResult]:
        if not hits:
            return []

        primary_score = hits[0].score
        primary_threshold = (
            self._symptom_answer_score_threshold
            if is_symptom_query
            else self._answer_score_threshold
        )
        if primary_score < primary_threshold:
            return []

        min_score = max(self._ANSWER_SECONDARY_FLOOR, primary_score - self._ANSWER_SCORE_GAP)
        return [hit for hit in hits if hit.score >= min_score]

    def _select_source_hits(
        self,
        hits: Sequence[RetrievalResult],
        *,
        is_symptom_query: bool,
    ) -> list[RetrievalResult]:
        if not hits:
            return []

        primary_score = hits[0].score
        primary_threshold = (
            self._symptom_source_score_threshold
            if is_symptom_query
            else self._source_score_threshold
        )
        if primary_score < primary_threshold:
            return []

        secondary_floor = (
            self._SYMPTOM_SOURCE_SECONDARY_FLOOR
            if is_symptom_query
            else self._SOURCE_SECONDARY_FLOOR
        )
        min_score = max(secondary_floor, primary_score - self._SOURCE_SCORE_GAP)
        return [hit for hit in hits if hit.score >= min_score]

    @staticmethod
    def _fallback_answer(*, symptom_like: bool) -> str:
        if symptom_like:
            return (
                "当前知识库主要覆盖狗狗的日常照护、基础训练、作息和喂养入门，"
                "还没有足够直接的依据来回答这类涉及症状判断、原因推测或处理建议的问题。"
                "如果已经出现明显异常，建议不要只依赖当前知识库的内容。"
            )
        return (
            "当前知识库里还没有足够直接的依据来回答这个问题。"
            "你可以换一种更具体的狗狗日常照护问题再试试。"
        )

    @staticmethod
    def _build_answer_messages(
        *,
        user_message: str,
        hits: Sequence[RetrievalResult],
        symptom_like: bool,
    ) -> list[dict[str, str]]:
        context_blocks = [
            {
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.title,
                "chunk_id": hit.chunk.chunk_id,
                "source": hit.chunk.source,
                "text": hit.chunk.text,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ]
        system_prompt = (
            "你是一名宠物知识问答助手，当前只回答狗狗日常照护、基础训练、"
            "作息安排、清洁梳理、奖励使用和基础喂养相关问题。"
            "你必须只依据提供的知识片段作答，不要编造知识库里没有的事实。"
            "不要给出医疗诊断、用药建议或急症处理判断。"
            "如果知识片段不足以支撑明确结论，要直接说明当前知识库未覆盖或依据不足。"
            "请使用简体中文，语气清晰、温和、实用。"
        )
        if symptom_like:
            system_prompt += (
                "当前问题带有症状、异常或原因推测色彩时，回答应更谨慎，"
                "优先强调知识范围有限，不要把日常照护知识包装成诊断结论。"
            )
        user_prompt = {
            "question": user_message,
            "knowledge_context": context_blocks,
            "instruction": "请基于以上知识片段回答问题。不要输出 JSON，只返回自然语言答案。",
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ]

    @staticmethod
    def _to_source(hit: RetrievalResult) -> RagSource:
        return RagSource(
            doc_id=hit.chunk.doc_id,
            title=hit.chunk.title,
            snippet=hit.snippet,
            source=hit.chunk.source,
            chunk_id=hit.chunk.chunk_id,
            score=round(hit.score, 4),
        )
