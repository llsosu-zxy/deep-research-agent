from __future__ import annotations

import logging
import re
import time

from agents.llm import OpenAICompatibleLLM
from core.config import Settings
from core.guardrails.input import redact_pii, validate_input
from core.guardrails.output import validate_citations
from core.models import GraphState, Plan, Source, SubTask, ToolResult
from core.tools.registry import ToolRegistry

KNOWN_ENTITIES = [
    "Shopee",
    "TikTok",
    "ByteDance",
    "Grab",
    "Google",
    "Meta",
    "NVIDIA",
    "A*STAR",
    "GovTech",
    "OpenAI",
    "Microsoft",
    "Alibaba",
    "Lazada",
    "Tencent",
    "Huawei",
    "Sea",
    "Garena",
    "Amazon",
    "Apple",
    "SAP",
    "PayPal",
    "Visa",
    "Salesforce",
    "DBS",
    "OCBC",
    "UOB",
    "Standard Chartered",
    "JPMorgan",
    "GIC",
    "ST Engineering",
    "Singtel",
    "Razer",
    "Micron",
    "Cynapse",
    "Guidesify",
    "AIPilot",
    "ESGPedia",
    "X Star",
    "Hong Ye",
    "YY Circle",
    "LinkWave",
]
logger = logging.getLogger(__name__)

SALARY_RE = re.compile(r"S\$\s*[\d,]+\s*(?:–|-)\s*[\d,]+")
CALC_PATTERNS = [
    (
        re.compile(r"(\d+)\s*hours?\D+(\d+)\s*days?\D+(\d+)\s*weeks?", re.IGNORECASE),
        lambda h, d, w: f"print({h} * {d} * {w})",
    ),
    (
        re.compile(r"(\d+)\s*days?\D+(\d+)\s*weeks?", re.IGNORECASE),
        lambda d, w: f"print({d} * {w})",
    ),
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), lambda a, b: f"print({a} + {b})"),
]


def _calculation_code(question: str) -> str | None:
    for pattern, builder in CALC_PATTERNS:
        match = pattern.search(question)
        if match:
            numbers = [int(value) for value in match.groups()]
            return builder(*numbers)
    return None


def _is_calculation_question(question: str) -> bool:
    has_marker = bool(re.search(r"calculate|how many|total hours|sum of", question, re.IGNORECASE))
    return has_marker and _calculation_code(question) is not None


def _merge_by_doc(results: list[dict]) -> list[dict]:
    """Merge chunks from the same document into one richer evidence item."""
    groups: dict[str, dict] = {}
    for item in results:
        doc_id = str(item.get("doc_id") or item.get("id") or "")
        if doc_id not in groups:
            groups[doc_id] = {"item": dict(item), "snippets": []}
        groups[doc_id]["snippets"].append(str(item.get("snippet", "")))
    merged = []
    for group in groups.values():
        item = group["item"]
        raw = "\n".join(group["snippets"][:3])
        seen: set[str] = set()
        clean_lines: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                clean_lines.append(stripped)
        item["snippet"] = "\n".join(clean_lines)[:1600]
        merged.append(item)
    return merged


def _find_conflicts(state: GraphState) -> list[tuple[str, list[str]]]:
    by_entity: dict[str, set[str]] = {}
    for source in state.context:
        entity = next(
            (
                known
                for known in KNOWN_ENTITIES
                if known.lower() in source.title.lower() or known.lower() in source.snippet.lower()
            ),
            None,
        )
        if entity is None:
            continue
        ranges = {match.group(0) for match in SALARY_RE.finditer(source.snippet)}
        if ranges:
            by_entity.setdefault(entity, set()).update(ranges)
    return [(entity, sorted(ranges)) for entity, ranges in by_entity.items() if len(ranges) > 1]


class AgentBrain:
    """Planning, tool execution, critique and synthesis logic.

    In mock mode this uses deterministic heuristics over the local corpus so the
    full graph runs offline. With an OpenAI-compatible endpoint configured it
    routes through the LLM and falls back to heuristics on API errors.
    """

    def __init__(self, settings: Settings, registry: ToolRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self.llm: OpenAICompatibleLLM | None = None
        if settings.llm_provider == "openai_compatible" and settings.llm_api_key:
            self.llm = OpenAICompatibleLLM(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
            )

    def _extract_entities(self, question: str) -> list[str]:
        found = [entity for entity in KNOWN_ENTITIES if entity.lower() in question.lower()]
        return found or [question.strip()[:80]]

    def plan(self, question: str) -> Plan:
        if self.llm is not None:
            try:
                return self._plan_with_llm(question)
            except Exception:
                logger.warning("LLM planning failed; falling back to heuristics", exc_info=True)
        entities = self._extract_entities(question)
        subtasks = [
            SubTask(
                id=f"ev-{index}",
                question=f"What are the AI internship roles, required skills and hiring signals at {entity}?",
                tools=["retrieve", "web_search"],
            )
            for index, entity in enumerate(entities, start=1)
        ]
        subtasks.append(
            SubTask(
                id="compare",
                question="Compare the AI internship opportunities across companies, including skills and hiring signals.",
                tools=["retrieve", "python_sandbox"],
            )
        )
        if _is_calculation_question(question):
            subtasks.append(
                SubTask(
                    id="calc",
                    question=question,
                    tools=["python_sandbox"],
                )
            )
        return Plan(
            objective=question,
            subtasks=subtasks,
            rationale="Split the research question by entity, then synthesize a comparison.",
            dependencies={subtask.id: [] for subtask in subtasks},
        )

    def _plan_with_llm(self, question: str) -> Plan:
        system = (
            "You are a research planner. Return JSON with keys objective, rationale, "
            "subtasks (array of id, question, tools). Tools: retrieve, web_search, arxiv_search, "
            "pdf_parse, python_sandbox, sqlite_query."
        )
        raw = self.llm.complete_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            tools=self.registry.schemas(),
        )
        subtasks = [
            SubTask(
                id=str(item.get("id", f"st-{index}")),
                question=str(item.get("question", "")),
                tools=[str(t) for t in item.get("tools", ["retrieve"])],
            )
            for index, item in enumerate(raw.get("subtasks", []))
        ]
        if not subtasks:
            raise ValueError("LLM returned an empty plan")
        return Plan(
            objective=str(raw.get("objective", question)),
            subtasks=subtasks,
            rationale=str(raw.get("rationale", "")),
            dependencies=raw.get("dependencies", {}),
        )

    def execute_subtask(self, subtask: SubTask) -> tuple[str, list[Source], ToolResult]:
        """Run one subtask and return (evidence_text, sources)."""
        started = time.perf_counter()
        if subtask.id == "calc":
            code = _calculation_code(subtask.question)
            if code is None:
                subtask.status = "failed"
                subtask.error = "no calculation pattern found"
                subtask.duration_ms = (time.perf_counter() - started) * 1000
                return "", [], ToolResult(tool="python_sandbox", ok=False, output="", error="no calculation pattern found")
            tool_result = self.registry.call("python_sandbox", code=code)
            subtask.duration_ms = (time.perf_counter() - started) * 1000
            if not tool_result.ok:
                subtask.status = "failed"
                subtask.error = tool_result.error
                return "", [], tool_result
            subtask.status = "done"
            subtask.result = f"Calculation result: {tool_result.output.strip()}"
            return subtask.result, [], tool_result

        entities = [entity for entity in KNOWN_ENTITIES if entity.lower() in subtask.question.lower()]
        retrieve_k = max(self.settings.retrieval_top_k, 20) if entities else self.settings.retrieval_top_k
        result = self.registry.call("retrieve", query=subtask.question, k=retrieve_k)
        if not result.ok or not result.data or not result.data.get("results"):
            subtask.status = "failed"
            subtask.error = result.error or "no evidence returned"
            subtask.duration_ms = (time.perf_counter() - started) * 1000
            return "", [], result

        results = result.data["results"]
        if entities:
            entity = entities[0]
            filtered = [
                item
                for item in results
                if entity.lower() in str(item.get("title", "")).lower()
                or entity.lower() in str(item.get("snippet", "")).lower()
            ]
            if not filtered:
                subtask.status = "failed"
                subtask.error = f"no evidence found for {entity}"
                subtask.duration_ms = (time.perf_counter() - started) * 1000
                return "", [], result
            results = filtered[: min(3, len(filtered))]

        results = _merge_by_doc(results)
        sources: list[Source] = []
        for item in results:
            source = Source(
                id=str(item["id"]),
                title=str(item.get("title", item.get("id", ""))),
                url=str(item.get("url", "")),
                snippet=str(item.get("snippet", "")),
                metadata={
                    "doc_id": item.get("id", ""),
                    "heading": item.get("heading", ""),
                    "score": item.get("score", 0.0),
                },
            )
            sources.append(source)

        evidence_lines = []
        for index, source in enumerate(sources, start=1):
            evidence_lines.append(f"- {source.title} [candidate-{index}]: {source.snippet[:220]}")
        subtask.status = "done"
        subtask.duration_ms = (time.perf_counter() - started) * 1000
        subtask.sources = sources
        subtask.result = "\n".join(evidence_lines)
        return subtask.result, sources, result

    def critique(self, state: GraphState) -> tuple[bool, str]:
        issues: list[str] = []
        if not state.plan or not state.plan.subtasks:
            return False, "plan is empty"
        for subtask in state.plan.subtasks:
            if subtask.status == "failed":
                if "no evidence" in subtask.error and "no direct evidence" in state.draft:
                    continue
                issues.append(f"subtask {subtask.id} failed: {subtask.error}")
            elif subtask.status != "done":
                issues.append(f"subtask {subtask.id} not executed")
            elif "python_sandbox" in subtask.tools and not subtask.result:
                issues.append(f"calculation subtask {subtask.id} has no result")
        evidence_subtasks = [
            subtask
            for subtask in state.plan.subtasks
            if "python_sandbox" not in subtask.tools
        ]
        if evidence_subtasks and not state.context:
            issues.append("no evidence collected")
        if not state.draft:
            issues.append("draft report is missing")
        elif state.context:
            ok, citation_issues = validate_citations(state.draft, state.context)
            if not ok:
                issues.extend(citation_issues[:5])
        return not issues, "; ".join(issues)

    def synthesize(self, state: GraphState) -> str:
        if self.llm is not None:
            try:
                return self._synthesize_with_llm(state)
            except Exception:
                logger.warning("LLM synthesis failed; falling back to templates", exc_info=True)
        lines: list[str] = []
        lines.append(f"# Research Report: {state.question}")
        lines.append("")
        lines.append("## Executive Summary")
        for subtask in state.plan.subtasks if state.plan else []:
            if not subtask.sources:
                if "python_sandbox" in subtask.tools and subtask.result:
                    lines.append(f"- {subtask.question}: {subtask.result}")
                else:
                    lines.append(f"- {subtask.question}: no direct evidence was found in the current corpus.")
                continue
            citations = "".join(f"[{self._citation_number(state, source.id)}]" for source in subtask.sources)
            lines.append(f"- {subtask.question} Evidence: {subtask.result.splitlines()[0][:180]} {citations}")
        lines.append("")
        lines.append("## Evidence")
        for subtask in state.plan.subtasks if state.plan else []:
            lines.append(f"### {subtask.id}: {subtask.question}")
            lines.append(subtask.result or "No evidence collected.")
            lines.append("")
        lines.append("## Comparison Notes")
        lines.append(
            "The retrieved sources are listed below. Claims should be cross-checked against "
            "the primary job pages before use."
        )
        lines.append("")
        conflicts = _find_conflicts(state)
        if conflicts:
            lines.append("## Conflicting Evidence")
            for entity, ranges in conflicts:
                lines.append(f"- {entity}: multiple sources report {', '.join(ranges)}.")
            lines.append("")
        lines.append("## Sources")
        for index, source in enumerate(state.context, start=1):
            title = source.title or source.url or source.id
            if source.url:
                lines.append(f"{index}. [{title}]({source.url})")
            else:
                lines.append(f"{index}. {title}")
        return "\n".join(lines)

    def _synthesize_with_llm(self, state: GraphState) -> str:
        context = "\n\n".join(
            f"[{index}] {source.title} ({source.url})\n{source.snippet}"
            for index, source in enumerate(state.context, start=1)
        )
        system = (
            "You are a research report writer. Write a structured Markdown report with an "
            "Executive Summary, Evidence, Comparison Notes, and Sources. Cite every source "
            "with bracket numbers like [1] and do not invent facts."
        )
        prompt = f"Question: {state.question}\n\nRetrieved evidence:\n{context}"
        return self.llm.complete([{"role": "system", "content": system}, {"role": "user", "content": prompt}]).content

    def _citation_number(self, state: GraphState, source_id: str) -> int:
        for index, source in enumerate(state.context, start=1):
            if source.id == source_id:
                return index
        return 0

    def sanitize(self, question: str) -> tuple[bool, str, list[str]]:
        return validate_input(redact_pii(question))

    def finalize(self, state: GraphState) -> GraphState:
        if not state.report:
            state.report = state.draft
        state.summary = {
            "question": state.question,
            "iterations": state.iterations,
            "passed_critique": state.passed,
            "subtasks": len(state.plan.subtasks) if state.plan else 0,
            "sources": len(state.context),
            "tool_calls": len(state.tool_log),
            "report_chars": len(state.report),
        }
        return state
