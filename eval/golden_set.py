from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenQuestion:
    id: str
    question: str
    question_type: str  # single-hop | multi-hop | contradictory | no-answer | calculation
    expected_keywords: list[str] = field(default_factory=list)
    note: str = ""


GOLDEN_SET: list[GoldenQuestion] = [
    GoldenQuestion(
        id="sg-01",
        question="What AI skills does Shopee expect from interns in Singapore?",
        question_type="single-hop",
        expected_keywords=["Shopee", "Python", "PyTorch"],
        note="Covered by seed corpus.",
    ),
    GoldenQuestion(
        id="sg-02",
        question="What does TikTok's AI research internship cover?",
        question_type="single-hop",
        expected_keywords=["TikTok", "recommendation", "LLM"],
        note="Covered by seed corpus.",
    ),
    GoldenQuestion(
        id="sg-03",
        question="How long is a Grab AI internship and what do interns work on?",
        question_type="single-hop",
        expected_keywords=["Grab", "pricing", "fraud"],
        note="Covered by seed corpus.",
    ),
    GoldenQuestion(
        id="sg-04",
        question="What skills are useful for an NVIDIA AI internship in Singapore?",
        question_type="single-hop",
        expected_keywords=["NVIDIA", "CUDA", "C++"],
        note="Covered by seed corpus.",
    ),
    GoldenQuestion(
        id="sg-05",
        question="Compare Shopee, TikTok and Grab AI internship hiring signals.",
        question_type="multi-hop",
        expected_keywords=["Shopee", "TikTok", "Grab"],
        note="Requires cross-entity evidence.",
    ),
    GoldenQuestion(
        id="sg-06",
        question="Which Singapore public agencies hire AI researchers?",
        question_type="single-hop",
        expected_keywords=["GovTech", "A*STAR"],
        note="Covered by seed corpus.",
    ),
    GoldenQuestion(
        id="sg-07",
        question="What interview topics are common for Singapore AI internships in 2026?",
        question_type="single-hop",
        expected_keywords=["retrieval", "evaluation", "prompt"],
        note="Covered by trends document.",
    ),
    GoldenQuestion(
        id="sg-08",
        question="Do Shopee AI interns receive full-time return offers?",
        question_type="single-hop",
        expected_keywords=["Shopee", "return"],
        note="Covered by Shopee document.",
    ),
    GoldenQuestion(
        id="sg-09",
        question="What is the typical TikTok internship duration?",
        question_type="single-hop",
        expected_keywords=["TikTok", "10-12"],
        note="Covered by TikTok document.",
    ),
    GoldenQuestion(
        id="sg-10",
        question="Compare Shopee ML intern skills with Grab data science intern skills.",
        question_type="multi-hop",
        expected_keywords=["Shopee", "Grab"],
        note="Cross-entity comparison.",
    ),
    GoldenQuestion(
        id="sg-11",
        question="Which companies value deployed demos or GitHub projects?",
        question_type="single-hop",
        expected_keywords=["GitHub", "demo"],
        note="Covered by trends document.",
    ),
    GoldenQuestion(
        id="sg-12",
        question="What full-time roles can TikTok research interns move into?",
        question_type="single-hop",
        expected_keywords=["TikTok", "researcher"],
        note="Covered by TikTok document.",
    ),
    GoldenQuestion(
        id="sg-13",
        question="What does AI Singapore do for students?",
        question_type="single-hop",
        expected_keywords=["AI Singapore", "accelerator"],
        note="Covered by GovTech/A*STAR document.",
    ),
    GoldenQuestion(
        id="sg-14",
        question="How do Singapore employers evaluate AI intern candidates?",
        question_type="single-hop",
        expected_keywords=["GitHub", "paper", "competition"],
        note="Covered by trends document.",
    ),
    GoldenQuestion(
        id="sg-15",
        question="Compare NVIDIA and Shopee AI internship technical focus.",
        question_type="multi-hop",
        expected_keywords=["NVIDIA", "Shopee"],
        note="Cross-entity comparison.",
    ),
    GoldenQuestion(
        id="sg-16",
        question="What models do Singapore companies ask interns to understand?",
        question_type="single-hop",
        expected_keywords=["LLM", "transformer"],
        note="Covered by corpus.",
    ),
    GoldenQuestion(
        id="sg-17",
        question="What is the hiring process for Shopee AI interns?",
        question_type="single-hop",
        expected_keywords=["coding", "machine learning", "interview"],
        note="Covered by Shopee document.",
    ),
    GoldenQuestion(
        id="sg-18",
        question="Which Grab teams use generative AI?",
        question_type="single-hop",
        expected_keywords=["Grab", "generative"],
        note="Covered by Grab document.",
    ),
    GoldenQuestion(
        id="sg-19",
        question="Why do employers ask for hands-on AI work?",
        question_type="single-hop",
        expected_keywords=["hands-on", "deployed"],
        note="Covered by trends document.",
    ),
    GoldenQuestion(
        id="sg-20",
        question="Summarize 2026 AI internship trends in Singapore.",
        question_type="multi-hop",
        expected_keywords=["agentic", "responsible", "LLM"],
        note="Covered by trends document.",
    ),
]


def _build_generated_questions() -> list[GoldenQuestion]:
    """Deterministic expansion to 60 offline cases for the regression harness."""
    entity_specs = [
        {
            "entity": "Shopee",
            "teams": ["search", "recommendation"],
            "skills": ["Python", "PyTorch", "SQL"],
            "process": ["coding", "machine learning", "interview"],
            "duration": ["Shopee"],
            "after": ["full-time", "return"],
            "unique": ["logistics", "analytics"],
        },
        {
            "entity": "TikTok",
            "teams": ["recommendation", "multimodal"],
            "skills": ["Python", "deep learning", "LLM"],
            "process": ["mentor", "project"],
            "duration": ["10-12"],
            "after": ["researcher", "applied scientist"],
            "unique": ["speech", "alignment"],
        },
        {
            "entity": "Grab",
            "teams": ["pricing", "fraud"],
            "skills": ["Python", "SQL", "statistics"],
            "process": ["experiments", "feature engineering"],
            "duration": ["4-6"],
            "after": ["production", "user-facing"],
            "unique": ["transportation", "mapping"],
        },
        {
            "entity": "NVIDIA",
            "teams": ["CUDA", "inference"],
            "skills": ["C++", "Python", "CUDA"],
            "process": ["kernel", "profiling"],
            "duration": ["NVIDIA"],
            "after": ["deployment", "edge"],
            "unique": ["robotics", "quantization"],
        },
        {
            "entity": "GovTech",
            "teams": ["computer vision", "NLP"],
            "skills": ["fundamentals", "reproducibility"],
            "process": ["communicate", "non-experts"],
            "duration": ["GovTech"],
            "after": ["trustworthy", "research"],
            "unique": ["public", "AI roles"],
        },
        {
            "entity": "A*STAR",
            "teams": ["computer vision", "NLP"],
            "skills": ["fundamentals", "reproducibility"],
            "process": ["accelerator", "industry"],
            "duration": ["A*STAR"],
            "after": ["research", "internship"],
            "unique": ["Singapore", "institutes"],
        },
    ]

    questions: list[GoldenQuestion] = []
    index = 21
    for spec in entity_specs:
        entity = spec["entity"]
        templates = [
            ("single-hop", f"What AI teams do {entity} interns join?", spec["teams"]),
            ("single-hop", f"What skills matter most for {entity} AI interns?", spec["skills"]),
            ("single-hop", f"What is the hiring process for {entity} AI interns?", spec["process"]),
            ("single-hop", f"How long is the {entity} AI internship?", spec["duration"]),
            ("single-hop", f"What opportunities can {entity} AI interns expect after the program?", spec["after"]),
            ("single-hop", f"What makes the {entity} AI internship technically unique?", spec["unique"]),
        ]
        for question_type, question, keywords in templates:
            questions.append(
                GoldenQuestion(
                    id=f"sg-{index:02d}",
                    question=question,
                    question_type=question_type,
                    expected_keywords=[entity, *keywords],
                    note="Generated from seed corpus templates.",
                )
            )
            index += 1

    trend_questions = [
        (
            "single-hop",
            "What evidence do Singapore employers want from AI candidates?",
            ["GitHub", "demo"],
        ),
        (
            "single-hop",
            "What LLM-related topics appear in Singapore AI interviews?",
            ["retrieval", "prompt"],
        ),
        (
            "multi-hop",
            "What 2026 trends should AI internship candidates know?",
            ["agentic", "responsible"],
        ),
        (
            "single-hop",
            "How should AI candidates communicate their impact?",
            ["communication", "quantify"],
        ),
    ]
    for question_type, question, keywords in trend_questions:
        questions.append(
            GoldenQuestion(
                id=f"sg-{index:02d}",
                question=question,
                question_type=question_type,
                expected_keywords=keywords,
                note="Generated from trends document.",
            )
        )
        index += 1

    return questions


GENERATED_QUESTIONS = _build_generated_questions()


def _build_real_data_questions() -> list[GoldenQuestion]:
    companies = [
        "Tencent",
        "Huawei",
        "Alibaba",
        "Amazon",
        "Apple",
        "SAP",
        "PayPal",
        "Visa",
        "Salesforce",
        "DBS",
        "OCBC",
        "UOB",
        "JPMorgan",
        "GIC",
        "ST Engineering",
        "Singtel",
        "Razer",
        "Micron",
        "Cynapse",
        "ESGPedia",
    ]
    questions: list[GoldenQuestion] = []
    index = 61
    for company in companies:
        questions.append(
            GoldenQuestion(
                id=f"sg-{index:02d}",
                question=f"What AI or data internship roles does {company} offer in Singapore?",
                question_type="single-hop",
                expected_keywords=[company, "AI"],
                note="Imported from the real Singapore internship opportunity sheet.",
            )
        )
        index += 1
        questions.append(
            GoldenQuestion(
                id=f"sg-{index:02d}",
                question=f"What compensation range is listed for {company} internships in Singapore?",
                question_type="single-hop",
                expected_keywords=[company, "S$"],
                note="Imported from the real Singapore internship opportunity sheet.",
            )
        )
        index += 1
    return questions


REAL_DATA_QUESTIONS = _build_real_data_questions()

DIFFICULT_QUESTIONS = [
    GoldenQuestion(
        id="sg-101",
        question="If a Shopee intern works 8 hours per day, 5 days per week for 10 weeks, how many total hours do they work?",
        question_type="calculation",
        expected_keywords=["400"],
        note="Requires python_sandbox tool call.",
    ),
    GoldenQuestion(
        id="sg-102",
        question="What is the exact monthly stipend for OpenAI AI interns in Singapore?",
        question_type="no-answer",
        expected_keywords=["no", "evidence"],
        note="Should refuse with no-evidence instead of fabricating a number.",
    ),
    GoldenQuestion(
        id="sg-103",
        question="What monthly compensation ranges are reported for Shopee AI interns in Singapore?",
        question_type="contradictory",
        expected_keywords=["Shopee", "S$", "Conflicting"],
        note="Two sources disagree; report must flag the conflict.",
    ),
]
GOLDEN_SET_FULL = GOLDEN_SET + GENERATED_QUESTIONS + REAL_DATA_QUESTIONS + DIFFICULT_QUESTIONS


def get_golden_set(limit: int | None = None) -> list[GoldenQuestion]:
    questions = GOLDEN_SET_FULL
    if limit is not None:
        questions = questions[:limit]
    return questions
