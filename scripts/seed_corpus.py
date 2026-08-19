from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "data" / "corpus" / "seed"

DOCUMENTS = [
    {
        "title": "Shopee AI Internship Opportunities in Singapore",
        "source_url": "https://careers.shopee.sg/example-ai-intern",
        "source_type": "job_page",
        "tags": ["shopee", "ai", "internship", "singapore"],
        "content": """
Shopee Singapore runs AI internship programs across machine learning, NLP, computer
vision and recommendation systems. Interns join teams working on search ranking,
chat support, logistics forecasting and seller analytics.

Common requirements for AI interns include Python, PyTorch or TensorFlow, SQL, and
a solid foundation in linear algebra and probability. Teams prefer candidates who
have completed at least one ML project, Kaggle competition or published paper.

The hiring process usually includes a coding assessment, a machine learning
interview and a system design or case round. Interns who perform well often receive
full-time return offers.
""",
    },
    {
        "title": "TikTok AI Research and Engineering Internships in Singapore",
        "source_url": "https://careers.tiktok.com/example-ai-research-intern",
        "source_type": "job_page",
        "tags": ["tiktok", "ai", "research", "internship", "singapore"],
        "content": """
TikTok's Singapore office hires AI research and engineering interns for LLM
alignment, recommendation systems, multimodal understanding and speech. Research
interns usually work with a mentor on a defined project and present results at the
end of the internship.

Preferred skills are Python, deep learning frameworks, strong mathematical
background, and familiarity with large language models or recommender systems.
Publications in top conferences are a plus but not mandatory.

TikTok internships run for 10-12 weeks and are often a pipeline into graduate
researcher or applied scientist roles.
""",
    },
    {
        "title": "Grab AI and Data Science Internships in Singapore",
        "source_url": "https://grab.careers/example-ai-intern",
        "source_type": "job_page",
        "tags": ["grab", "ai", "data-science", "internship", "singapore"],
        "content": """
Grab's AI teams in Singapore focus on transportation optimization, pricing,
fraud detection, mapping and generative AI for merchant tools. Data science and ML
engineering interns contribute to experiments, feature engineering and model
deployment in production-like settings.

Requirements include Python, SQL, statistics, and hands-on experience with ML
libraries. Product sense matters at Grab because models must balance accuracy with
operational constraints such as latency and fairness.

Internship durations are typically 4-6 months, and students are encouraged to work
with business teams to shape real user-facing features.
""",
    },
    {
        "title": "NVIDIA AI Internships in Singapore",
        "source_url": "https://nvidia.wd5.myworkdayjobs.com/example-ai-intern",
        "source_type": "job_page",
        "tags": ["nvidia", "ai", "gpu", "internship", "singapore"],
        "content": """
NVIDIA Singapore hires interns for GPU-accelerated AI, CUDA programming, LLM
inference optimization and robotics. Interns may work on kernels, profiling,
quantization or deployment pipelines for data center and edge platforms.

Useful skills are C/C++, Python, CUDA or Triton, and understanding of transformer
architectures and memory optimization. Experience with deep learning compilers is
valuable for inference roles.

The interview process focuses on systems programming, numerical algorithms and
hands-on GPU experiments.
""",
    },
    {
        "title": "Singapore GovTech and A*STAR AI Research Opportunities",
        "source_url": "https://www.tech.gov.sg/example-ai-roles",
        "source_type": "job_page",
        "tags": ["govtech", "astar", "ai", "research", "singapore"],
        "content": """
Singapore's public research ecosystem includes GovTech AI roles and A*STAR research
institutes working on computer vision, NLP and trustworthy AI. These positions
value strong fundamentals, reproducibility and the ability to communicate research
to non-experts.

AI Singapore also runs internship and accelerator programs that connect students
with industry projects. Candidates with LLM, multimodal or edge deployment
experience are increasingly in demand.
""",
    },
    {
        "title": "2026 Singapore AI Internship Hiring Trends",
        "source_url": "https://example.com/singapore-ai-hiring-2026",
        "source_type": "news",
        "tags": ["singapore", "ai", "hiring", "trends", "2026"],
        "content": """
In 2026, Singapore companies are hiring more AI interns for LLM applications,
agentic workflows and responsible AI. Employers increasingly ask for evidence of
hands-on work such as GitHub projects, papers, competitions or deployed demos.

Common interview topics include prompt design, retrieval augmented generation,
model evaluation, cost optimization and failure analysis. Strong communication and
the ability to quantify impact differentiate candidates.
""",
    },
    {
        "title": "Shopee AI Intern Stipend Survey 2026",
        "source_url": "https://example.com/shopee-stipend-survey",
        "source_type": "news",
        "tags": ["shopee", "compensation", "stipend"],
        "content": """
A community survey in 2026 reported that Shopee AI interns in Singapore can earn
around S$8,000-10,000 per month depending on team and prior experience. This is a
market estimate and should be verified against the official offer letter.

The same survey noted that published ranges vary by source and that candidates
should not rely on a single number.
""",
    },
]


def slugify(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return cleaned[:60]


def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    for index, doc in enumerate(DOCUMENTS, start=1):
        tags = ", ".join(doc["tags"])
        front_matter = (
            "---\n"
            f"title: {doc['title']}\n"
            f"source_url: {doc['source_url']}\n"
            f"source_type: {doc['source_type']}\n"
            "collected_at: 2026-08-19\n"
            f"tags: [{tags}]\n"
            "---\n"
        )
        path = SEED_DIR / f"{index:02d}_{slugify(doc['title'])}.md"
        path.write_text(front_matter + doc["content"].strip() + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
