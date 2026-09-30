"""Prompt-engineering templates for agentic collection.

The persona template turns a one-line area of expertise into a full collection
manager persona grounded in intelligence-collection tradecraft (PIR -> EEI ->
SIR -> gap analysis). The planner/assessor/brief prompts then drive each stage
of the loop. All of these are sent through LiteLLM exactly like extractor.py.
"""

# Filled once when an agent is created; the result is stored as agents.persona_prompt.
PERSONA_TEMPLATE = """You are an expert research collection manager specializing in {expertise}.

You approach every research question the way an intelligence analyst runs a \
collection effort:
- Treat the user's question as a Priority Intelligence Requirement (PIR).
- Decompose it into Essential Elements of Information (EEIs): the distinct \
sub-questions that, taken together, fully answer the PIR.
- For each EEI, think in terms of Specific Information Requirements: the concrete \
things you would search the open web for to satisfy it.
- Judge coverage honestly. A requirement is only satisfied when retrieved \
sources actually address it; otherwise it is a gap to be re-tasked.

You are rigorous, source-driven, and you do not pad coverage with weakly \
relevant material. Your domain expertise in {expertise} shapes which facets \
matter and which sources are authoritative."""


def build_persona(expertise: str) -> str:
    return PERSONA_TEMPLATE.format(expertise=expertise.strip() or "general research")


# --- Planning ---

PLAN_INSTRUCTION = """Build a collection plan for this research question.

QUESTION: {question}

Decompose the question into 3-7 collection requirements (EEIs) that together \
fully cover it. For each requirement, propose 1-3 concrete web-search queries \
that would retrieve sources to satisfy it.

Respond with ONLY valid JSON in exactly this shape:
{{
  "requirements": [
    {{
      "title": "short label",
      "description": "what this requirement needs to establish",
      "rationale": "why it matters for the question",
      "queries": ["search query 1", "search query 2"]
    }}
  ]
}}"""


def build_plan_prompt(question: str) -> str:
    return PLAN_INSTRUCTION.format(question=question)


# --- Assessment / gap analysis ---

ASSESS_INSTRUCTION = """Assess whether the collected sources satisfy this \
collection requirement.

REQUIREMENT: {title}
WHAT IT NEEDS: {description}

COLLECTED SOURCES (title + excerpt):
{sources}
{search_note}
Decide if these sources, taken together, give a well-grounded answer to the \
requirement. To count as satisfied they must cover the requirement's specific \
substance — the facts, mechanisms, or evidence it asks for — not merely mention \
the topic. Exhaustive or academic-grade completeness is NOT required. If the \
specifics are missing or only superficially touched, mark it unsatisfied and \
propose 1-3 refined web-search queries aimed at the gap (do not repeat queries \
that clearly failed).

Respond with ONLY valid JSON in exactly this shape:
{{
  "satisfied": true,
  "confidence": "high|medium|low",
  "missing": "what is still missing, or empty string if satisfied",
  "next_queries": ["refined query 1"]
}}"""


def build_assess_prompt(title: str, description: str, sources_block: str,
                        search_note: str = "") -> str:
    """`search_note` is an optional one-line summary of what this pass's
    searches returned (e.g. "Search coverage this pass: 3 queries, 11
    results (brave)"), so the grader can tell thin coverage from a thin web."""
    return ASSESS_INSTRUCTION.format(
        title=title, description=description or "(no detail given)",
        sources=sources_block or "(no sources collected yet)",
        search_note=f"\n{search_note}\n" if search_note else "",
    )


# --- Brief synthesis ---

BRIEF_INSTRUCTION = """Write a concise research brief answering the question, \
grounded ONLY in the collected sources.

QUESTION: {question}

REQUIREMENT COVERAGE:
{coverage}

SOURCES (numbered; cite them as [n]):
{sources}
{delta}
Write the brief in Markdown with these sections:
## Summary
A 3-5 sentence answer to the question, citing its sources inline as [n].
## Key Findings
Bullet points of the most important findings. Cite supporting sources inline as \
[n]. Only state what the sources support.
## Coverage & Gaps
Note which requirements are well covered and explicitly call out any that are \
unmet or thin.

Do not invent facts or sources. Keep it tight and analytic."""


def build_brief_prompt(question: str, coverage_block: str, sources_block: str, delta_block: str = "") -> str:
    delta = f"\n{delta_block}\n" if delta_block else "\n"
    return BRIEF_INSTRUCTION.format(
        question=question, coverage=coverage_block,
        sources=sources_block, delta=delta,
    )
