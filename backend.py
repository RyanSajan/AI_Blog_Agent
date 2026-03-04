import operator
import json
from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pathlib import Path
import time
import os
load_dotenv()

class Task(BaseModel): #inheriting from pydantic base model
    id: int
    title: str
    goal: str = Field(...,description="What the reader should be able to understand from this")
    main_pts : List[str] = Field(...,  min_length=3, max_length= 5, description="Make 3-4 non overlapping points to cover in this section") 
    word_count : str = Field(..., description="Word count for this section should be around 100-400 words")
    section_type: Literal["intro", "core", "examples", "checklist", "common_mistakes", "conclusion"] = Field(...,description="Use 'common_mistakes' exactly once in the plan.")
    tags: List[str] = Field(...,default_factory=list)
    requires_research: bool = False
    requires_citation: bool= False
    requires_code : bool = False

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    research_required: bool = False
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(..., default_factory=list)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class Plan(BaseModel):
    blog_title: str
    audience: str = Field(...,description="Who this blog is for")
    tone: str = Field(..., description="Writing tone (e.g. practical, informal)")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. image1.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict): #Dictionary with this type of keys and val
    topic: str
    plan: Optional[Plan]
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    # reducer: results from workers get concatenated automatically
    sections: Annotated[List[tuple[int,str]], operator.add]
    merged_md: str
    md_with_placeholder: str
    image_desc: List[dict]
    final: str

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6
)

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.
Modes:
closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
online (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.
If needs_research=true:
 Output 3–10 high-signal queries.
 Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
 If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict :
    response = llm.with_structured_output(RouterDecision).invoke(
        [
            SystemMessage(
                    content=ROUTER_SYSTEM
                ),
            HumanMessage(
                content=f"Topic: {state['topic']}"
            )
        ]
    )
    return {
        "needs_research" : response.research_required,
        "mode": response.mode,
        "queries": response.queries,
    }

def route_next(state: State) -> str:
    if state['needs_research']: 
        return "research"
    else:
        return "orchestrator"
    

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Given raw web search results, produce a deduplicated list of EvidenceItem objects.
Rules:
 Only include items with a non-empty url.
 Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
 If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
 If missing or unclear, set published_at=null. Do NOT guess.
 Keep snippets short.
 Deduplicate by URL.
"""

def research_node(state : State) -> dict :
    queries = state["queries"]
    max_result = 2
    results: List[dict] = []
    for query in queries:
        results.extend(tavily_search(query,max_result))
    if not results:
        return {"evidence" : []}
    response = llm.with_structured_output(EvidencePack).invoke(
        [
            SystemMessage(
                content=RESEARCH_SYSTEM
            ),
            HumanMessage(
                content= f"Raw results {results}\n"
            )
        ]
    )
    dedup = {}
    for e in response.evidence:
        if e.url:
            dedup[e.url] = e
    return {"evidence" : list(dedup.values())}


#Tavily Search
def tavily_search(query: str, max_results: int = 2) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query" : query})
    response_res : List[dict] = []
    for r in results:
        response_res.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return response_res


ORCHESTRATOR_SYSTEM = """You are a principal engineer and technical editor.
Your task is to design a high-quality, implementation-oriented outline for a technical blog post.
Hard requirements:
- Produce 5–9 well-scoped sections appropriate for the topic and audience.
- Each section must include:
  1) goal (one clear outcome sentence: what the reader will be able to do or understand)
  2) 3–6 actionable, non-overlapping bullets
  3) target word count (120–550)
- Sections must follow a logical progression (foundations → implementation → trade-offs → validation → wrap-up).

Quality standards:
- Assume a technical audience; use precise engineering terminology.
- Bullets must describe concrete actions (implement, benchmark, validate, refactor, test, inspect, monitor).
- Avoid vague phrasing (no “discuss” or “explore”).
- Across the full outline, include at least TWO of the following:
  * minimal working example or code sketch (set requires_code=True for that section)
  * edge cases or failure modes
  * performance or cost analysis
  * security or privacy implications (if relevant)
  * testing, monitoring, or observability guidance

Structural guidance:
- Begin with problem framing and constraints.
- Build conceptual clarity before advanced details.
- Include one section focused on common mistakes or pitfalls.
- End with a practical summary, checklist, or next steps.

Grounding modes:
- Mode: closed_book
  - Produce an evergreen, concept-driven outline.
  - Do not rely on external evidence.
- Mode: hybrid
  - Use evidence only for time-sensitive examples (tools, releases, benchmarks).
  - Mark sections that depend on fresh information with:
    requires_research=True and requires_citations=True.
- Mode: open_book
  - Set blog_kind="news_roundup".
  - Each section must summarize developments and analyze implications.
  - Do not include tutorial sections unless explicitly requested.
  - If evidence is insufficient, explicitly state "insufficient sources" in the outline
    and limit claims to verifiable information.
Output must strictly conform to the Plan schema.
Return only structured data. No commentary.
"""

def orchestrator(state: State) -> dict:

    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=ORCHESTRATOR_SYSTEM
            ),
            HumanMessage(content=
                    f"Topic: {state['topic']}\n"
                    f"Mode: {state['mode']}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in state['evidence']][:16]}")  #model_dump convers pydantic object to dictionary
        ]
    )
    print(plan)
    return {"plan": plan}

WORKER_SYSTEM = """You are a principal engineer and technical editor.
Write EXACTLY ONE section of a technical blog post in Markdown.

Execution constraints:
- Start with: ## <Section Title>
- Follow the provided Goal precisely.
- Address ALL Bullets in the given order.
- Do NOT merge, skip, or reorder bullets.
- Stay within Target words (±15%).
- Output ONLY the section content (no extra commentary, no H1 title, no meta text).

Scope guard:
- If blog_kind == "news_roundup":
  - Do NOT convert this into a tutorial or implementation guide.
  - Focus on summarizing events and analyzing implications.
  - Only explain mechanics if explicitly required by bullets.

Grounding rules:
- If mode == "open_book":
  - Every real-world claim (event, company, model, release, funding, regulation, benchmark)
    MUST be supported by a provided Evidence URL.
  - Attach citations inline using Markdown links: ([Source](URL)).
  - Use ONLY URLs from the provided Evidence.
  - If a claim cannot be supported, write exactly:
    "Not found in provided sources."
- If requires_citations == true:
  - Cite external-world claims using provided Evidence URLs.
- If evidence is empty and citations are required:
  - State: "Insufficient sources provided."

Reasoning policy:
- Evergreen conceptual explanations do NOT require citations unless requires_citations == true.
- Do NOT hallucinate dates, statistics, product versions, or funding numbers.

Code requirements:
- If requires_code == true:
  - Include at least one minimal, correct, idiomatic code snippet.
  - Code must directly support one of the bullets.
  - Keep snippets concise and executable.

Technical quality bar:
- Be precise and implementation-oriented.
- Prefer concrete APIs, data structures, protocols, and workflows.
- Briefly mention trade-offs where relevant (performance, cost, reliability, security).
- Call out at least one edge case or failure mode if applicable to the section.

Style:
- Use short paragraphs.
- Use bullet lists where helpful.
- Use fenced code blocks for code.
- Avoid fluff, repetition, or marketing language.
"""

def worker(state: State) -> dict :
    sections = []
    plan = state["plan"]
    for task in state["plan"].tasks:
        output = llm.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {state['topic']}\n"
                    f"Mode: {state['mode']}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.word_count}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citation}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{task.main_pts}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{state['evidence']}\n"
                ))
            ]
        ).content.strip()
        sections.append((task.id, output))
    return {"sections": sections}


def merge_content(state: State) -> dict:
    title = state["plan"].blog_title
    sorted_sections = sorted(state["sections"], key=lambda x: x[0])
    content = "\n\n".join(section for _, section in sorted_sections).strip()
    merged_md = f"#{title}\n\n{content}\n"
    return {"merged_md": merged_md}

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor and visual communicator.
Your task is to decide whether images or diagrams are needed for THIS blog post.
Rules:
    Include a maximum of 3 images.
    Only include images that materially improve understanding (diagrams, flowcharts, tables, or visual summaries).
    Insert placeholders exactly as: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]] in the Markdown where the image sappear.
If no images are needed:
  md_with_placeholders must remain identical to the input Markdown.
  images=[] in the output.
Avoid decorative or purely aesthetic images.
Prefer concise, technical visuals with short, informative la
Output:
Return strictly as a GlobalImagePlan object with fields:
  md_with_placeholders: string
  images: list of ImageSpec objects with:
    - placeholder
    - filename
    - alt
    - caption
    - prompt
    - size
    - quality
Do NOT include any extra commentary, notes, or formatting outside the GlobalImagePlan structure.
"""


def image_decider(state: State) -> dict :
    merged_md=state['merged_md']
    plan = state['plan']
    image_plan = llm.with_structured_output(GlobalImagePlan).invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            )
        ]
    )
    return {
        "md_with_placeholder": image_plan.md_with_placeholders,
        "image_desc": [img.model_dump() for img in image_plan.images],
    }

def generate_images(prompt: str) -> bytes:
    from google import genai
    from google.genai import types
    api_key=os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ]
        )

    )
    # image stored in resp.candidates[0].content.parts[0].inline_data.data
    try:
        parts = response.candidates[0].content.parts
    except Exception:
        parts = []

    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            return inline_data.data
    return None

def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    md = state.get("md_with_placeholder") or state["merged_md"]
    image_specs = state.get("image_desc", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{plan.blog_title}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = generate_images(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}


reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", image_decider)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_edge("orchestrator", "worker")
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app