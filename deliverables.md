Loom video
https://www.loom.com/share/8b101f55a6e3472388dd7bbbd052f8cb?sid=b1d44892-7db3-4150-8178-73a88afb297d
https://www.loom.com/share/475b30deb292475a9be8289781819a2a?sid=ed167750-b462-45b8-ab53-e43bb286765a


Task 1 - Problem and audience

Ideal client profile

Small and medium-sized CEO/business owner
We have 20+ real clients on waitlist.


Problem

Many SMEs do not have instant financial insights that will help them make strategic decisions.  They decide based on gut feel foregoing the benefits that data-backed insights can give them.


Task 2 - Solution

Solution

Roni AI is an AI CFO for SMEs that gives instant financial insights that can help SME owners and enterprises make strategic business decisions.

Just imagine a world where every business owner and business leader has access to Top 3% Finance and Accounting Experts in the world.
Gut feel + top finance insights = real business power

It will be user-friendly to users of all backgrounds even without a finance background, and can give the data and the analysis they want when they need.

Potential questions:
Can I afford to hire a social media person next month for the coming 3 months?
Did my business do well last month?
Are my costs high or low compared to industry peers?

1 — Proposed solution (≤ 3 sentences)
A chat box with financial insights sits in the finance portal: the business owners asks “How was my performance in May 2025?” and instantly sees a cited explanation plus a drill-down table. The system pulls numbers from Postgres, retrieves policy or commentary snippets from a tiny vector store, and the LLM stitches them into plain-English answers.

2 — Stack & tooling (≤ 3 sentences)
Stack: 
LLM - GPT-4.1
Embedding model - text-embedding-3-small (low-cost vectors)
Orchestration - LangChain/LangGraph (easy orchestration)
Qdrant Lite - free vector DB
RAGAS - quality metrics
User interface - React + shadcn/ui (polished UI)
Serving and inference - FastAPI on Railway (zero-cost deploy)

Each pick offers a free tier or rock-bottom pricing and has good docs, letting a small team ship fast without DevOps overhead.


3 — Agents & agentic reasoning (≤ 3 sentences)
A Router Agent decides if a query needs RAG or search.  The router inspects the user’s intent and content requirements.  If the question can be answered with fresh web facts it forwards to a search tool, but if it needs domain-specific knowledge, it triggers the RAG pipeline to retrieve and ground the LLM in your private corpus. This keeps responses both cost-efficient and contextually accurate by only invoking retrieval when necessary.


Task 3 - Data

Data sources and external APIs
Insights from the team - can be from a chat or recordings of meetings
CFO insights - this will be the basis for the financial policies
Tavily API - real-time search for financial statements of industry peers or accounting and tax questions

Default chunking strategy
We cut every document into blocks of 750 GPT-4o tokens, 10% or 75-token overlap, so each piece is small enough for the model to handle easily.  This keeps chunks consistent for fast search without repeating content.  The overlap reduces hallucinations caused by missing headers or footnotes.

Other specific data
We need the historical and forecast financials that can be tied up to the corpus.


Task 4 - End-to-end agentic rag


Task 5 - Golden dataset using ragas

The RAG pipeline is strong on retrieval and relevance but weak on fact grounding:
Context recall 0.95 on answer relevancy 0.96 - the retriever consistently surfaces passages that do contain the answer, and the LLM stays on-topic.
Faithfulness 0.88 – most answers rely on the provided context rather than hallucinating, but the score isn’t perfect, so some slips occur
Factual correctness 0.57 & entity recall 0.45 – over 40 % of the time the answer still contains wrong numbers or omits key entities, showing that the model either misreads tables or loses precision when it rewrites
Noise-sensitivity 0.40 – the pipeline degrades noticeably when extra, irrelevant text is added, meaning reranking or chunk-selection isn’t filtering noise robustly

Retrieval quality is solid, but we would need tighter grounding, e.g. smaller/overlapping chunks for numeric tables, a stronger reranker, or a “cite-and-verify” post-processor—to lift factual correctness and entity recall without sacrificing the already high relevance scores.


Task 6 - Advanced retrieval

Tried all the advanced retrieval techniques:

BM25 keyword search - complements vectors by catching exact words that dense embeddings sometimes miss
Contextual compression retriever - re-ranks and trims the top-k hits so the LLM sees only the most salient rows or paragraphs, reducing token cost and hallucinations
Multi-query expansion - generates several paraphrased questions to widen recall, ideal when SMEs phrase the same variance inquiry in many ways
Parent-document retrieval - pulls the full source file section once a sub-chunk matches, preserving crucial table headers and footnotes for accurate numeric answers
Ensemble (BM25 ∪ dense): adds scores from both keyword and semantic search, giving robust results across structured ledgers and narrative board memos
Semantic retriever with smart chunking - uses coherence-based splits so each embedding captures a complete financial concept, boosting relevance without needing overlaps


Task 7 - Performance assessment

Key improvements over the original naive setup:

Accuracy gains - the fine-tuned embeddings + Contextual Compression or Semantic Retriever lift factual correctness by 6 – 6.5 pp while keeping answer relevance intact.
Entity coverage - Semantic Retriever shows the biggest jump in context-entity recall (+16 pp), meaning more numeric or named entities are correctly carried into answers
Robustness to noise - both Contextual Compression and Parent-Document nearly halve noise-sensitivity, indicating stronger filtering of irrelevant chunks.
Trade-offs - Multi-query raises entity recall but hurts faithfulness and noise-sensitivity; the Ensemble variant narrows recall for entities, suggesting score fusion needs retuning.

Overall, swapping the baseline for fine-tuned embeddings plus contextual-compression retrieval delivers the best all-round boost—higher factual grounding and lower hallucination risk at virtually no cost to latency or relevance.

Contextual compression tops or ties every key metric that matters—highest factual-correctness 
(0.492 vs ≤ 0.455 for others), highest faithfulness except the semantic variant, and the lowest noise-sensitivity (0.346)—while keeping answer-relevancy and entity recall solid, so it improves accuracy and robustness simultaneously without the trade-offs seen in the other strategies.

Would like to make the app end-to-end from ingestion of excel files to making computations, and then analysis.  This will be incorporated into RAG.

I would also like to automate nightly RAGAS regression in LangSmith; alert on > 3 pp drop in factual-correctness.