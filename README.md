MiniRAG Lab — 2 hours
Build a grounded HR Q&A assistant over three policy documents. By the end it answers questions with a citation, and refuses when the answer isn't in the documents.
Everything is scaffolded. You fill in five TODOs — and those five are the whole point of the lab.
Before the clock starts (do this the day before, with internet)
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
 
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b
 
# Then the check that actually matters:
python preflight.py
pip install succeeding does not mean you are ready. The embedding model (~90 MB) downloads on first use, not during installation — so the first person to discover it is usually sitting in the training room on shared wifi. preflight.py pulls it, caches it, and proves the whole stack works end to end.
Running the lab without internet
Yes, it runs fully offline — after that one-time setup. Nothing in the lab needs a network once the two models are cached locally.
| Component | Needs internet |
| --- | --- |
| pip install | Once, at setup |
| ollama pull llama3.2:3b | Once, at setup (~2 GB) |
| all-MiniLM-L6-v2 embedding model | Once, on first use — not at install |
| Ollama inference | Never |
| ChromaDB, text splitter, your code | Never |

Prove it rather than assume it. Turn wifi off and run:
python preflight.py --offline
Two environment variables make the offline run clean. The scripts set the second for you; set the first in the room:
export HF_HUB_OFFLINE=1           # skip the model version check
export ANONYMIZED_TELEMETRY=False # ChromaDB's telemetry call
Without them nothing breaks — but on a blocked network both calls wait for a timeout instead of failing fast, which reads as the lab hanging on startup.
For an air-gapped room, copy these two caches to each machine instead of having twenty people download the same models over the same wifi:
~/.ollama/models/
~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
The five decisions
| TODO | Decision | Where |
| --- | --- | --- |
| 1 | chunk_size and chunk_overlap | build_index() |
| 2 | top_k | retrieve() |
| 3 | The grounding prompt | PROMPT |
| 4 | The relevance threshold | answer() |
| 5 | The refusal short-circuit | answer() |

Nothing else needs editing. If you find yourself rewriting the loader or the CLI, you've drifted.
Timetable
| Time | Stage | Done when |
| --- | --- | --- |
| 0:00–0:10 | Read the three policies in corpus/. Actually read them. | You know roughly where each answer lives |
| 0:10–0:30 | TODO 1 — split and index | python lab.py --reset prints a chunk count that looks sane |
| 0:30–0:55 | TODO 2 — retrieve, then inspect | --ask "..." --verbose shows plausible chunks for Q01 and Q02 |
| 0:55–1:20 | TODO 3 — the grounding prompt | Q01 answers correctly with a citation |
| 1:20–1:40 | TODO 4 and 5 — threshold and short-circuit | Q10, Q11, Q12 are refused |
| 1:40–2:00 | --evaluate, record both scores, write the tuning log | You have two numbers and three log entries |

Running behind at 1:20? Skip tuning and do TODO 4 and 5 anyway. A system that scores 6/9 and refuses correctly has met the objective. One that scores 9/9 and invents HR policy has not.
Running it
python lab.py --reset                               # build the index
python lab.py --ask "How many casual leaves?" -v    # one question, show chunks
python lab.py --evaluate                            # all 12 test questions
python lab.py --evaluate --verbose                  # ...with distances
Rebuild with --reset after changing chunk size. Forgetting to is the most common wasted ten minutes in this lab — your new setting isn't in the index until you do.
The test set
12 questions in test_questions.csv: 9 answerable, 3 not. The tests column says what each one is probing. Four are deliberate traps:
Q02 — "days off for my sister's wedding". The policy says "personal exigencies". No shared words. If this fails, your embeddings aren't earning their place.
Q05 — asks the accumulation cap. The numbers 18, 15, 45 and 10 all sit in that same section.
Q08 — car mileage is Rs 14/km. Rs 6/km (two-wheeler) is the bullet directly above it.
Q10–Q12 — not in the corpus at all. Study leave is the nastiest: it sits beside four real leave types.
Score the two groups separately and never average them. 9/9 answerable with 0/3 refusals is a worse system than 7/9 with 3/3 — the first invents company policy for employees, confidently.
Choosing the threshold honestly
Chroma returns squared L2 distance: lower is more similar. Getting the comparison backwards makes it refuse everything or nothing, and both look like the threshold "not working".
Don't guess the value:
1. Run python lab.py --evaluate --verbose
2. Note the best distance for Q01 (clearly answerable) and Q12 (clearly not)
3. Put the threshold in the gap between them
If the two groups overlap, the threshold isn't your problem — your chunking is, so go back to TODO 1.
The tuning log
Three entries minimum, in tuning-log.md. One line each:
chunk_size 1000 -> 600 : Q06 went from partial to correct; Q02 unchanged
top_k 2 -> 4          : Q06 now complete; Q05 got worse, picked up 18 instead of 45
threshold 2.0 -> 1.15 : refusals 0/3 -> 3/3; no answerable question lost
Include the change that made things worse. Everyone has one, and a log without it has been written backwards from the result.
Done when
☐  Q01 answers "12 days" and cites leave-policy.md, Section 2
☐  Q02 retrieves Casual Leave despite sharing no words with it
☐  Q08 says Rs 14/km, not Rs 6
☐  All three of Q10, Q11, Q12 are refused
☐  The refusal happens before the model is called — prove it with --verbose
☐  Both scores recorded, plus three tuning-log entries
What you've actually built
The four components every RAG system has — ingestion, retrieval, grounded generation, and a refusal path — plus the one habit that matters most: measuring before tuning.
The full-size version adds hybrid search, re-ranking, agents, cloud models and guardrails. All of them attach to these same four parts. Nothing here is a toy version that gets thrown away.
Corpus and questions are synthetic. Northwind Technologies Private Limited is a fictional company; every figure and address is invented.
