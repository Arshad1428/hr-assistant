# Tuning log

1. Original prompt, k=3, threshold 1.1: many valid questions were refused. 
2. Threshold 1.1 -> 1.43: Q01 started answering, Q08 missed details and citation, Q02/Q03/Q05/Q07 still refused. 
3. General grounding prompt: Q03 and Q05 improved, Q02 and Q07 still refused, Q06 and Q08 incomplete. 
4. k=3 -> k=1: no real improvement, and printed chunks showed every needed fact was already in the top chunk. 
5. Evidence-first prompt: Q02 and Q07 stopped refusing, but no answer had a source filename. Score: 
6. Four required fields: worse, Q05 said 18 instead of 45, Q06 said no deadline past 30 days, Q11 refusal had extra fields so refusals were 2/3. 
7. Single-answer prompt with a (Source: <file>, <section>) template: Q02 and Q05 printed "<file>" and "[file]", Q11 refusal was reworded so refusals were 2/3.
8. Prompt with no citation on refusals plus a "related policy" line: refusals 3/3, Q02/Q03/Q05 more complete, but Q07 refused wrongly and Q06/Q08 still incomplete. 


## Findings

- Retrieval was not the problem, the facts were in the top chunk.
- Fixing over-refusal and fixing short answers pulled against each other in the prompt.
- The evaluator only accepts the exact refusal sentence, so a reworded refusal fails.
- Q11 (distance 0.931) is below several answerable questions, so no single threshold separates them.
- At 1.43, Q10 and Q12 are refused before the model call, but Q11 only by the model (2 of 3).
- The 1.43 threshold has a thin margin (Q08 1.416, Q12 1.441) and may not generalize.

