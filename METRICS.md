# Evaluation Metrics Structure (Text Track)

Our evaluation script will measure and populate the comparison table using the following metrics across Levels 0 to 3:

1. **Retrieval (RAG - Level 1):**
   - **Recall@k:** Measures if the relevant documents are retrieved in the top $k$ results.
   - **MRR (Mean Reciprocal Rank):** Evaluates the ranking quality of the retrieval system.

2. **Response Generation (Levels 0, 1, 2, 3):**
   - **Exactitude:** Measures if the response is correct based on rules or an LLM judge (evaluated on 20 annotated test cases).
   - **Fidélité aux sources:** Checks whether all factual claims properly cite the provided corpus documents without hallucination.
   - **Abstention rate:** Measures correct refusal when information is absent.

3. **Agent Performance (Level 2 & 3):**
   - **Task Success:** Verification of the final state via code checks.
   - **pass^3:** Success rate over 10 tasks where all 3 execution attempts must succeed.

4. **Cost & Latency:**
   - **Tokens / Request:** Total input and output tokens consumed per query.
   - **Latency:** Execution time in seconds from request to final answer.
