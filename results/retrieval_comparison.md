# Retrieval comparison (k=5; cases: original=15, paraphrased=20)

| Test set | Chunk size (words) | Method | Recall@5 | MRR@5 | Context words / query | Latency (ms) |
|---|---|---|---|---|---|---|
| original | 100 | dense | 1.000 | 1.000 | 499 | 26 |
| original | 100 | hybrid | 1.000 | 1.000 | 500 | 27 |
| original | 200 | dense | 1.000 | 1.000 | 986 | 25 |
| original | 200 | hybrid | 1.000 | 0.967 | 987 | 26 |
| original | 400 | dense | 1.000 | 1.000 | 1914 | 26 |
| original | 400 | hybrid | 1.000 | 1.000 | 1961 | 26 |
| paraphrased | 100 | dense | 0.900 | 0.875 | 492 | 30 |
| paraphrased | 100 | hybrid | 0.950 | 0.917 | 502 | 32 |
| paraphrased | 200 | dense | 0.900 | 0.808 | 969 | 30 |
| paraphrased | 200 | hybrid | 0.950 | 0.900 | 990 | 31 |
| paraphrased | 400 | dense | 0.900 | 0.875 | 1945 | 30 |
| paraphrased | 400 | hybrid | 0.950 | 0.892 | 1929 | 30 |

Context tokens are roughly 1.3 x context words.
