# Retrieval and Generation Evaluation Metrics

## Overview
This document explains the evaluation metrics used in code to assess Retrieval-Augmented Generation (RAG) performance. The evaluation is divided into two parts:

**Retrieval Metrics**: Measures how well the retrieved documents match the expected user/conversation or relevant text.

**Generation Metrics**: Measures the accuracy and semantic similarity of the generated response compared to the ground truth.

---

## **🔹 Retrieval Metrics**
These metrics assess whether the retrieved **context (chunks)** is relevant to the query.

### **1️⃣ Recall@k**
✔ **Measures**: The proportion of relevant documents retrieved within the top **k** results.  
✔ **Why it's useful**: Indicates whether the correct document is retrieved at all.  
✔ **Formula**:
```
Recall@k = (Number of relevant documents in top k) / (Total relevant documents)
```
✔ **Example**:
- Expected user: `"JohnDoe"`
- Retrieved documents: `["JohnDoe", "JaneDoe", "JohnDoe"]`
- **Recall@3 = 1.0** (since at least 1 relevant document was retrieved)

---

### **2️⃣ Mean Reciprocal Rank (MRR)**
✔ **Measures**: How high the **first relevant document** appears in the ranked list.  
✔ **Why it's useful**: Encourages retrieving relevant documents **early** in the ranking.  
✔ **Formula**:
```
MRR = 1 / (Rank of the first relevant document)
```
✔ **Example**:
- Expected user: `"JohnDoe"`
- Retrieved documents: `["Giulia", "Giulia", "JohnDoe"]`
- First relevant document is at **Rank 3** → `MRR = 1/3 = 0.33`

---

### **3️⃣ Content-Based Similarity (TO BE ADDED)** 
✔ **Measures**: Textual similarity between retrieved and expected content.  
✔ **Why it's useful**: Checks **semantic relevance** beyond metadata matching.  
✔ **Approaches**:
- **Cosine Similarity** (vector embeddings)
- **Jaccard Similarity** (word overlap)
- **TF-IDF Score** (word importance)

---

## **🔹 Generation Metrics**
These metrics measure **how close the generated response is** to the correct answer.

### **1️⃣ Exact Match (EM)**
✔ **Measures**: Whether the **generated response exactly matches** the expected answer.  
✔ **Why it's useful**: Strict correctness check.  
✔ **Formula**:
```
EM = 1 if (generated text == ground truth) else 0
```
✔ **Example**:
- Expected: `"The capital of France is Paris."`
- Generated: `"The capital of France is Paris."`
- ✅ **EM = 1.0**

---

### **2️⃣ BLEU Score (Bilingual Evaluation Understudy)**
✔ **Measures**: **n-gram overlap** between generated and reference text.  
✔ **Why it's useful**: Measures **word-level accuracy** but not meaning.  
✔ **Formula (Simplified)**:
```
BLEU = exp(sum(w_n * log P_n))
```
✔ **Example**:
- Expected: `"The dog is running in the park."`
- Generated: `"A dog runs in a park."`
- **BLEU Score ≈ 0.7** (some overlap, different phrasing)

---

### **3️⃣ ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)**
✔ **Measures**: **Recall-based similarity** for text generation.  
✔ **Why it's useful**: Evaluates **content coverage** rather than word-for-word matching.  
✔ **Variants**:
- **ROUGE-1** → **Unigram (word-level) overlap**  
- **ROUGE-2** → **Bigram (two-word) overlap**  
- **ROUGE-L** → **Longest common subsequence (LCS)**  

✔ **Example**:
- Expected: `"AI helps improve efficiency in business."`
- Generated: `"AI increases efficiency in companies."`
- **ROUGE-1** = High (similar words)
- **ROUGE-2** = Medium (some bigram overlap)
- **ROUGE-L** = High (similar structure)

---

### **4️⃣ BERTScore (Deep Semantic Similarity)**
✔ **Measures**: Uses **transformer embeddings** (e.g., BERT) to compare meaning.  
✔ **Why it's useful**: Captures meaning beyond word matching.  
✔ **Example**:
- Expected: `"The cat is sleeping on the couch."`
- Generated: `"A feline rests on the sofa."`
- **BLEU** = Low (word mismatch)
- **ROUGE** = Low (few matching words)
- **BERTScore** = **High** (same meaning)

---

### **5️⃣ F1-Score for Generative Evaluation**
✔ **Measures**: Balances **precision and recall** for responses.  
✔ **Why it's useful**: Evaluates both correctness **and** completeness.  
✔ **Formula**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
✔ **Example**:
- Expected: `"The Eiffel Tower is in Paris."`
- Generated: `"The famous Eiffel Tower is in France."`
- **Precision**: Medium (extra words)  
- **Recall**: High (includes key facts)  
- **F1-Score** = **Balanced Accuracy**

---

## **🔹 Summary Table**
| **Metric**      | **Type** | **What It Measures** | **Best For** |
|---------------|--------|----------------|------------|
| **Recall@k** | Retrieval | % of relevant documents in top-k | Ranking quality |
| **MRR** | Retrieval | Rank of first correct document | Early relevance |
| **Content Similarity** | Retrieval | Textual overlap between retrieved docs | Precision check |
| **Exact Match (EM)** | Generation | Is response **exactly** correct? | Strict correctness |
| **BLEU Score** | Generation | **n-gram similarity** | Word-level accuracy |
| **ROUGE Score** | Generation | **Content coverage** (recall) | Summarization |
| **BERTScore** | Generation | **Deep meaning similarity** | Semantic similarity |
| **F1-Score** | Generation | Balances **precision & recall** | Partial correctness |



