# Evaluator Instructions — PDF-Constrained Conversational Agent

## Overview

This document provides structured test cases for evaluating the PDF-Constrained Conversational Agent.
Upload the included **`sample.pdf`** ("Global Climate Change Report 2024") to the agent interface before running any of the queries below.

**Sample PDF:** `sample.pdf` — A 5-page technical report by the International Climate Research Consortium (ICRC), covering causes, impacts, and mitigation strategies for global climate change.

---

## How to Run the Agent

1. Launch the agent: `python app.py`
2. Open your browser at `http://127.0.0.1:7860`
3. Upload `sample.pdf` using the "Upload PDF" panel on the left
4. Wait for the ✅ confirmation message
5. Begin asking the queries below in the chat box

---

## ✅ Valid Queries (Agent Must Answer with Citation)

These queries are **fully answerable** from the PDF. The agent must produce an accurate response ending with a `[Source: Page X]` citation.

---

### Query 1
**Input:** `What is the current concentration of CO2 in the atmosphere?`

**Expected behavior:** The agent identifies the specific statistic from Page 1 of the document.

**Expected response (approximate):**
> The atmospheric CO2 concentration has reached **421 parts per million (ppm)**, which is the highest level in over 800,000 years. [Source: Page 1]

**Grading criteria:** Exact figure (421 ppm), correct citation.

---

### Query 2
**Input:** `What are the two main factors driving rising sea levels?`

**Expected behavior:** The agent retrieves the precise breakdown from Section 2.1 (Page 3).

**Expected response (approximate):**
> Rising sea levels are driven by two main factors: **thermal expansion** of warming ocean waters (accounting for 42% of the observed rise), and the **melting of land-based ice sheets and glaciers** (accounting for 58%). [Source: Page 3]

**Grading criteria:** Both factors named, percentages accurate, correct citation.

---

### Query 3
**Input:** `By how much has the cost of solar PV fallen over the last decade?`

**Expected behavior:** The agent retrieves the clean statistic from Section 3.1 (Page 4).

**Expected response (approximate):**
> The cost of solar photovoltaic (PV) energy has fallen by **89%** over the past decade, making it one of the cheapest sources of new electricity generation in most of the world. [Source: Page 4]

**Grading criteria:** Correct percentage (89%), correct citation.

---

### Query 4
**Input:** `How many countries have carbon pricing initiatives in place, and what percentage of global emissions do they cover?`

**Expected behavior:** The agent retrieves the multi-part statistic from Section 3.2 (Page 4).

**Expected response (approximate):**
> As of 2024, **73 carbon pricing initiatives** are in place globally, covering approximately **23% of global greenhouse gas emissions**. [Source: Page 4]

**Grading criteria:** Both figures correct (73 initiatives, 23%), correct citation.

---

### Query 5 (Follow-up / Conversational Memory Test)
*Ask Query 4 first, then immediately ask this follow-up:*

**Input:** `What are the two forms of carbon pricing mentioned?`

**Expected behavior:** The agent uses its conversational memory to understand this is a follow-up question about the carbon pricing topic, and retrieves the two specific forms from the same section.

**Expected response (approximate):**
> There are two main forms of carbon pricing: a **carbon tax**, which sets a direct price on emissions, and an **emissions trading system (ETS)** (also known as cap-and-trade), which sets a cap on total emissions and allows entities to buy and sell allowances. [Source: Page 4]

**Grading criteria:** Both forms named correctly, demonstrates memory of prior conversational context.

### Query 6 (Agentic Tool-Use)
**Input:** `Exactly how many times does the word 'climate' appear in this document?`

**Expected behavior:** The agent invokes the `count_word` Python tool. The UI briefly shows `🔧 Running tool: count_word...` before the LLM returns the exact mathematical count.

**Grading criteria:** Exact word count derived from Python execution, not hallucinated by token prediction.

---

## ❌ Invalid / Out-of-Scope Queries (Agent Must Refuse)

These queries **cannot be answered** from the PDF. The agent must explicitly refuse and not hallucinate an answer.

---

### Invalid Query 1
**Input:** `Who is the current President of the United States?`

**Expected behavior:** The agent refuses, as this is a general knowledge question with no relation to the document's content.

**Expected response (approximate):**
> I cannot answer that based on the provided document.

**Anti-hallucination check:** The agent must NOT name any politician or make any inference. A failure here constitutes a critical hallucination.

---

### Invalid Query 2
**Input:** `What is the GDP of India in 2024?`

**Expected behavior:** The agent refuses. The document discusses climate statistics, not economic indicators for individual countries.

**Expected response (approximate):**
> I cannot answer that based on the provided document.

**Anti-hallucination check:** The agent must not confuse climate-related economic data (carbon pricing coverage) with national GDP figures.

---

### Invalid Query 3
**Input:** `Write me a Python script to calculate carbon emissions.`

**Expected behavior:** The agent refuses to generate code. This is a request for creative/technical generation, not an information retrieval task from the document.

**Expected response (approximate):**
> I cannot answer that based on the provided document.

**Anti-hallucination check:** The agent must NOT produce any code. This tests whether the LLM can resist instruction-following requests that bypass its grounding constraints.

---

## 🌐 Bonus: Multilingual Test Queries

The agent's primary embedding model (Gemini) and reranker (BAAI/bge-reranker-m3) are multilingual. These queries test cross-lingual retrieval — asking a question in a different language about content that exists in both English and the respective language on Page 5.

---

### Multilingual Query 1 (Hindi)
**Input:** `CO2 की वायुमंडलीय सांद्रता कितनी है?`
*(Translation: "What is the atmospheric concentration of CO2?")*

**Expected behavior:** The agent retrieves the relevant chunk (Page 1 has the figure in English; Page 5 has a Hindi summary), and responds with the correct figure.

**Expected response (approximate):**
> वायुमंडल में CO2 की सांद्रता **421 ppm** तक पहुँच गई है। [Source: Page 1]
> *(or an equivalent English response with the correct data)*

---

### Multilingual Query 2 (French)
**Input:** `Quel est l'objectif de l'Accord de Paris concernant la température?`
*(Translation: "What is the Paris Agreement's goal regarding temperature?")*

**Expected behavior:** The agent retrieves the relevant information from Section 4.1 / Page 5 and responds correctly.

**Expected response (approximate):**
> L'Accord de Paris vise à limiter le réchauffement climatique à **bien en dessous de 2 degrés Celsius** et à poursuivre des efforts pour le limiter à **1,5 degré Celsius**. [Source: Page 5]
> *(or an accurate English translation of the same)*

---

## Evaluation Rubric Summary

| Criterion | Test Cases | What to Look For |
|---|---|---|
| **Accuracy** | Valid Q1–Q4 | Exact figures match the PDF |
| **Conversational Memory** | Valid Q5 | Correct follow-up without re-specifying topic |
| **Hallucination Robustness** | Invalid Q1–Q3 | Clean refusal, no fabricated data |
| **Citation Quality** | All valid queries | Every answer ends with `[Source: Page X]` |
| **Multilingual Grounding** | Bonus Q1–Q2 | Correct answer from non-English query |
