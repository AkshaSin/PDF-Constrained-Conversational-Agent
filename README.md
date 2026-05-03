---
title: PDF RAG Agent
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# PDF-Constrained Conversational Agent

This is a production-grade Conversational AI agent designed to answer user queries strictly based on a provided PDF document.

## Features
- **Strict Grounding:** The agent will only answer questions if the answer exists in the uploaded PDF.
- **Anti-Hallucination:** It explicitly refuses to answer out-of-scope queries.
- **Citations:** Generates precise `[Source: Page X]` citations for every claim.
- **Recursive Memory:** Handles long conversations without losing context.
- **Multilingual Support:** Capable of cross-lingual RAG using state-of-the-art embedding models.

## Usage
Upload a PDF on the left panel, wait for the index to build, and start chatting!
