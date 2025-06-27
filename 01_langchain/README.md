# Langchain Basics

This section will guide you through the core concepts of **Langchain** and help you build applications that can integrate language models effectively. Whether you're a beginner or an experienced developer, this section is designed to help you understand how to use Langchain in various real-world scenarios.

## Overview
- **LangChain framework**: simplifies building apps that leverage large language models (LLMs).
- **Use-cases**: chatbots, text generation, complex conversational flows, and AI-driven utilities.
- **Course goal**: guide you from zero to production-ready LangChain projects, no matter your experience level.

---

## 1. ChatModel
- **Purpose**: handles conversation objects (messages) instead of raw strings.  
- **Message types**  
  - **SystemMessage** – sets the role / behavior.  
  - **HumanMessage**  – user input.  
  - **AIMessage**     – model reply.  
- **Multi-turn context**: remembers dialogue history.  
- **Providers supported**: OpenAI, Google Gemini, open-source LLMs, etc.  
- **`invoke()`**: single call that runs the prompt stack and returns the model’s response.

---

## 2. Prompt Template
- **Reusable pattern**: fixed text with placeholders for runtime variables.  
- **Dynamic fields**: `PromptTemplate(input_variables=[...], template=...)`.  
- **Consistency**: enforces uniform prompt structure across calls.  
- **Typical uses**: content generation, translation, support replies, code scaffolding.

---

## 3. Chains
- **Definition**: ordered (or branched/parallel) workflows that pass data from one step to the next.  
- **Variants**  
  - **Sequential / Extended** – linear pipeline.  
  - **Parallel** – steps run concurrently, results merged.  
  - **Branching** – conditional paths for flexible logic.  
- **Benefit**: lets you compose simple LLM calls into complex, end-to-end systems.

---

## 4. Retrieval-Augmented Generation (RAG)
- **Goal**: ground LLM outputs in external, up-to-date knowledge.  
- **Pipeline**  
  1. **Chunk** source docs (PDFs, web pages, code, video transcripts).  
  2. **Embed** chunks → vectors; store in a **vector database**.  
  3. **Retrieve** the most relevant vectors for each user query.  
  4. **Augment** the LLM prompt with those chunks for accurate, context-aware answers.  
- **Benefits**: reduces hallucinations, scales to large corpora, lowers token cost.

![High-Level Overview](https://myapplication-logos.s3.ap-south-1.amazonaws.com/HighLevel+Overveiw+RAG.jpg)

---

## 5. Agents & Tools
- **Agents**: LLMs that follow a state-machine loop—*Action → Observation → Thought → Result*—driven by a prompt.  
- **Tools**: external functions an agent can call (search web, run code, query DB).  
- **Workflow**: agent decides which tool to use, executes it, observes output, thinks, and continues until the task is done.  
- **Power**: unlocks dynamic, goal-oriented behavior far beyond static prompting.

---