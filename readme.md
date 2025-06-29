# Generative AI Course Repository

Generative AI is the branch of artificial intelligence concerned with producing novel content—text, images, audio, video, code, even molecular structures—rather than merely categorizing or retrieving existing data. At its core sit large-scale generative models such as large language models (LLMs) and large multimodal models; these networks contain billions of parameters and are pretrained on vast corpora that capture the statistical patterns of language, vision, and other modalities. When a user supplies a prompt—a natural-language instruction or question—the model reacts by predicting the most probable next token (or pixel or audio sample) again and again until a coherent new sequence emerges. This prompt-driven, “reactive” loop is the essence of generative AI: it does not plan, reason about external state, or call tools on its own; it simply synthesizes fresh content in response to the instructions it receives.

Because the same underlying mechanism can handle different data types, contemporary systems are increasingly multimodal, allowing a single model to produce paragraphs, illustrations, and soundtracks from a shared semantic space. Developers harness this capability through frameworks such as LangChain, LangGraph, Llama‑Index, or the native SDKs for providers like OpenAI and Groq, which wrap model calls, manage prompts, interface with vector databases, and simplify deployment. A basic generative‑AI application, therefore, consists of an LLM (or image/audio generator), a well‑crafted prompt, and any surrounding glue code required to feed input and display output; the model’s job is solely to generate. More sophisticated behaviours—deciding which external API to query, orchestrating multi‑step workflows, or collaborating with other agents—are not part of generative AI itself; they belong to the higher‑level paradigms of AI agents and agentic AI, which layer planning, tool use, and inter‑agent communication on top of the content‑generation engine.

---

## [LangChain](https://github.com/MuhammadAhsaanAbbasi/generative-ai/tree/main/01_langchain)

**LangChain** is an open‑source Python framework that streamlines the creation of applications powered by large language models. It offers a high‑level interface for **prompt management**, **chaining multiple model calls**, **connecting to external data sources**, and **orchestrating tool usage** such as web search or code execution. By abstracting away repetitive boilerplate API calls, token counting, retry logic, and rate‑limiting, LangChain lets developers focus on product logic while still retaining low‑level control when needed.

### Key Pillars

- **Models** – Wrappers for proprietary (OpenAI, Anthropic) and open-source (Llama, Mistral) LLMs.  
- **Prompts & Prompt Templates** – Reusable, parameterised instructions that keep your prompt engineering organized.  
- **Chains** – Composable workflows that pass the output of one step into the next, enabling multi-step reasoning or data pipelines.  
- **Retrievers & Vector Stores** – Plug-ins for Retrieval-Augmented Generation (RAG) that ground model output in private or real-time knowledge.  
- **Agents & Tools** – Higher-order abstractions allowing an LLM to decide which actions to take (e.g., run a search query, call an API) in order to meet a user goal.  

Together, these components transform a raw LLM into a fully-fledged, context-aware application with minimal code.

---

## Prompt Engineering 👩‍🔧📝

Prompt engineering is the craft of **getting more signal than noise out of a pre-trained LLM** purely by changing *what you say to it*—never its weights, never its training data. Think of it as UX design for language models: the prompt is the interface layer between human intent and model behaviour.

---

### 1. Why it matters

| Advantage               | What it buys you                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **Zero infrastructure** | A tweak to a text string costs pennies and ships in minutes.                         |
| **Rapid iteration**     | Non-technical teammates can experiment directly in a playground or prompt file.      |
| **Safe exploration**    | You can probe the model’s existing knowledge before investing in RAG or fine-tuning. |

> *Key takeaway*: Prompt tuning is your cheapest experiment, and often “good enough” for prototypes or content work.

---

### 2. How it actually works

Prompt engineering is less one trick, more a toolbox. Below are the most common components you mix and match.

| Component                | Purpose               | Micro-example                                       |
| ------------------------ | --------------------- | --------------------------------------------------- |
| **Role**                 | Sets the persona      | “You are a veteran HR advisor…”                     |
| **Task**                 | Specifies the goal    | “…explain our leave policy to an intern.”           |
| **Format**               | Controls output shape | “Answer in three short bullet points.”              |
| **Tone/Style**           | Adjusts voice         | “Keep the language friendly and jargon-free.”       |
| **Few-shot examples**    | Teaches by showing    | Q-A pairs that illustrate the desired answer style. |
| **Chain-of-thought cue** | Coaxes reasoning      | “Think step-by-step before you reply.”              |

Under the hood the model simply sees a longer input string, but those extra cues nudge its next-token probabilities into a narrower, more useful band.

---

### 3. Popular prompting patterns

1. **Zero-shot role prompting**

   > *Prompt*: “You are a sarcastic movie critic. Rate *The Matrix* in two sentences.”
   > **Good for**: quick voice shifts.

2. **Few-shot in-context learning**

   > Provide two Q-A pairs, then a third question.
   > **Good for**: structured data extraction, code translation, style mimicry.

3. **Chain-of-thought or “let’s think”**

   > *Prompt*: “First, outline your reasoning step-by-step, then give the final answer.”
   > **Good for**: maths, multi-hop reasoning, complex instructions.

4. **Self-consistency / “debate with yourself”**

   > Ask the model to produce multiple answers, then pick the majority or best-scored one.
   > **Good for**: reducing random errors in reasoning tasks.

---

### 4. Concrete example (putting the pieces together)

**Plain request (baseline)**

> “Explain leave policy.”

**Engineered prompt**

```
You are a helpful HR chatbot for ACME Inc.
► Task: Explain ACME's leave policy to a new intern.
► Constraints:
  • Use simple language suitable for a 19-year-old.
  • Bullet-point the key allowances.
  • End with: “Let me know if you have any other questions!”
Context: {insert policy excerpt here}
```

**Outcome**

* Clear persona and audience
* Structured, easy-to-scan format
* Friendly sign-off
* (If you later plug this into a RAG system, the `{insert policy excerpt}` slot becomes your dynamic context.)

---

### 5. Best-practice checklist

| ✅ Do                                                   | ❌ Avoid                                        |
| ------------------------------------------------------ | ---------------------------------------------- |
| Start with *one* well-defined task per prompt.         | Bundling multiple unrelated asks.              |
| State output format explicitly (JSON, table, bullets). | Hoping the model “guesses” your layout.        |
| Provide examples when precision matters.               | Excessively long context that buries the goal. |
| Iterate, test, and version-control your prompts.       | Treating the first decent answer as final.     |
| Use temperature = 0–0.3 for deterministic outputs.     | High randomness for financial or legal copy.   |

---

### 6. Pain points to watch

* **Brittleness** – a single extra adjective can change the answer.
* **Token limits** – long prompts reduce room for the model’s reply.
* **Knowledge cutoff** – no prompt can summon facts the model never saw; that’s where RAG or fine-tuning step in.

---

### 7. When to stop tweaking and move on

| Symptom                                                                      | Next step                                                 |
| ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| Model hallucinates company-specific facts.                                   | Integrate **RAG** so it can reference live documents.     |
| Output must always follow rigid company style *at scale*.                    | Invest in **fine-tuning** to bake the style into weights. |
| Prompt iterations are producing diminishing returns yet latency is critical. | Consider lightweight adapter tuning or small RAG context. |

---

### 8. Take-home summary

1. **Prompt engineering = cheapest optimisation layer.**
2. It shines for **speed, flexibility, and low-volume tasks**.
3. It struggles with **out-of-scope knowledge and strict consistency**.
4. Use it first, then layer **RAG** for fresh data and **fine-tuning** for fixed behaviour.

Master the prompt toolbox, and you’ll squeeze far more value out of even the plainest foundation model—often without spending a cent on GPUs or retraining.

---

## [Retrieval-Augmented Generation (RAG)](https://github.com/MuhammadAhsaanAbbasi/generative-ai/tree/main/02_RAG)

**RAG** is a framework that enhances the generation of responses from a language model by augmenting it with external, up-to-date, and relevant information retrieved from specific data sources (like the web, documents, or databases).

### High-Level Overview:

In the context of RAG, think of the LLM Knowledge as the central piece, but it is being fed with various retrieval sources:

- **Web**: External data fetched from the internet.
- **PDF**: Information extracted from documents.
- **Code**: Programmatic knowledge or examples retrieved from code bases.
- **Video Transcript**: Information extracted from video transcripts to include context from audiovisual content.

![High-Level Overview](https://myapplication-logos.s3.ap-south-1.amazonaws.com/HighLevel+Overveiw+RAG.jpg)

The RAG process involves retrieving relevant data from these sources, and the LLM then uses this data to generate a response. The retrieved information supplements the LLM's knowledge, resulting in more accurate and context-aware outputs.

---

### Detailed Overview of RAG:

![Detail Overview](https://myapplication-logos.s3.ap-south-1.amazonaws.com/Detailed+Overview+RAG.jpg)

## Core Components

### 1. External Knowledge Sources

Web pages, PDFs, policy docs, code bases, video transcripts, anything you can turn into text is fair game.  These sources live **outside** the model and can be updated independently.

### 2. Chunking

Large documents are sliced into small, overlapping passages (typically 300-1 000 tokens).  This keeps each piece topically focused and cheap to search.

### 3. Embeddings

Every chunk is converted into a numerical vector that captures semantic meaning.  Similar ideas land near one another in this high-dimensional space.

### 4. Vector Store

A purpose-built database (Pinecone, Chroma, Weaviate, etc.) stores those vectors and supports lightning-fast similarity search, plus metadata filters and access controls.

### 5. Similarity Retrieval

When a user asks a question, their query is embedded too.  The system pulls the *k* most similar chunks—often refined with hybrid keyword + semantic search to boost precision.

### 6. Prompt Assembly

Retrieved chunks are “stapled” onto the user’s question, along with system instructions like *“Cite all sources and answer only from provided context.”*

### 7. Generation

The language model now drafts its reply, grounding every claim in the supplied passages.  Many implementations surface inline citations so users can audit each fact.

----

## Benefits of RAG:

- **Efficiency:** Instead of passing an entire document to the model, RAG retrieves only the most relevant chunks, reducing the computational load.
- **Accuracy:** By using real-time data retrieval, the model can generate more accurate and context-aware answers.
- **Scalability:**: RAG can scale to handle large volumes of text, as it uses chunking and efficient retrieval techniques to access specific parts of the document.

----

<!-- ## Agents & Tools

**Agents** & **Tools** are two key concepts in LangChain that allow language models to perform actions, interact with external systems, and generate results dynamically.

![Agents & Tools](https://myapplication-logos.s3.ap-south-1.amazonaws.com/Agents+tool.jpg)

### Agents
In LangChain, an Agent is essentially a language model (LLM) that has been provided with a specific prompt to define its behavior. The behavior of an agent is comparable to a state machine, where different actions are performed depending on the agent's state. Each state has its own action, and the agent moves from one state to the next, looping through tasks as defined by the prompt.

#### Agent Process:
- **Action:** The agent takes an action (like answering a question, performing a task, etc.).
- **Observation:** After taking an action, the agent observes the result or feedback from the action.
- **Thought:** Based on its observations, the agent thinks or processes the information.
- **Result:** Finally, the agent produces a result or output based on its thought process.

This cycle repeats, allowing the agent to handle tasks dynamically. Each time, the agent's actions are guided by the prompts you design, which tell it how to behave in different states.

### Tools
Tools in LangChain are interfaces that an agent, chain, or LLM can use to interact with the external world. These tools enable agents to perform actions beyond simple text generation, such as searching the web, executing code, or querying a database.

#### Common Tools:
- **Search Internet:** The agent uses this tool to retrieve information from the web, accessing real-time data to supplement its responses.
- **Execute Code:** With this tool, the agent can run scripts or code to perform computations or other programmatic tasks.
- **Query Database:** This tool allows the agent to access and retrieve information from databases, providing more structured data or facts in its outputs.
By using these tools, the agent can perform more complex tasks and retrieve relevant data from external sources, enhancing its functionality and making it more versatile.

### Conclusion
In summary, *agents* in LangChain are state-driven language models that move through a sequence of actions, observations, and thoughts to produce a result. *Tools* enhance the agent's capabilities by providing interfaces to interact with the external world, enabling it to search for information, run code, or query databases. Together, agents and tools allow you to create highly dynamic, flexible, and intelligent systems capable of complex tasks. -->

<h2 align="center">
Dear Brother and Sister Show some ❤ by <img src="https://imgur.com/o7ncZFp.jpg" height=25px width=25px> this repository!
</h2>