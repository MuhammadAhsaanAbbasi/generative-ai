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

## [LangChain Overview (Course Focus)](https://github.com/MuhammadAhsaanAbbasi/generative-ai/tree/main/01_langchain)

This course is a practical deep-dive into LangChain. You will:

1. **Understand the Core Abstractions** – Models, Prompts, Chains, Retrievers, Agents, and Tools.  
2. **Build End-to-End Projects** – Starting with a simple Q&A bot, progressing to Retrieval-Augmented Generation, and finishing with agentic workflows that call external APIs.  
3. **Master Best Practices** – Secure key management, cost control, prompt testing, evaluation metrics, and deployment strategies.  
4. **Integrate with Real-World Data** – Load PDFs, SQL databases, REST endpoints, and live web content.  
5. **Deploy & Monitor** – Ship your LangChain apps via Streamlit/FastAPI and monitor usage, latency, and token spend.  

Whether you are a beginner exploring LLMs for the first time or an experienced engineer looking to accelerate product development, the lessons are structured so you can apply LangChain in diverse scenarios—from customer-support chatbots to document search engines and workflow automation.

> **Outcome:** By the end of the course you will be able to design, implement, and deploy robust language-model applications that combine sophisticated prompting, external knowledge retrieval, and dynamic tool use—all powered by LangChain.

---

## Retrieval-Augmented Generation (RAG)

**RAG** is a framework that enhances the generation of responses from a language model by augmenting it with external, up-to-date, and relevant information retrieved from specific data sources (like the web, documents, or databases).

### High-Level Overview:

In the context of LangChain and RAG, think of the LLM Knowledge as the central piece, but it is being fed with various retrieval sources:

- **Web**: External data fetched from the internet.
- **PDF**: Information extracted from documents.
- **Code**: Programmatic knowledge or examples retrieved from code bases.
- **Video Transcript**: Information extracted from video transcripts to include context from audiovisual content.

![High-Level Overview](https://myapplication-logos.s3.ap-south-1.amazonaws.com/HighLevel+Overveiw+RAG.jpg)

The RAG process involves retrieving relevant data from these sources, and the LLM then uses this data to generate a response. The retrieved information supplements the LLM's knowledge, resulting in more accurate and context-aware outputs.

### Detailed Overview of RAG:

#### Initial Input (PDF with 10M Tokens):
- You start with a large source of information, such as a PDF containing millions of tokens. Tokens are essentially the building blocks of text (like words or pieces of words).

#### Chunking:
- Since handling the entire document at once can be computationally expensive, the text is chunked into smaller pieces, typically of 1K tokens each.

- Each chunk represents a small segment of the original text, making it more manageable for processing and retrieval.

#### Text Embeddings (Vectorization):
- Embeddings are the core of this process. They are numerical representations of text, meaning the chunks of text are converted into vectors (arrays of numbers).

- For example, as shown in the diagram, simple embeddings for "Dog" could be [1,1,2,4,1], and similarly for "Cat" and "House."

- These embeddings help the machine understand and compare the chunks of text by converting the semantic meaning into a format it can mathematically process.

- LLM Embedder: The task of converting text into embeddings is done by a Language Model Embedder. This process may have a cost, because generating embeddings for large texts requires significant computational resources.

#### Vector Store:
- Once the text is embedded (converted to numerical form), the embeddings are stored in a Vector Store.

-A Vector Store is a specialized database designed to handle and store high-dimensional vectors (i.e., embeddings).

- It allows for efficient retrieval of relevant chunks based on similarity searches, meaning it can quickly find the chunks most closely related to a specific query.

#### Retrieval Process:
- When a user asks a Question, the question itself is also converted into an embedding.

- This embedding is then matched against the embeddings stored in the Vector Store, and the closest matches (in terms of similarity) are retrieved.

- The most relevant chunks of text (based on the vector similarity) are pulled from the vector store.

#### Combining Retrieved Chunks with the Question:
- The retrieved chunks are combined along with the user's original question to form a final input.

- This input, consisting of both the relevant chunks of text and the question, is sent to a Language Model to generate a final response.

- Final Output: The output includes all relevant chunks that can help the model generate an informed answer, as well as the specific response to the user’s question.

![Detail Overview](https://myapplication-logos.s3.ap-south-1.amazonaws.com/Detailed+Overview+RAG.jpg)

### Process Flow Overview:
- A large document (like a PDF) is split into smaller, manageable pieces (chunks).

- Each chunk is converted into a vector (embedding), which is a mathematical representation of the text.

- These embeddings are stored in a vector database (Vector Store).

- When a user asks a question, it is also converted into an embedding, and the system retrieves the most relevant chunks based on vector similarity.

- The relevant chunks, along with the user’s question, are processed together to generate a more informed response.

### Benefits of RAG:
- **Efficiency:** Instead of passing an entire document to the model, RAG retrieves only the most relevant chunks, reducing the computational load.
- **Accuracy:** By using real-time data retrieval, the model can generate more accurate and context-aware answers.
- **Scalability:**: RAG can scale to handle large volumes of text, as it uses chunking and efficient retrieval techniques to access specific parts of the document.

<hr />

## 5. Agents & Tools

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
In summary, *agents* in LangChain are state-driven language models that move through a sequence of actions, observations, and thoughts to produce a result. *Tools* enhance the agent's capabilities by providing interfaces to interact with the external world, enabling it to search for information, run code, or query databases. Together, agents and tools allow you to create highly dynamic, flexible, and intelligent systems capable of complex tasks.

<h2 align="center">Dear Brother and Sister Show some ❤ by <img src="https://imgur.com/o7ncZFp.jpg" height=25px width=25px> this repository!</h2>