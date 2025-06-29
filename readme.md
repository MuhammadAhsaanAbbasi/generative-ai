# Generative AI Course Repository

**Generative AI** refers to artificial intelligence systems capable of creating new content—such as text, images, audio, or code—by learning patterns from existing data. These systems utilize sophisticated machine learning models, particularly deep learning algorithms, to understand and replicate the structures found in their training datasets. For example, models like OpenAI's DALL-E can generate images from textual descriptions, while ChatGPT produces human-like text responses based on input prompts.

**Large Language Models (LLMs)** are a subset of generative AI focused on processing and generating human language. Trained on vast amounts of text data, LLMs can perform a variety of natural language processing tasks, including text generation, translation, summarization, and question-answering. They achieve this by leveraging deep learning architectures, such as transformers, to capture the nuances of language and context. Notable examples include OpenAI's GPT series and Google's BERT.

In summary, generative AI encompasses a broad range of AI technologies capable of creating new content across various modalities, while LLMs specifically deal with understanding and generating human language, forming a crucial component of the generative AI landscape.

---

## [Prompt Engineering](https://github.com/MuhammadAhsaanAbbasi/generative-ai/tree/main/00_foundations/01_prompt_engineering)

[Watch: Prompt Engineering Fundamentals](https://learn.microsoft.com/en-gb/shows/generative-ai-for-beginners/understanding-prompt-engineering-fundamentals-generative-ai-for-beginners)

[Watch: Creating Advanced Prompts](https://learn.microsoft.com/en-gb/shows/generative-ai-for-beginners/creating-advanced-prompts-generative-ai-for-beginners)

**Prompt engineering** is the process of designing and refining inputs—known as prompts—to guide generative AI models in producing desired outputs. This practice involves crafting specific instructions or questions that effectively communicate the user's intent to the AI system, thereby enhancing the relevance and accuracy of the generated responses. 

In the context of **generative AI**, which includes models capable of creating text, images, or other content, prompt engineering is crucial for several reasons:

1. **Improving Output Quality**: Well-crafted prompts help AI models better understand the nuances and intent behind a query, leading to more accurate and contextually appropriate responses. 

2. **Enhancing Model Comprehension**: By providing clear and detailed prompts, users can assist AI models in comprehending complex or technical queries, thereby improving the model's ability to generate relevant outputs. 

3. **Mitigating Biases and Errors**: Effective prompt engineering involves iterative refinement of prompts to minimize biases and confusion, which helps in producing more accurate and unbiased responses from AI models. 

4. **Expanding AI Capabilities**: Through strategic prompt design, users can leverage AI models to perform a wide range of tasks, from answering simple questions to generating complex technical content, thereby maximizing the utility of generative AI systems. 

In summary, prompt engineering serves as a vital interface between human intent and AI-generated content, playing a key role in harnessing the full potential of generative AI technologies. 

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