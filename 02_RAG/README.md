# RAG ADVANCED TECHNIQUES

Welcome to the **RAG ADVANCED TECHNIQUES** section! This folder contains code samples, exercises, and resources designed to help you get started & getting knowledge of ADVanced Techniques of RAGs using *Pinecone* & *ChromaDB* for building applications that interact with language models like *GPT-4*, *GPT-4o*, *llama* & others *Open-Source* Models.

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
