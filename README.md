
README.md
# RAG-Based Support Assistant

A Python-based AI-powered **support assistant** using Retrieval-Augmented Generation (RAG) for context-aware FAQ responses and ticket handling. This assistant can answer questions using a PDF knowledge base, an existing KB, or both, and also supports ticket logging, escalation, and conversation tracking.

---

## **Features**

- **FAQ Assistant**
  - Provides precise, bullet-point answers.
  - Can use PDFs, KB, or both as context.
  - Tracks conversation history.
  - Cleans duplicate responses for clarity.
  
- **Ticket Management**
  - AI attempts initial resolution.
  - Option to escalate tickets to relevant teams (HR, IT, Finance, etc.).
  - Logs detailed conversation, feedback, and ratings.
  
- **Session Management**
  - Assigns unique `user_id` and `session_id` for tracking.
  - Logs FAQ and ticket sessions in JSON files for later analysis.

---


## Folder Structure

RAG_Project/
│
├── codes/                  # Python scripts
│   ├── main.py
│   └── utils.py
│
├── data/                   # Dataset PDFs
│   ├── dataset1.pdf
│   └── dataset2.pdf
│
├── analysis/               # Analysis images / plots
│   ├── plot1.jpg
│   └── plot2.jpg
│
├── requirements.txt        # Python dependencies
└── README.md               # Project overview



---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/Mohith-Nukesh/RAG_Project.git
cd RAG_Project


Install required packages:

pip install -r requirements.txt


Set your API key:

# In api.py
API_KEY = "your_groq_api_key_here"

Usage

Run the main pipeline:

python codes/main.py


Follow prompts:

Enter your User ID.

Choose between FAQ or Ticket.

Provide PDFs if required.

Rate and give feedback after session.

JSON Logs

faq_data.json – stores FAQ session data.

ticket_ai.json – stores AI-handled ticket sessions.

ticket_internal.json – stores escalated tickets and team logs.

Notes

Works with HuggingFace embeddings (all-MiniLM-L6-v2) and Groq LLM (llama-3.1-8b-instant).

Designed for step-by-step, concise answers.

Supports cancellation at any prompt by typing cancel.
