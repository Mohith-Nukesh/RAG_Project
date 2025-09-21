

from api import API_KEY
import os
import warnings
import logging
import datetime
import uuid
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader

# -------------------------
# Setup & Warnings
# -------------------------
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

os.environ["USE_TF"]="0"
os.environ["USE_TORCH"]="1"
os.environ["TRANSFORMERS_NO_TF_IMPORT"]="1"

# -------------------------
# Paths
# -------------------------
CSV_DIR=r"C:\Users\mohith nukesh\OneDrive\Desktop\Python\my_rag\csv"
os.makedirs(CSV_DIR,exist_ok=True)

FAQ_DATA_JSON=os.path.join(CSV_DIR,"faq_data.json")
TICKET_AI_JSON=os.path.join(CSV_DIR,"ticket_ai.json")
TICKET_INTERNAL_JSON=os.path.join(CSV_DIR,"ticket_internal.json")

# -------------------------
# JSON Helpers
# -------------------------
def append_json(path,new_entry):
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f:
            try:
                data=json.load(f)
                if not isinstance(data,list):
                    data=[data]
            except json.JSONDecodeError:
                data=[]
    else:
        data=[]
    data.append(new_entry)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2)

# -------------------------
# Embeddings
# -------------------------
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device":"cpu"}
)

def load_main_kb():
    return FAISS.load_local(
        "faiss_index_2",
        embeddings,
        allow_dangerous_deserialization=True
    )

# -------------------------
# Initialize LLM
# -------------------------
llm=ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=API_KEY,
    temperature=0.2
)


qa_prompt = PromptTemplate.from_template(
"""
You are a precise and helpful support assistant.

### Rules:
- Always use the provided context when possible.  
- If context is insufficient, infer a possible answer but add: "⚠️ As per my knowledge, verify with team",**once at the bottom** 
- Format answers as **bullet points** for clarity.  
- Keep answers concise:
  - If the question is short/simple → reply in **2–4 bullet points** max.  
  - If the question is complex → reply in **5–10 structured bullet points**.  
- Avoid long paragraphs. Each bullet should be 1–2 short sentences.
- Trim unnecessary details, and do not  repeat senetences again.
- Maintain a consistent tone: concise, step-by-step, professional but user-friendly.  
### Context:
{context}

Previous QA history (if any):
{history}

Question: {question}

Answer:
"""
)


faq_chain=LLMChain(llm=llm,prompt=qa_prompt,output_key="answer")

# -------------------------
# Helper to remove duplicate lines from AI answer
# -------------------------
def clean_ai_answer(answer):
    lines = answer.split("\n")
    seen = set()
    cleaned_lines = []
    for line in lines:
        line_strip = line.strip()
        if line_strip and line_strip not in seen:
            cleaned_lines.append(line_strip)
            seen.add(line_strip)
    return "\n".join(cleaned_lines)

# -------------------------
# Input helpers
# -------------------------
def get_number_input(prompt, valid_choices=None, allow_cancel=True, number_range=None):
    while True:
        user_input=input(prompt).strip()
        if allow_cancel and user_input.lower()=="cancel":
            return "cancel"
        if valid_choices and user_input in valid_choices:
            return user_input
        if number_range and user_input.isdigit() and number_range[0]<=int(user_input)<=number_range[1]:
            return int(user_input)
        print("⚠️ Invalid input. Please enter a valid number corresponding to the choices provided.")

# -------------------------
# FAQ Flow
# -------------------------
def run_faq(user_id,session_id):
    main_kb=load_main_kb()

    print("Choose source:\n1. PDF\n2. KB\n3. Both\n4. Cancel")
    choice=get_number_input("Enter number: ", valid_choices=["1","2","3","4"])
    if choice=="cancel" or choice=="4":
        print("❌ FAQ session cancelled.")
        return

    pdf_kb=None
    if choice in ["1","3"]:
        while True:
            pdf_path=input("Enter PDF path (or type 'cancel'): ").strip()
            if pdf_path.lower()=="cancel":
                print("❌ FAQ session cancelled.")
                return
            if not os.path.exists(pdf_path):
                print("⚠️ Path not found. Please enter a valid PDF path.")
                continue
            try:
                loader=PyPDFLoader(pdf_path)
                pdf_docs=loader.load()
                pdf_kb=FAISS.from_documents(pdf_docs,embeddings)
                break
            except Exception as e:
                print(f"⚠️ Error loading PDF: {str(e)}. Please check the file and try again.")

    history=""
    num_sub_queries=0
    conversation=[]

    while True:
        question=input("\nEnter your question (or type 'cancel' to finish): ").strip()
        if question.lower()=="cancel":
            break

        if choice=="2":
            results=main_kb.as_retriever(search_kwargs={"k":5}).get_relevant_documents(question)
        elif choice=="1":
            results=pdf_kb.as_retriever(search_kwargs={"k":5}).get_relevant_documents(question)
        else:
            kb_results=main_kb.as_retriever(search_kwargs={"k":3}).get_relevant_documents(question)
            pdf_results=pdf_kb.as_retriever(search_kwargs={"k":3}).get_relevant_documents(question)
            results=kb_results+pdf_results

        context_text="\n\n".join([doc.page_content for doc in results])
        answer=faq_chain.run({"context":context_text,"question":question,"history":history})
        answer=clean_ai_answer(answer)

        print("\n🤖 Answer:",answer)

        # sources=[f"{doc.metadata.get('source')} (p{doc.metadata.get('page')})" for doc in results]
        sources=list(dict.fromkeys([f"{doc.metadata.get('source')} (p{doc.metadata.get('page')})" for doc in results]))
        conversation.append({"complaint":question,"ai_answer":answer,"sources":sources})
        history+=f"\nQ: {question}\nA: {answer}"
        num_sub_queries+=1

        print("Do you want to ask another question?\n1. Yes\n2. No\n3. Cancel")
        cont=get_number_input("Enter number: ", valid_choices=["1","2","3"])
        if cont=="cancel" or cont!="1":
            break

    if conversation:
        rating=get_number_input("Rate this FAQ session (1–10): ", number_range=(1,10))
        feedback=input("Any feedback on this session?: ").strip()
        faq_id=f"F{uuid.uuid4().hex[:4]}"

        faq_data={
            "session_id":session_id,
            "faq_id":faq_id,
            "user_id":user_id,
            "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rating":rating,
            "feedback":feedback,
            "num_sub_queries":num_sub_queries,
            "conversation":conversation
        }
        append_json(FAQ_DATA_JSON,faq_data)
        print(f"✅ FAQ session logged to {FAQ_DATA_JSON}")

# -------------------------
# Ticket Flow
# -------------------------
def run_ticket(user_id,session_id):
    main_kb=load_main_kb()

    conversation=[]
    num_sub_queries=0
    rating=None
    feedback=None
    escalated=False
    ticket_id=f"T{uuid.uuid4().hex[:4]}"

    while True:
        complaint=input("\nEnter your complaint (or type 'cancel' to quit): ").strip()
        if complaint.lower()=="cancel":
            print("❌ Ticket session cancelled.")
            return

        results=main_kb.as_retriever(search_kwargs={"k":5}).get_relevant_documents(complaint)
        context_text="\n\n".join([doc.page_content for doc in results])
        ai_answer=faq_chain.run({"context":context_text,"question":complaint,"history":""})
        ai_answer=clean_ai_answer(ai_answer)

        print("\n🤖 AI Attempt:",ai_answer)
        # sources=[f"{doc.metadata.get('source')} (p{doc.metadata.get('page')})" for doc in results]
        sources=list(dict.fromkeys([f"{doc.metadata.get('source')} (p{doc.metadata.get('page')})" for doc in results]))


        conversation.append({"complaint":complaint,"ai_answer":ai_answer,"sources":sources})
        num_sub_queries+=1

        print("Options:\n1. Escalate\n2. Ask again\n3. Solved\n4. Cancel")
        next_step=get_number_input("Enter number: ", valid_choices=["1","2","3","4"])
        if next_step=="cancel" or next_step=="4":
            print("❌ Ticket session cancelled.")
            return
        elif next_step=="2":
            continue
        elif next_step=="3":
            rating=get_number_input("Rate this ticket session (1–10): ", number_range=(1,10))
            feedback=input("Any feedback for AI attempt?: ").strip()
            break
        elif next_step=="1":
            rating=get_number_input("Rate AI attempt before escalation (1–10): ", number_range=(1,10))
            feedback=input("Any feedback for AI attempt?: ").strip()
            escalated=True

            print("Which team should handle this?\n1. HR\n2. IT\n3. Finance\n4. Facilities\n5. Legal\n6. General\n7. Cancel")
            subdomain_choice=get_number_input("Enter number: ", valid_choices=["1","2","3","4","5","6","7"])
            mapping={"1":"HR","2":"IT","3":"Finance","4":"Facilities","5":"Legal","6":"General","7":"cancel"}
            subdomain=mapping[subdomain_choice]
            if subdomain.lower()=="cancel":
                print("❌ Escalation cancelled, ending ticket session.")
                break

            final_complaint=input("Enter final complaint for escalation (or type 'cancel'): ").strip()
            if final_complaint.lower()=="cancel":
                print("❌ Escalation cancelled, ending ticket session.")
                break

            ticket_internal={
                "ticket_id":ticket_id,
                "session_id":session_id,
                "user_id":user_id,
                "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_complaint":final_complaint,
                "subdomain":subdomain,
                "status":"Pending",
                "team_solution":None,
                "resolution_notes":None
            }
            append_json(TICKET_INTERNAL_JSON,ticket_internal)
            print(f"🚨 Ticket escalated and logged to {TICKET_INTERNAL_JSON}")
            break

    ticket_ai={
        "session_id":session_id,
        "ticket_id":ticket_id,
        "user_id":user_id,
        "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rating":rating,
        "feedback":feedback,
        "num_sub_queries":num_sub_queries,
        "escalated":escalated,
        "conversation":conversation
    }
    append_json(TICKET_AI_JSON,ticket_ai)
    print(f"✅ Ticket logged to {TICKET_AI_JSON}")

# -------------------------
# Main Pipeline
# -------------------------

def run_pipeline():
    print("=== Welcome to Support Assistant ===")

    while True:
        user_id=input("Enter your User ID (or type 'exit' to quit): ").strip()
        if user_id.lower() in ["exit","cancel"]:
            print("👋 Exiting Support Assistant.")
            return
        if user_id:  # ensure non-empty ID
            break
        print("⚠️ Invalid input. Please enter a valid User ID or type 'exit' to quit.")
    
    session_id=f"S{str(uuid.uuid4())[:5]}"  # persistent session_id for this run

    while True:
        print("\nMain Menu → Choose:\n1. FAQ\n2. Ticket\n3. Exit")
        mode=get_number_input("Enter number: ", valid_choices=["1","2","3"])
        if mode=="3":
            print("👋 Exiting Support Assistant.")
            break
        elif mode=="1":
            run_faq(user_id,session_id)
        elif mode=="2":
            run_ticket(user_id,session_id)

if __name__=="__main__":
    run_pipeline()














