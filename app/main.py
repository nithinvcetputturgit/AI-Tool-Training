from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# --- Load environment ---
load_dotenv()

# --- App setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- Knowledge Base (Direct Resume Content) ---
dummy_texts = [
    # Contact Information
    "Nithin Shetty M's email address is shettyn517@gmail.com.",
    "You can contact Nithin via email at shettyn517@gmail.com.",
    "His contact email is shettyn517@gmail.com.",
    "His LinkedIn profile is https://linkedin.com/in/nithin-shetty-m-530274265.",
    "Nithin's GitHub: https://github.com/nithinshettygit.",
    "Nithin is located in Dakshina Kannada, Karnataka, 574325.",

    # Executive Summary
    "Nithin Shetty M is an AI & ML engineering student with hands-on experience in deep learning, LLM-based agents, and real-time systems.",
    "He is skilled in deploying AI/ML solutions using LangChain, FAISS, and Python.",
    "Nithin is actively seeking AI/ML/GenAI Engineer roles to contribute to impactful, intelligent applications.",

    # Education
    "Nithin is a Final Year B.E. student in Artificial Intelligence and Machine Learning at Vivekananda College of Engineering and Technology (VCET), Puttur, Karnataka.",
    "He is affiliated with Visvesvaraya Technological University and expected to graduate in June 2026.",
    "He maintains a CGPA of 8.60 / 10.00 (First 6 Semesters)."
    "Courses: Data Structures, Algorithms, DBMS, Computer Networks, Operating Systems, AI & ML Fundamentals, Deep Learning, NLP.",  
    "High School: Shri Ramachandra High School Perne (2020) with 95.36% marks in state board exam.",
    "PUC Sri Rama PUC Kalladka Board: Karnataka State Board, Percentage: 85%."
    ,
    # Projects
    "Project Title: AIRA – AI Powered Smart Teaching Robot. Period: March 2025 – June 2025. "
    "Nithin built a RAG and LLM-based AI teaching agent that features real-time Q&A with interruption-resume logic. "
    "He integrated FAISS vector search for fast response. Tools: LangChain, GeminiFlash LLM, FAISS, Python, React.js, FastAPI.",

    "Project Title: Autonomous Wheelchair using Deep Learning. Period: July 2024 – Oct 2024. "
    "He developed a CNN-based model for real-time wheelchair navigation with ESP8266 hardware and Flask UI. "
    "Tools: Python, PyTorch, OpenCV, Flask, Arduino.",

    "Project Title: Hand Gesture Controlled Wheelchair. "
    "Designed real-time gesture recognition for controlling wheelchair direction. "
    "Technologies: Python, OpenCV, MediaPipe, Arduino. "
    "GitHub Link: https://github.com/nithinshettygit/Hand-Gesture-Controlled-Wheelchair.",

    # Skills
    "Programming Languages: Python, Java.",
    "AI/ML: PyTorch, scikit-learn, OpenCV, NLP, Generative AI.",
    "LLM & GenAI Tools: LangChain, OpenAI API, LLaMA, FAISS, RAG, Sentence-Transformers.",
    "Web/UI: Flask, Streamlit, React.js.",
    "Soft Skills: Communication, Teamwork, Problem-solving.",

    # Certifications
    "Nithin has a certification from Udemy: 'AI & LLM Engineering Mastery: GenAI, RAG Complete Guide'."
]

# --- Embeddings & Vectorstore ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(dummy_texts, embedding=embeddings)

# --- LLM ---
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.4
)

# --- Prompt ---
CUSTOM_PROMPT_TEMPLATE = """
You are Nithin Shetty M's highly detailed and helpful AI portfolio assistant.
Your job is to answer ONLY using the provided context.

Rules:
- If asked about projects → mention ALL projects with full details (title, role, technologies, GitHub).
- If asked about skills → list ALL categories (programming, AI/ML, GenAI tools, Web/UI, soft skills).
- If asked for contact → always give email (shettyn517@gmail.com), LinkedIn, and GitHub.
- For greetings (hi, hello) → reply friendly and offer help.
- For closings (bye, thanks) → reply politely.
- If info is missing → say: "I cannot find that information in Nithin's portfolio. Contact him at shettyn517@gmail.com."
- Never invent information.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=CUSTOM_PROMPT_TEMPLATE
)

# --- Memory & Chain ---
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True, output_key="answer", k=5
)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False
)

# --- Schemas ---
class Query(BaseModel):
    question: str

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(query: Query):
    response = conversation_chain.invoke(
        {"question": query.question, "chat_history": memory.buffer_as_messages}
    )
    return {"answer": response["answer"]}

@app.post("/clear")
async def clear_chat():
    memory.clear()
    return {"status": "cleared"}
