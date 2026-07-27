from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import GROQ_API_KEY


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

messages = [
    SystemMessage(content="You are a professional Data Science interviewer."),
    HumanMessage(content="Ask me a beginner Python question.")
]

response = llm.invoke(messages)
print(response.content)