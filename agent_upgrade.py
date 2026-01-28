import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup
load_dotenv() # This looks for a .env file locally but won't exist on GitHub
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
chat_history = []

@tool
def check_fraud_risk(amount: float, user_id: str) -> str:
    """
    REQUIRED for any money transfers or large amounts. 
    Call this ONLY ONCE per request.
    """
    url = "http://13.61.22.182:8000/detect"
    payload = {
        "amount": amount,
        "user_id": user_id,
        "location_lat": 40.7128, "location_lon": -74.006,
        "merchant_id": "agent_007", "timestamp": "2026-01-28T16:00:00",
        "transaction_id": "txn_agent_live"
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        score = response.json().get('prediction', 0.5)
        return f"Risk Score: {score}. (Threshold: > 0.7 is High Risk)"
    except Exception as e:
        return f"API Error: {str(e)}"

# 2. Modern Agent Setup with Memory Placeholder
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [check_fraud_risk]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Bank Security AI. You MUST use 'check_fraud_risk' for money transfers. "
               "Remember the user's name or ID if they tell you."),
    MessagesPlaceholder(variable_name="chat_history"), # <--- Memory goes here
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. Interactive Loop
if __name__ == "__main__":
    print("\n🏦 Bank Security Agent Active (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]: break
            
        # Invoke with history
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # Update history state
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["output"]))
        
        print(f"\nAssistant: {response['output']}")