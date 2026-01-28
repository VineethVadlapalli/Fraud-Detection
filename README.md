# Agentic Fraud Detection System 🛡️🤖

## 🚀 Overview
A high-performance Fraud detection pipeline designed to identify fraudulent financial transactions. The system combines an Unsupervised Ensemble (Isolation Forest, KNN) with a Business Rule Engine to provide real-time, explainable risk assessments.

An end-to-end, production-ready fraud detection ecosystem featuring a **Dockerized ML API** and an **Autonomous LangChain Agent**.

## 🚀 System Architecture
- **Model:** Ensemble (XGBoost/RandomForest, Isolation Forest etc..) trained for financial transaction anomaly detection.
- **Inference API:** Built with **FastAPI**, containerized via **Docker**, and deployed on **AWS EC2**.
- **Orchestration:** **LangChain** agent utilizing **Groq (Llama 3.3 70B)** for real-time tool calling and conversational risk assessment.

## 📊 Performance Metrics (Stress Tested)
- **Reliability:** 100% Success Rate over 100+ concurrent requests.
- **Latency:** ~746ms average response time on a `t3.micro` instance.
- **Self-Healing:** Implemented Docker restart policies and optimized Linux swap memory for 24/7 uptime.

## 🛠️ Tech Stack
- **Languages:** Python (Scikit-learn, Keras, Pandas)
- **Infrastructure:** AWS (EC2, Security Groups), Docker
- **Agentic AI:** LangChain, Groq API, Tool-Calling Logic
- **Database/Tools:** SQL, Tableau (for visualization)

## 💡 Key Features
- **Context-Aware Safety:** The Agent maintains chat history to identify users and transaction context before invoking the fraud model.
- **Autonomous Decisioning:** The LLM decides when a transaction is "suspicious" enough to require an API-based risk score.
- **Secure Deployment:** Environment variable isolation and automated secret scanning protection.

---
*Developed by Vineeth Vadlapalli | Data Scientist & AI Specialist*
