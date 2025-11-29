# Agentic Financial Risk Assessment System 📉🤖

An autonomous AI agent designed to analyze corporate financial health, detect bankruptcy risks, and generate audit-grade risk reports. Built on **LangGraph**, this system features a self-correcting multi-agent architecture that validates generated insights against real-time market data.

## 🌟 Key Features

* **Multi-Agent Workflow:** Orchestrates two distinct agents:
    * **Analyst:** Generates risk reports based on financial data.
    * **Validator:** Critiques the report against raw metrics to ensure logical consistency.
* **Live Market Data:** Custom tool integration with **Yahoo Finance (yfinance)** to fetch real-time stock and ratio data.
* **Strict Validation:** A specialized JSON-based validation node cross-references 15+ financial ratios (e.g., Altman Z-Score, Current Ratio) to eliminate hallucinations.
* **Cloud Native:** Containerized with **Docker** and deployed as a serverless microservice on **Google Cloud Run**.

## 🏗️ Tech Stack

* **Framework:** LangChain, LangGraph
* **LLM:** Google Gemini 2.0 Flash
* **Frontend:** Streamlit
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Deployment:** Docker, Google Cloud Platform (Cloud Run)

## 📦 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/financial-risk-agent.git](https://github.com/yourusername/financial-risk-agent.git)
    cd financial-risk-agent
    ```

2.  **Environment Variables**
    Create a `.env` file and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    ```

3.  **Run Locally (Python)**
    ```bash
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```

## 🐳 Docker Deployment

1.  **Build the Container**
    ```bash
    docker build -t risk-agent .
    ```

2.  **Run Container Locally**
    ```bash
    docker run -p 8080:8080 --env-file .env risk-agent
    ```

3.  **Deploy to Google Cloud Run**
    ```bash
    gcloud run deploy risk-agent-service \
      --source . \
      --region us-central1 \
      --allow-unauthenticated
    ```

## 🔄 Workflow Logic

1.  **Input:** User provides a Ticker Symbol (e.g., `AAPL`).
2.  **Tool Call:** Agent fetches live balance sheet and income statement data via `yfinance`.
3.  **Analysis:** The *Analyst Agent* computes ratios (Liquidity, Solvency, Profitability) and drafts a preliminary risk assessment.
4.  **Validation:** The *Validator Agent* reviews the draft against the calculated ratios. If discrepancies are found (e.g., claiming "High Solvency" despite a high Debt/Equity ratio), it rejects the report.
5.  **Output:** A finalized, validated risk report is displayed on the Streamlit dashboard.