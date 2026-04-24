# 🏅 Olympic Medal Analysis with GenAI

A complete **data analysis + Generative AI project** that analyzes Olympic medal data and generates human-like insights using LLMs.

---

## 🚀 Project Overview

This project combines:

* 📊 Data Analysis using **Pandas**
* 📈 Visualization using **Matplotlib & Seaborn**
* 🤖 AI-generated insights using **LLM (Groq + LangChain)**

It performs:

* Country-wise medal analysis
* Year-wise trends
* Sport-wise breakdown
* AI-generated summary reports

---

## 📂 Project Structure

```
olympic_analysis/
├── graphs/      # Saved charts (.png)
├── reports/     # AI-generated analysis (.txt)
├── logs/        # Execution logs
```

---

## 📊 Features

### 1. Top Countries Analysis

* Displays top N countries by total medals
* Stacked bar chart (Gold, Silver, Bronze)

### 2. Year-wise Trend

* Medal trend over years for a selected country

### 3. Sport-wise Distribution

* Top sports contributing to medals

### 4. GenAI Integration 🤖

* Uses LLM to generate:

  * Insights
  * Patterns
  * Key observations

---

## ⚙️ Tech Stack

* Python 🐍
* pandas
* matplotlib
* seaborn
* LangChain
* Groq

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Olympic-medal-analysis-by-country.git
cd Olympic-medal-analysis-by-country
```

---

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn langchain langchain-core langchain-groq
```

---

### 3. Add Dataset

Place:

```
athlete_events.csv
```

in the root folder.

---

### 4. Add API Key

Edit config:

```python
"groq_api_key": "your_api_key_here"
```

---

### 5. Run the project

```bash
python olympic_analysis.py
```

---

## 📈 Output

After running, you’ll get:

* 📊 Graphs → `olympic_analysis/graphs/`
* 📝 AI Report → `olympic_analysis/reports/`
* 🪵 Logs → `olympic_analysis/logs/`

---

## 🧠 How GenAI is Used

The project:

1. Generates structured data summaries
2. Sends them to an LLM
3. Produces human-readable explanations

Example output:

> "USA shows a consistent rise in medal count over the years, indicating strong athletic dominance..."

---

## 🔥 Key Highlights

* End-to-end pipeline (Data → Visualization → AI Insight)
* Clean modular structure
* Real-world dataset
* Resume-ready project

---

## 📌 Future Improvements

* Add user input (country selection)
* Convert to web app (Streamlit)
* Add predictive ML models

---

## 👨‍💻 Author

Your Name

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
