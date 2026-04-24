# ============================================================
# 🏅 Olympic Medal Count Analysis - Structured Project
# ============================================================
# OUTPUT STRUCTURE:
#   olympic_analysis/
#   ├── graphs/          ← all charts saved as .png
#   ├── reports/         ← LLM response saved as .txt
#   └── logs/            ← run logs with timestamps
# ============================================================

import os
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ─────────────────────────────────────────────
# SECTION 1: CONFIG — change settings here only
# ─────────────────────────────────────────────
CONFIG = {
    "dataset_path":   "athlete_events.csv",
    "output_dir":     "olympic_analysis",
    "graphs_dir":     "olympic_analysis/graphs",
    "reports_dir":    "olympic_analysis/reports",
    "logs_dir":       "olympic_analysis/logs",
    "top_n":          30,          # how many top countries to show
    "country_trend":  "USA",       # country for year-wise trend chart
    "country_sports": "IND",       # country for sport-wise breakdown chart
    "groq_api_key":   "your api key",
    "groq_model":     "llama-3.1-8b-instant",
    "llm_temperature": 0.2,
}

# ─────────────────────────────────────────────
# SECTION 2: SETUP — folders + logging
# ─────────────────────────────────────────────

def setup_environment(cfg: dict) -> logging.Logger:
    """Create output folders and configure logging."""
    for folder in [cfg["graphs_dir"], cfg["reports_dir"], cfg["logs_dir"]]:
        os.makedirs(folder, exist_ok=True)

    log_path = os.path.join(cfg["logs_dir"], "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),          # also prints to console
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SECTION 3: DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV and return:
      - df:         full dataset
      - medals_df:  rows where a medal was awarded
    """
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    medals_df = df.dropna(subset=["Medal"])
    logger.info(f"Total rows: {len(df):,} | Medal rows: {len(medals_df):,}")
    return df, medals_df


# ─────────────────────────────────────────────
# SECTION 4: ANALYSIS HELPERS
# ─────────────────────────────────────────────

def get_top_countries(medals_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Group medals by country & type, return top N by total."""
    country_medals = (
        medals_df.groupby(["NOC", "Medal"])
        .size()
        .unstack()
        .fillna(0)
    )
    country_medals["Total"] = country_medals.sum(axis=1)
    return country_medals.sort_values("Total", ascending=False).head(top_n)


def get_year_trend(medals_df: pd.DataFrame, noc: str) -> pd.Series:
    """Year-wise medal count for a single country."""
    country_df = medals_df[medals_df["NOC"] == noc]
    return country_df.groupby("Year")["Medal"].count()


def get_sport_distribution(medals_df: pd.DataFrame, noc: str, top_n: int) -> pd.Series:
    """Top sports by medal count for a single country."""
    country_df = medals_df[medals_df["NOC"] == noc]
    return country_df["Sport"].value_counts().head(top_n)


# ─────────────────────────────────────────────
# SECTION 5: PLOTTING — all charts saved to disk
# ─────────────────────────────────────────────

def save_figure(fig: plt.Figure, filename: str, graphs_dir: str, logger: logging.Logger):
    """Save a matplotlib figure and close it."""
    path = os.path.join(graphs_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Graph saved → {path}")


def plot_top_countries(top_countries: pd.DataFrame, graphs_dir: str, logger: logging.Logger):
    """Stacked bar chart: top N countries by Gold / Silver / Bronze."""
    fig, ax = plt.subplots(figsize=(14, 6))
    top_countries[["Gold", "Silver", "Bronze"]].plot(
        kind="bar", stacked=True, ax=ax, colormap="viridis"
    )
    ax.set_title(f"Top {len(top_countries)} Countries by Olympic Medals 🏅", fontsize=14)
    ax.set_xlabel("Country (NOC Code)")
    ax.set_ylabel("Number of Medals")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Medal Type")
    fig.tight_layout()
    save_figure(fig, "top_countries_medals.png", graphs_dir, logger)


def plot_year_trend(year_medals: pd.Series, noc: str, graphs_dir: str, logger: logging.Logger):
    """Line chart: year-wise medal count for one country."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=year_medals, marker="o", color="green", ax=ax)
    ax.set_title(f"{noc} – Olympic Medal Count Over the Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Medals")
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, f"{noc.lower()}_year_trend.png", graphs_dir, logger)


def plot_sport_distribution(sport_counts: pd.Series, noc: str, graphs_dir: str, logger: logging.Logger):
    """Horizontal bar chart: sport-wise medals for one country."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sport_counts.values, y=sport_counts.index, palette="magma", ax=ax)
    ax.set_title(f"Top Sports – {noc} Medal Count")
    ax.set_xlabel("Number of Medals")
    ax.set_ylabel("Sport")
    fig.tight_layout()
    save_figure(fig, f"{noc.lower()}_sport_distribution.png", graphs_dir, logger)


# ─────────────────────────────────────────────
# SECTION 6: LLM ANALYSIS
# ─────────────────────────────────────────────

def build_llm_prompt(top_countries: pd.DataFrame, year_medals: pd.Series,
                     sport_counts: pd.Series, noc_trend: str, noc_sports: str) -> str:
    """Compose the text block sent to the LLM."""
    return f"""
Top {len(top_countries)} countries by total Olympic medals:
{top_countries.head(10).to_string()}

Year-wise medal trend for {noc_trend}:
{year_medals.to_string()}

Top sports for {noc_sports}:
{sport_counts.to_string()}
"""


def call_llm(prompt: str, cfg: dict, logger: logging.Logger) -> str:
    """Send prompt to Groq LLM and return the response text."""
    logger.info("Calling LLM for analysis...")
    os.environ["GROQ_API_KEY"] = cfg["groq_api_key"]

    model = ChatGroq(model=cfg["groq_model"], temperature=cfg["llm_temperature"])
    messages = [
        SystemMessage(content=(
            "You are a helpful sports analyst. Explain Olympic medal analysis "
            "in clear, simple human language. Highlight interesting patterns, "
            "surprising facts, and key takeaways."
        )),
        HumanMessage(content=prompt),
    ]
    response = model.invoke(messages)
    logger.info("LLM response received ✅")
    return response.content


# ─────────────────────────────────────────────
# SECTION 7: SAVE REPORT
# ─────────────────────────────────────────────

def save_report(llm_response: str, reports_dir: str, logger: logging.Logger):
    """
    Write the LLM analysis to a timestamped .txt file.
    A new file is created on every run so no data is overwritten.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"llm_report_{timestamp}.txt"
    path      = os.path.join(reports_dir, filename)

    header = (
        "=" * 60 + "\n"
        f"  Olympic Medal Analysis — LLM Report\n"
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + llm_response)

    logger.info(f"LLM report saved → {path}")
    return path


# ─────────────────────────────────────────────
# SECTION 8: MAIN — orchestrates everything
# ─────────────────────────────────────────────

def main():
    # 1. Setup
    logger = setup_environment(CONFIG)
    logger.info("=== Olympic Medal Analysis Started ===")

    # 2. Load data
    df, medals_df = load_data(CONFIG["dataset_path"], logger)

    # 3. Compute summaries
    top_countries = get_top_countries(medals_df, CONFIG["top_n"])
    year_medals   = get_year_trend(medals_df, CONFIG["country_trend"])
    sport_counts  = get_sport_distribution(medals_df, CONFIG["country_sports"], CONFIG["top_n"])

    # 4. Save all graphs  ← no plt.show() anywhere; all go to disk
    plot_top_countries(top_countries, CONFIG["graphs_dir"], logger)
    plot_year_trend(year_medals, CONFIG["country_trend"], CONFIG["graphs_dir"], logger)
    plot_sport_distribution(sport_counts, CONFIG["country_sports"], CONFIG["graphs_dir"], logger)

    # 5. LLM analysis
    prompt       = build_llm_prompt(top_countries, year_medals, sport_counts,
                                    CONFIG["country_trend"], CONFIG["country_sports"])
    llm_response = call_llm(prompt, CONFIG, logger)

    # 6. Save LLM report
    report_path  = save_report(llm_response, CONFIG["reports_dir"], logger)

    # 7. Done
    logger.info("=== Analysis Complete ===")
    logger.info(f"📊 Graphs  → {CONFIG['graphs_dir']}/")
    logger.info(f"📝 Report  → {report_path}")
    logger.info(f"🪵 Logs    → {CONFIG['logs_dir']}/run.log")
    print("\n✅ All done! Check the olympic_analysis/ folder for outputs.")


if __name__ == "__main__":
    main()