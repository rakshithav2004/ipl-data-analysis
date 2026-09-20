# 🏏 IPL Data Analysis Dashboard

An interactive **Streamlit** dashboard for exploring **Indian Premier League (IPL)** match and ball-by-ball data (2008–2024). It covers team performance, player statistics, toss decisions, venues, and scoring patterns across phases of an innings.

## 📌 Project Overview

The app uses match-level and delivery-level datasets to compute statistics and render charts on demand. It is built to answer questions such as:

* Which teams have performed consistently across IPL seasons?
* How does a given batter or bowler perform against each opponent, and in winning vs losing matches?
* Does winning the toss actually help?
* Which teams dominate which venues?
* How do teams score in the Powerplay, middle overs, and death overs?

## 📂 Dataset

| Dataset          | Description                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------- |
| `matches.csv`    | Match-level information including teams, season, venue, toss, winner, player of the match, umpires |
| `deliveries.csv` | Ball-by-ball information including batting team, batter, bowler, runs, and dismissals       |

## 🛠️ Tech Stack

* **Python**
* **Streamlit**: interactive dashboard
* **Pandas**: data manipulation and aggregation
* **Matplotlib & Seaborn**: visualization

## 📁 Project Structure

```text
ipl-data-analysis/
│
├── app.py            # Streamlit dashboard
├── matches.csv
├── deliveries.csv
└── README.md
```

## 🔍 Dashboard Views

Use the sidebar to switch between:

| View | What it shows |
| ---- | ------------- |
| Batter Analysis | Runs, balls, 4s/6s, strike rate, runs vs each opponent, runs in won vs lost matches |
| Bowler Analysis | Wickets, economy, bowling strike rate and average, wickets vs each opponent |
| Team Wins Over Years | Season-wise win trends for all teams |
| Match Summary | Teams, venue, toss, winner, player of the match, umpires for any match |
| Toss Impact Analysis | Toss winner vs match winner, toss decisions overall and by venue |
| Venue Impact Analysis | Busiest venues and team wins per venue |
| Seasonal Analysis | Matches and winners per season, most successful teams |
| Player of the Match Analysis | Top award winners and season-wise heatmap |
| Most Successful Team | All-time win leaderboard |
| Team Performance by Season | Wins per season for a selected team |
| Head-to-Head Comparison | Wins, ties, and no-results between any two teams |
| Phase-wise Analysis | Runs, run rate, and wickets in Powerplay, Middle, and Death overs |
| Stadium-wise Performance | Dominant team at each venue and per-team venue records |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rakshithav2004/ipl-data-analysis.git
cd ipl-data-analysis
```

### 2. Install dependencies

```bash
pip install streamlit pandas matplotlib seaborn
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## 🎯 Skills Demonstrated

* Exploratory Data Analysis (EDA)
* Data cleaning and column standardization with Pandas
* Merging, grouping, and aggregation across match and delivery data
* Building an interactive Streamlit application
* Data visualization with Matplotlib and Seaborn
* Cricket analytics: strike rate, economy, phase-wise scoring, head-to-head records

## 📌 Future Improvements

* Season and team filters across all views
* Interactive charts (Plotly)
* Advanced player comparison metrics
* Predictive analysis for match outcomes
* Deploy on Streamlit Community Cloud

## 👩‍💻 Author

**Rakshitha Bai V**

GitHub: [rakshithav2004](https://github.com/rakshithav2004)
