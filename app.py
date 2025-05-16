import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="IPL Analysis", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    try:
        df_deliveries = pd.read_csv("data/deliveries.csv")
        df_matches = pd.read_csv("data/matches.csv")
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None

    df_deliveries.columns = df_deliveries.columns.str.strip().str.lower()
    df_matches.columns = df_matches.columns.str.strip().str.lower()

    # Standardize column names
    if 'striker' in df_deliveries.columns:
        df_deliveries.rename(columns={'striker': 'batter'}, inplace=True)
    if 'batsman' in df_deliveries.columns:
        df_deliveries.rename(columns={'batsman': 'batter'}, inplace=True)
    if 'batsman_runs' in df_deliveries.columns:
        df_deliveries.rename(columns={'batsman_runs': 'batter_runs'}, inplace=True)

    return df_deliveries, df_matches

def batter_analysis(df_deliveries, df_matches):
    st.header("Batter Performance Analysis")
    batters = df_deliveries['batter'].unique()
    selected_batter = st.selectbox("Select a Batter", sorted(batters))

    df_batter = df_deliveries[df_deliveries['batter'] == selected_batter]
    total_runs = df_batter['batter_runs'].sum()
    total_balls = df_batter.shape[0]
    total_fours = df_batter[df_batter['batter_runs'] == 4].shape[0]
    total_sixes = df_batter[df_batter['batter_runs'] == 6].shape[0]
    strike_rate = round((total_runs / total_balls) * 100, 2) if total_balls > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Runs", total_runs)
    col2.metric("Balls Faced", total_balls)
    col3.metric("4s", total_fours)
    col4.metric("6s", total_sixes)
    col5.metric("Strike Rate", strike_rate)

    fig, ax = plt.subplots()
    df_batter['batter_runs'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Runs Scored Distribution")
    ax.set_xlabel("Runs per Ball")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Runs scored against different teams
    st.subheader("Runs Scored Against Each Team")
    merged_batter = df_batter.merge(df_matches[['id', 'team1', 'team2']], left_on='match_id', right_on='id', how='left')
    merged_batter['opponent'] = merged_batter.apply(
        lambda row: row['team2'] if row['team1'] == selected_batter else row['team1'], axis=1
    )
    runs_vs_opponent = merged_batter.groupby('opponent')['batter_runs'].sum().sort_values(ascending=False)
    fig_opponent, ax_opponent = plt.subplots()
    runs_vs_opponent.plot(kind='bar', ax=ax_opponent)
    ax_opponent.set_title(f"Runs Scored by {selected_batter} Against Each Team")
    ax_opponent.set_xlabel("Opponent Team")
    ax_opponent.set_ylabel("Total Runs")
    st.pyplot(fig_opponent)

    # Performance in winning vs losing matches
    st.subheader("Performance in Winning vs Losing Matches")
    merged_win_loss = df_batter.merge(df_matches[['id', 'winner']], left_on='match_id', right_on='id', how='left')
    merged_win_loss['match_result'] = merged_win_loss.apply(
        lambda row: 'Won' if row['winner'] == selected_batter else ('Lost' if pd.notna(row['winner']) else 'No Result'), axis=1
    )
    runs_win_loss = merged_win_loss.groupby('match_result')['batter_runs'].sum()
    fig_win_loss, ax_win_loss = plt.subplots()
    runs_win_loss.plot(kind='bar', ax=ax_win_loss)
    ax_win_loss.set_title(f"Runs Scored by {selected_batter} in Winning vs Losing Matches")
    ax_win_loss.set_xlabel("Match Result")
    ax_win_loss.set_ylabel("Total Runs")
    st.pyplot(fig_win_loss)

def bowler_analysis(df_deliveries, df_matches):
    st.header("Bowler Performance Analysis")
    bowlers = df_deliveries['bowler'].unique()
    selected_bowler = st.selectbox("Select a Bowler", sorted(bowlers))

    df_bowler = df_deliveries[df_deliveries['bowler'] == selected_bowler]
    total_balls = df_bowler.shape[0]
    total_runs_conceded = df_bowler['total_runs'].sum()
    total_wickets = df_bowler[df_bowler['dismissal_kind'].notna()].shape[0]
    economy = round((total_runs_conceded / (total_balls / 6)), 2) if total_balls > 0 else 0
    strike_rate_bowling = round(total_balls / total_wickets, 2) if total_wickets > 0 else 0
    average = round(total_runs_conceded / total_wickets, 2) if total_wickets > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Balls Bowled", total_balls)
    col2.metric("Runs Conceded", total_runs_conceded)
    col3.metric("Wickets", total_wickets)
    col4.metric("Economy Rate", economy)
    col5.metric("Bowling Strike Rate", strike_rate_bowling)

    col_avg = st.columns(1)
    col_avg[0].metric("Bowling Average", average)

    fig, ax = plt.subplots()
    df_bowler['total_runs'].hist(bins=20, ax=ax)
    ax.set_title("Runs Conceded per Delivery")
    ax.set_xlabel("Runs")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Wickets taken against different teams
    st.subheader("Wickets Taken Against Each Team")
    dismissals_bowler = df_bowler.dropna(subset=['dismissal_kind'])
    merged_bowler = dismissals_bowler.merge(df_matches[['id', 'team1', 'team2']], left_on='match_id', right_on='id', how='left')
    merged_bowler['opponent'] = merged_bowler.apply(
        lambda row: row['team2'] if row['team1'] == row['bowling_team'] else row['team1'], axis=1
    )
    wickets_vs_opponent = merged_bowler.groupby('opponent')['dismissal_kind'].count().sort_values(ascending=False)
    fig_wicket_opponent, ax_wicket_opponent = plt.subplots()
    wickets_vs_opponent.plot(kind='bar', ax=ax_wicket_opponent)
    ax_wicket_opponent.set_title(f"Wickets Taken by {selected_bowler} Against Each Team")
    ax_wicket_opponent.set_xlabel("Opponent Team")
    ax_wicket_opponent.set_ylabel("Total Wickets")
    st.pyplot(fig_wicket_opponent)

    # Performance in winning vs losing matches
    st.subheader("Performance in Winning vs Losing Matches")
    merged_bowl_win_loss = df_bowler.merge(df_matches[['id', 'winner']], left_on='match_id', right_on='id', how='left')
    merged_bowl_win_loss['match_result'] = merged_bowl_win_loss.apply(
        lambda row: 'Won' if row['winner'] == row['bowling_team'] else ('Lost' if pd.notna(row['winner']) else 'No Result'), axis=1
    )
    wickets_win_loss = merged_bowl_win_loss.groupby('match_result')['dismissal_kind'].count()
    fig_bowl_win_loss, ax_bowl_win_loss = plt.subplots()
    wickets_win_loss.plot(kind='bar', ax=ax_bowl_win_loss)
    ax_bowl_win_loss.set_title(f"Wickets Taken by {selected_bowler} in Winning vs Losing Matches")
    ax_bowl_win_loss.set_xlabel("Match Result")
    ax_bowl_win_loss.set_ylabel("Total Wickets")
    st.pyplot(fig_bowl_win_loss)

def team_wins_over_years(df_matches):
    st.header("Team Wins Over the Years")
    df_win = df_matches[df_matches['winner'].notna()]
    df_yearly_wins = df_win.groupby(['season', 'winner']).size().reset_index(name='wins')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_yearly_wins, x='season', y='wins', hue='winner', marker='o', ax=ax)
    ax.set_title("Team Wins by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Wins")
    plt.xticks(rotation=45)
    st.pyplot(fig)



def match_summary(df_matches):
    st.header("Match Summary")
    match_ids = df_matches['id'].unique()
    match_id = st.selectbox("Select a Match ID", match_ids)

    match = df_matches[df_matches['id'] == match_id].iloc[0]
    st.subheader(f"{match['team1']} vs {match['team2']}")
    st.text(f"Date: {match['date']}")
    st.text(f"Venue: {match['venue']}")
    st.text(f"Toss: {match['toss_winner']} - {match['toss_decision']}")
    st.text(f"Winner: {match['winner']}")
    st.text(f"Player of the Match: {match['player_of_match']}")
    st.text(f"Umpires: {match['umpire1']}, {match['umpire2']}")



def toss_impact_analysis(df_matches):
    st.header("Toss Impact Analysis")
    toss_winner_wins = (df_matches['toss_winner'] == df_matches['winner']).mean() * 100
    st.write(f"Percentage of times toss winner also won the match: {toss_winner_wins:.2f}%")
    toss_decision_wins = df_matches.groupby('toss_decision')['winner'].count()
    fig, ax = plt.subplots()
    ax.pie(toss_decision_wins, labels=toss_decision_wins.index, autopct='%1.1f%%')
    ax.set_title("Toss Decision and Wins")
    st.pyplot(fig)
    fig, ax = plt.subplots()
    sns.countplot(data=df_matches, x='toss_decision', hue='winner', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Toss Decision vs Match Winner")
    st.pyplot(fig)
    st.write("Number of matches won by each toss decision:")
    st.write(toss_decision_wins)

    st.subheader("Toss Decision by Venue")
    venue_toss = df_matches.groupby('venue')['toss_decision'].value_counts().unstack(fill_value=0).sort_values(by='field', ascending=False)
    st.write(venue_toss)
    fig_venue_toss, ax_venue_toss = plt.subplots(figsize=(10, 6))
    venue_toss.plot(kind='bar', stacked=True, ax=ax_venue_toss)
    ax_venue_toss.set_title("Toss Decision Distribution by Venue")
    ax_venue_toss.set_xlabel("Venue")
    ax_venue_toss.set_ylabel("Number of Tosses")
    st.pyplot(fig_venue_toss)



def venue_impact_analysis(df_matches):
    st.header("Venue Impact Analysis")
    venue_wins = df_matches['venue'].value_counts().head(10)
    fig, ax = plt.subplots()
    venue_wins.plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Venues by Number of Matches")
    ax.set_xlabel("Venue")
    ax.set_ylabel("Number of Matches")
    st.pyplot(fig)
    st.write("Number of matches played at each venue:")
    st.write(df_matches['venue'].value_counts())

    st.subheader("Wins by Team at Each Venue")
    venue_team_wins = df_matches.groupby('venue')['winner'].value_counts().unstack(fill_value=0)
    st.write(venue_team_wins)
    fig_venue_team, ax_venue_team = plt.subplots(figsize=(12, 8))
    sns.heatmap(venue_team_wins, cmap='YlGnBu', annot=True, fmt='g', ax=ax_venue_team)
    ax_venue_team.set_title("Wins by Team at Each Venue")
    st.pyplot(fig_venue_team)

def seasonal_analysis(df_matches):
    st.header("Seasonal Analysis")
    matches_per_season = df_matches['season'].value_counts().sort_index()
    st.write("Number of matches per season:")
    st.write(matches_per_season)
    fig, ax = plt.subplots()
    matches_per_season.plot(kind='bar', ax=ax)
    ax.set_title("Matches per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Matches")
    st.pyplot(fig)

    winners_per_season = df_matches.groupby('season')['winner'].value_counts().unstack(fill_value=0)
    st.write("Winners per season:")
    st.write(winners_per_season)
    fig, ax = plt.subplots(figsize=(10, 6))
    winners_per_season.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Winners per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Wins")
    plt.legend(title='Team')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Most Successful Teams Over All Seasons")
    overall_winners = df_matches['winner'].value_counts()
    st.write("Most Successful Teams Over All Seasons:")
    st.write(overall_winners)
    fig_overall_win, ax_overall_win = plt.subplots()
    overall_winners.plot(kind='bar', ax=ax_overall_win)
    ax_overall_win.set_title("Most Successful Teams Over All Seasons")
    ax_overall_win.set_xlabel("Team")
    ax_overall_win.set_ylabel("Total Wins")
    st.pyplot(fig_overall_win)



def player_of_match_analysis(df_matches):
    st.header("Player of the Match Analysis")
    top_pom = df_matches['player_of_match'].value_counts().head(10)
    st.write("Top 10 Player of the Match Winners:")
    st.write(top_pom)
    fig, ax = plt.subplots()
    top_pom.plot(kind='bar', ax=ax)
    ax.set_title("Top 10 Player of the Match Winners")
    ax.set_xlabel("Player")
    ax.set_ylabel("Number of Awards")
    st.pyplot(fig)

    st.subheader("Player of the Match Awards Season-wise")
    pom_season = df_matches.groupby('season')['player_of_match'].value_counts().unstack(fill_value=0)
    st.write("Player of the Match Awards per Season:")
    st.write(pom_season)
    fig_pom_season, ax_pom_season = plt.subplots(figsize=(12, 8))
    sns.heatmap(pom_season, cmap='viridis', annot=True, fmt='g', ax=ax_pom_season)
    ax_pom_season.set_title("Player of the Match Awards per Season")
    ax_pom_season.set_xlabel("Season")
    ax_pom_season.set_ylabel("Player")
    st.pyplot(fig_pom_season)



def most_successful_team(df_matches):
    st.header("Most Successful Team Analysis")
    team_wins = df_matches['winner'].value_counts().reset_index()
    team_wins.columns = ['Team', 'Wins']
    most_successful = team_wins.iloc[0]

    st.write(f"The most successful team in IPL so far is **{most_successful['Team']}** with **{most_successful['Wins']}** wins.")

    fig, ax = plt.subplots()
    sns.barplot(data=team_wins, x='Wins', y='Team', ax=ax)
    ax.set_title("Team Wins in IPL")
    ax.set_xlabel("Number of Wins")
    ax.set_ylabel("Team")
    st.pyplot(fig)

def season_performance(df_matches):
    st.header("Team Performance by Season")
    teams = sorted(list(set(df_matches['team1'].unique()) | set(df_matches['team2'].unique())))
    selected_team = st.selectbox("Select a Team", teams)

    team_seasons = df_matches[((df_matches['team1'] == selected_team) | (df_matches['team2'] == selected_team))]
    team_season_wins = team_seasons.groupby('season')['winner'].apply(lambda x: (x == selected_team).sum()).reset_index(name='wins')

    st.subheader(f"{selected_team}'s Performance Over the Seasons")
    st.write(team_season_wins)

    fig, ax = plt.subplots()
    sns.lineplot(data=team_season_wins, x='season', y='wins', marker='o', ax=ax)
    ax.set_title(f"{selected_team}'s Wins per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Wins")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def head_to_head_comparison(df_matches):
    st.header("Head-to-Head Team Comparison")
    teams = sorted(df_matches['team1'].unique())
    team1 = st.selectbox("Select Team 1", teams)
    team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])

    if team1 and team2:
        df_team1_vs_team2 = df_matches[((df_matches['team1'] == team1) & (df_matches['team2'] == team2)) | ((df_matches['team1'] == team2) & (df_matches['team2'] == team1))]
        total_matches = df_team1_vs_team2.shape[0]
        wins_team1 = df_team1_vs_team2[df_team1_vs_team2['winner'] == team1].shape[0]
        wins_team2 = df_team1_vs_team2[df_team1_vs_team2['winner'] == team2].shape[0]
        ties = df_team1_vs_team2[df_team1_vs_team2['result'] == 'tie'].shape[0]
        no_result = df_team1_vs_team2[df_team1_vs_team2['result'] == 'no result'].shape[0]

        st.subheader(f"Head-to-Head Record: {team1} vs {team2}")
        st.write(f"Total Matches Played: {total_matches}")
        st.write(f"Wins for {team1}: {wins_team1}")
        st.write(f"Wins for {team2}: {wins_team2}")
        st.write(f"Ties: {ties}")
        st.write(f"No Result: {no_result}")

        labels = [team1, team2, 'Tie', 'No Result']
        sizes = [wins_team1, wins_team2, ties, no_result]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightgray']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        st.subheader("Match Results Over Seasons")
        df_h2h_with_season = df_team1_vs_team2.groupby('season')['winner'].value_counts().unstack(fill_value=0)
        st.write(df_h2h_with_season)
        fig_seasonal, ax_seasonal = plt.subplots(figsize=(10, 6))
        df_h2h_with_season[[team1, team2]].plot(kind='bar', stacked=False, ax=ax_seasonal)
        ax_seasonal.set_title(f"Match Results Between {team1} and {team2} Over Seasons")
        ax_seasonal.set_xlabel("Season")
        ax_seasonal.set_ylabel("Number of Wins")
        plt.xticks(rotation=45)
        plt.legend(title='Winner')
        st.pyplot(fig_seasonal)

def get_phase(over):
    if 0 <= over <= 5:
        return "Powerplay (0-6)"
    elif 6 < over <= 15:
        return "Middle Overs (7-15)"
    elif 15 < over <= 20:
        return "Death Overs (16-20)"
    return None

def phase_wise_analysis(df_deliveries, df_matches):
    st.header("Phase-wise Analysis (Powerplay, Middle, Death)")

    df_deliveries['phase'] = df_deliveries['over'].apply(get_phase)
    df_deliveries_filtered = df_deliveries.dropna(subset=['phase'])

    match_ids = sorted(df_deliveries_filtered['match_id'].unique())
    selected_match_id = st.selectbox("Select a Match ID", match_ids)

    df_match = df_deliveries_filtered[df_deliveries_filtered['match_id'] == selected_match_id].copy()

    if df_match.empty:
        st.warning("No delivery data found for the selected match.")
        return

    teams_in_match = sorted(df_match['batting_team'].unique())
    selected_team = st.selectbox("Select a Batting Team", teams_in_match)

    df_team_match = df_match[df_match['batting_team'] == selected_team].copy()

    if df_team_match.empty:
        st.warning(f"No batting data found for {selected_team} in the selected match.")
        return

    st.subheader(f"Phase-wise Analysis for {selected_team} in Match ID: {selected_match_id}")

    phase_stats = df_team_match.groupby('phase').agg(
        total_runs=('batter_runs', 'sum'),
        total_balls=('ball', 'count'),
        total_wickets=('is_wicket', 'sum')
    ).reset_index()

    phase_stats['run_rate'] = (phase_stats['total_runs'] / (phase_stats['total_balls'] / 6)).round(2)

    st.write(phase_stats)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.barplot(data=phase_stats, x='phase', y='total_runs', ax=axes[0])
    axes[0].set_title("Runs Scored in Each Phase")
    axes[0].set_xlabel("Phase")
    axes[0].set_ylabel("Total Runs")

    sns.barplot(data=phase_stats, x='phase', y='run_rate', ax=axes[1])
    axes[1].set_title("Average Run Rate in Each Phase")
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("Run Rate")

    sns.barplot(data=phase_stats, x='phase', y='total_wickets', ax=axes[2])
    axes[2].set_title("Wickets Lost in Each Phase")
    axes[2].set_xlabel("Phase")
    axes[2].set_ylabel("Total Wickets")

    st.pyplot(fig)

    st.subheader("Match-wise Phase Analysis (All Teams)")
    all_match_phase_stats = df_deliveries_filtered.groupby(['match_id', 'batting_team', 'phase']).agg(
        total_runs=('batter_runs', 'sum'),
        total_balls=('ball', 'count'),
        total_wickets=('is_wicket', 'sum')
    ).reset_index()
    all_match_phase_stats['run_rate'] = (all_match_phase_stats['total_runs'] / (all_match_phase_stats['total_balls'] / 6)).round(2)
    st.write(all_match_phase_stats[all_match_phase_stats['match_id'] == selected_match_id])

def stadium_wise_performance(df_matches):
    st.header("Stadium-wise Team Performance")

    venue_wins = df_matches.groupby(['venue', 'winner']).size().unstack(fill_value=0)
    st.subheader("Team Wins at Each Venue")
    st.write(venue_wins)

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))
    sns.heatmap(venue_wins, annot=True, cmap='YlGnBu', fmt='g', ax=ax_heatmap)
    ax_heatmap.set_title("Team Wins at Each Venue")
    ax_heatmap.set_xlabel("Winning Team")
    ax_heatmap.set_ylabel("Venue")
    st.pyplot(fig_heatmap)

    st.subheader("Dominant Teams at Each Venue")
    dominant_teams = venue_wins.idxmax(axis=1)
    dominant_wins = venue_wins.max(axis=1)
    dominant_df = pd.DataFrame({'Dominant Team': dominant_teams, 'Number of Wins': dominant_wins})
    st.write(dominant_df)

    fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
    dominant_teams.value_counts().plot(kind='bar', ax=ax_bar)
    ax_bar.set_title("Number of Venues Dominated by Each Team")
    ax_bar.set_xlabel("Team")
    ax_bar.set_ylabel("Number of Venues")
    st.pyplot(fig_bar)

    st.subheader("Wins per Team at Different Venues")
    teams = sorted(df_matches['team1'].unique())
    selected_team = st.selectbox("Select a Team to See Venue-wise Performance", teams)
    team_venue_wins = df_matches[df_matches['winner'] == selected_team]['venue'].value_counts().sort_values(ascending=False)
    st.write(f"Wins for {selected_team} at different venues:")
    st.write(team_venue_wins)
    fig_team_venue, ax_team_venue = plt.subplots(figsize=(10, 6))
    team_venue_wins.plot(kind='bar', ax=ax_team_venue)
    ax_team_venue.set_title(f"Wins for {selected_team} at Different Venues")
    ax_team_venue.set_xlabel("Venue")
    ax_team_venue.set_ylabel("Number of Wins")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_team_venue)

    

def main():
    st.title("IPL Data Analysis (2008–2024)")

    df_deliveries, df_matches = load_data()
    if df_deliveries is None or df_matches is None:
        return

    st.sidebar.title("Navigation")
    options = st.sidebar.radio(
        "Go to",
        [
            "Batter Analysis",
            "Bowler Analysis",
            "Team Wins Over Years",
            "Match Summary",
            "Toss Impact Analysis",
            "Venue Impact Analysis",
            "Seasonal Analysis",
            "Player of the Match Analysis",
            "Most Successful Team",
            "Team Performance by Season",
            "Head-to-Head Team Comparison",
            "Phase-wise Analysis (Powerplay, Middle, Death)",
            "Stadium-wise Team Performance"
        ],
    )
    if options == "Batter Analysis":
        batter_analysis(df_deliveries, df_matches)
    elif options == "Bowler Analysis":
        bowler_analysis(df_deliveries,df_matches)
    elif options == "Team Wins Over Years":
        team_wins_over_years(df_matches)
    elif options == "Match Summary":
        match_summary(df_matches)
    elif options == "Toss Impact Analysis":
        toss_impact_analysis(df_matches)
    elif options == "Venue Impact Analysis":
        venue_impact_analysis(df_matches)
    elif options == "Seasonal Analysis":
        seasonal_analysis(df_matches)
    elif options == "Player of the Match Analysis":
        player_of_match_analysis(df_matches)
    elif options == "Most Successful Team":
        most_successful_team(df_matches)
    elif options == "Most Successful Team":
        most_successful_team(df_matches)
    elif options == "Team Performance by Season":
        season_performance(df_matches)
    elif options == "Head-to-Head Team Comparison":
        head_to_head_comparison(df_matches)
    elif options == "Phase-wise Analysis (Powerplay, Middle, Death)":
        phase_wise_analysis(df_deliveries, df_matches)
    elif options == "Stadium-wise Team Performance":
        stadium_wise_performance(df_matches)
if __name__ == "__main__":
    main()
        


