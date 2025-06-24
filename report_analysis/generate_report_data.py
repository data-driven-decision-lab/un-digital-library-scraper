import pandas as pd
import os

# Create output directory if it doesn't exist
output_dir = 'report_analysis'
os.makedirs(output_dir, exist_ok=True)

# --- 1. Rank Changes ---
print("--- Calculating Rank Changes ---")
try:
    df_scores = pd.read_csv('un_report_api/app/required_csvs/annual_scores.csv')
    df_scores = df_scores.sort_values(by=['Country name', 'Year'])

    # --- 1a. Annual Rank Changes (Top 5 Gainers/Losers per year) ---
    df_scores['rank_change'] = df_scores.groupby('Country name')['Overall Rank'].diff()
    df_scores['initial_rank'] = df_scores['Overall Rank'] - df_scores['rank_change']
    df_scores.dropna(subset=['rank_change'], inplace=True)
    df_scores['rank_change'] = df_scores['rank_change'].astype(int)
    df_scores['initial_rank'] = df_scores['initial_rank'].astype(int)

    annual_changes_list = []
    for year, group in df_scores.groupby('Year'):
        # Ensure we don't select the same rows if a gainer is also a loser (e.g. rank_change is 0)
        # and there are few data points. Unlikely but good practice.
        losers = group.nlargest(5, 'rank_change')
        gainers = group.nsmallest(5, 'rank_change')
        
        gainers['Change_Type'] = 'Gainer'
        losers['Change_Type'] = 'Loser'
        annual_changes_list.append(gainers)
        annual_changes_list.append(losers)

    if annual_changes_list:
        annual_rank_changes = pd.concat(annual_changes_list)
        annual_rank_changes = annual_rank_changes[['Year', 'Change_Type', 'Country name', 'initial_rank', 'Overall Rank', 'rank_change']]
        annual_rank_changes.rename(columns={'Overall Rank': 'ending_rank'}, inplace=True)
        annual_rank_changes.sort_values(by=['Year', 'Change_Type', 'rank_change'], ascending=[True, True, True], inplace=True)
        annual_rank_changes.to_csv(os.path.join(output_dir, 'annual_top5_rank_changes.csv'), index=False)
        print("Successfully calculated annual top 5 rank changes.")

    # --- 1b. 5-Year Period Rank Changes ---
    min_year = df_scores['Year'].min()
    max_year = df_scores['Year'].max()
    
    start_year_period = (min_year // 5) * 5
    periods = range(start_year_period, max_year + 5, 5)
    labels = [f"{p}-{p+4}" for p in periods[:-1]]
    
    df_scores['period'] = pd.cut(df_scores['Year'], bins=periods, labels=labels, right=False, include_lowest=True)
    df_scores.dropna(subset=['period'], inplace=True)
    
    # Filter out empty groups before finding idxmin/idxmax
    g = df_scores.groupby(['Country name', 'period'], observed=True)
    first_ranks = df_scores.loc[g['Year'].idxmin()].reset_index()
    last_ranks = df_scores.loc[g['Year'].idxmax()].reset_index()

    merged_ranks = pd.merge(
        first_ranks[['Country name', 'period', 'Year', 'Overall Rank']],
        last_ranks[['Country name', 'period', 'Year', 'Overall Rank']],
        on=['Country name', 'period']
    )
    
    merged_ranks = merged_ranks[merged_ranks['Year_x'] != merged_ranks['Year_y']]
    
    merged_ranks.rename(columns={'Year_x': 'start_year', 'Overall Rank_x': 'initial_rank', 'Year_y': 'end_year', 'Overall Rank_y': 'ending_rank'}, inplace=True)
    merged_ranks['rank_change'] = merged_ranks['ending_rank'] - merged_ranks['initial_rank']

    period_changes_list = []
    # Convert 'period' to string to avoid issues with categorical type in groupby
    merged_ranks['period'] = merged_ranks['period'].astype(str)
    for period, group in merged_ranks.groupby('period'):
        gainers = group.nsmallest(5, 'rank_change')
        losers = group.nlargest(5, 'rank_change')
        gainers['Change_Type'] = 'Gainer'
        losers['Change_Type'] = 'Loser'
        period_changes_list.append(gainers)
        period_changes_list.append(losers)

    if period_changes_list:
        five_year_rank_changes = pd.concat(period_changes_list)
        five_year_rank_changes = five_year_rank_changes[['period', 'Change_Type', 'Country name', 'initial_rank', 'ending_rank', 'rank_change', 'start_year', 'end_year']]
        five_year_rank_changes.sort_values(by=['period', 'Change_Type', 'rank_change'], ascending=[True, True, True], inplace=True)
        five_year_rank_changes.to_csv(os.path.join(output_dir, 'five_year_top5_rank_changes.csv'), index=False)
        print("Successfully calculated 5-year period rank changes.")

except FileNotFoundError:
    print("annual_scores.csv not found.")
except Exception as e:
    print(f"An error occurred during rank change calculation: {e}")

# --- 2. Topic Alignment & Voting ---
print("\n--- Calculating Topic Metrics ---")
try:
    df_topics = pd.read_csv('un_report_api/app/required_csvs/topic_votes_yearly.csv')

    # --- 2a. Annual Topic Alignment ---
    annual_topic_votes = df_topics.groupby(['Year', 'TopicTag']).agg(
        total_yes=('YesVotes_Topic', 'sum'),
        total_votes=('TotalVotes_Topic', 'sum')
    ).reset_index()

    annual_topic_votes = annual_topic_votes[annual_topic_votes['total_votes'] > 0]
    annual_topic_votes['yes_vote_percentage'] = (annual_topic_votes['total_yes'] / annual_topic_votes['total_votes']) * 100
    
    annual_topic_alignment = annual_topic_votes.sort_values(by=['Year', 'yes_vote_percentage'], ascending=[True, False])
    annual_topic_alignment.to_csv(os.path.join(output_dir, 'annual_topic_alignment.csv'), index=False)
    print("Successfully calculated annual topic alignment.")

    # --- 2b. 5-Year Period Topic Alignment ---
    min_year_topic = df_topics['Year'].min()
    max_year_topic = df_topics['Year'].max()
    
    start_year_topic_period = (min_year_topic // 5) * 5
    topic_periods = range(start_year_topic_period, max_year_topic + 5, 5)
    topic_labels = [f"{p}-{p+4}" for p in topic_periods[:-1]]

    df_topics['period'] = pd.cut(df_topics['Year'], bins=topic_periods, labels=topic_labels, right=False, include_lowest=True)
    df_topics.dropna(subset=['period'], inplace=True)

    five_year_topic_votes = df_topics.groupby(['period', 'TopicTag']).agg(
        total_yes=('YesVotes_Topic', 'sum'),
        total_votes=('TotalVotes_Topic', 'sum')
    ).reset_index()

    five_year_topic_votes = five_year_topic_votes[five_year_topic_votes['total_votes'] > 0]
    five_year_topic_votes['yes_vote_percentage'] = (five_year_topic_votes['total_yes'] / five_year_topic_votes['total_votes']) * 100
    
    five_year_topic_alignment = five_year_topic_votes.sort_values(by=['period', 'yes_vote_percentage'], ascending=[True, False])
    five_year_topic_alignment.to_csv(os.path.join(output_dir, 'five_year_topic_alignment.csv'), index=False)
    print("Successfully calculated 5-year period topic alignment.")
    
    # --- 2c. Most Voted Topics (Overall) ---
    most_voted = df_topics.groupby('TopicTag')['TotalVotes_Topic'].sum().reset_index()
    most_voted = most_voted.sort_values(by='TotalVotes_Topic', ascending=False)
    most_voted.to_csv(os.path.join(output_dir, 'most_voted_topics.csv'), index=False)
    print("Successfully calculated most voted topics (overall).")

except FileNotFoundError:
    print("topic_votes_yearly.csv not found.")
except Exception as e:
    print(f"An error occurred during topic alignment calculation: {e}")

# --- 3. Average vote percentages (Overall) ---
print("\n--- Calculating Overall Vote Percentages ---")
try:
    output_files = [f for f in os.listdir('pipeline_output') if f.startswith('UN_VOTING_DATA_RAW_WITH_TAGS_') and f.endswith('.csv')]
    if not output_files:
        raise FileNotFoundError
    latest_file = sorted(output_files)[-1]
    
    df_resolutions = pd.read_csv(os.path.join('pipeline_output', latest_file), low_memory=False)
    
    df_resolutions = df_resolutions[df_resolutions['TOTAL VOTES'] > 0].copy()
    
    df_resolutions['yes_vote_percentage'] = (df_resolutions['YES COUNT'] / df_resolutions['TOTAL VOTES']) * 100
    df_resolutions['abstain_vote_percentage'] = (df_resolutions['ABSTAIN COUNT'] / df_resolutions['TOTAL VOTES']) * 100
    
    avg_yes_percentage = df_resolutions['yes_vote_percentage'].mean()
    avg_abstain_percentage = df_resolutions['abstain_vote_percentage'].mean()
    
    avg_percentages = pd.DataFrame({
        'Metric': ['Average Yes % on Resolution', 'Average Abstention % on Resolution'],
        'Value': [avg_yes_percentage, avg_abstain_percentage]
    })
    avg_percentages.to_csv(os.path.join(output_dir, 'average_vote_percentages.csv'), index=False)
    
    print("Successfully calculated average vote percentages.")
except FileNotFoundError:
    print("No voting data file found in pipeline_output.")
except Exception as e:
    print(f"An error occurred during average vote percentage calculation: {e}")

print(f"\nAll analysis files have been saved in the '{output_dir}' directory.") 