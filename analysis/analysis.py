# Analysis script for tasks: Level 1 + Level 3
# Produces CSV summary files and PNG plots in the analysis/output folder.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set(style="whitegrid")

ROOT = os.path.dirname(__file__)
# Support multiple dataset directories and filename variants
DATA_DIR_CANDIDATES = [
    os.path.join(ROOT, '..', 'Data analysis dataset'),
    os.path.join(ROOT, '..', 'DATA ANALYSIS TASK LIST AND DATASET'),
]
DATA_FILENAMES = ['Dataset .csv', 'dataset.csv', 'Dataset.csv']
DATA_PATH = None
for d in DATA_DIR_CANDIDATES:
    for name in DATA_FILENAMES:
        candidate = os.path.join(d, name)
        if os.path.exists(candidate):
            DATA_PATH = candidate
            break
    if DATA_PATH:
        break
if DATA_PATH is None:
    raise FileNotFoundError('Dataset file not found. Looked for variants in: ' + '\n'.join(DATA_DIR_CANDIDATES))
OUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# Mode control: FULL_MODE=1 keeps all detailed artifacts; otherwise only core deliverables
FULL_MODE = os.getenv('FULL_MODE', '0') == '1'
CORE_CSV = {
    'level1_city_analysis.csv',
    'level1_price_range_distribution.csv',
    'level1_online_delivery_summary.csv',
    'level1_online_vs_offline_rating.csv',
    'level3_price_range_vs_delivery_table.csv',
    'level1_top_cuisines.csv'
}
CORE_PNG = {
    'level1_top3_cuisines.png',
    'level1_city_most_restaurants.png',
    'level1_price_range_distribution.png',
    'level1_online_vs_offline_rating_bar.png',
    'level3_votes_vs_rating.png',
    'level3_price_range_vs_services.png'
}
if not FULL_MODE:
    # Clean any non-core artifacts
    for fname in os.listdir(OUT_DIR):
        if fname not in CORE_CSV and fname not in CORE_PNG:
            try:
                os.remove(os.path.join(OUT_DIR, fname))
            except OSError:
                pass

print('Loading dataset...', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Rows:', len(df), '(FULL_MODE=' + str(FULL_MODE) + ')')

# Basic cleaning
# Standardize column names
cols = [c.strip() for c in df.columns]
df.columns = cols

# Convert numeric columns
for c in ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Task: Level 1 - Top cuisines and percentages
# Note: 'Cuisines' can contain multiple cuisines separated by commas
cuisines_series = df['Cuisines'].dropna().astype(str)
all_cuisines = cuisines_series.str.split(',').explode().str.strip()
cuisine_counts = all_cuisines.value_counts()
cuisine_pct = (cuisine_counts / len(df) * 100).round(2)
cuisine_summary = pd.DataFrame({'count': cuisine_counts, 'percent_of_restaurants': cuisine_pct})
cuisine_summary.to_csv(os.path.join(OUT_DIR, 'level1_top_cuisines.csv'))

# Save top 10 cuisines & plot (full mode only)
if FULL_MODE:
    cuisine_summary.head(10).to_csv(os.path.join(OUT_DIR, 'level1_top10_cuisines.csv'))
    plt.figure(figsize=(10,6))
    sns.barplot(x=cuisine_summary.head(10).percent_of_restaurants.values, y=cuisine_summary.head(10).index, palette='viridis')
    plt.xlabel('Percent of restaurants (%)')
    plt.title('Top 10 Cuisines by % of Restaurants')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'level1_top10_cuisines.png'))
    plt.close()
# Plot top 3 cuisines
plt.figure(figsize=(7,4))
sns.barplot(x=cuisine_summary.head(3).percent_of_restaurants.values, y=cuisine_summary.head(3).index, palette='crest')
plt.xlabel('Percent of restaurants (%)')
plt.title('Top 3 Cuisines by % of Restaurants')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'level1_top3_cuisines.png'))
plt.close()

# Task: Level 1 - City analysis (count restaurants per city and average rating)
city_group = df.groupby('City').agg(restaurants_count=('Restaurant ID', 'count'), average_rating=('Aggregate rating', 'mean'))
city_group = city_group.sort_values('restaurants_count', ascending=False)
city_group.to_csv(os.path.join(OUT_DIR, 'level1_city_analysis.csv'))

# Top 10 cities by number of restaurants
city_group.head(10).to_csv(os.path.join(OUT_DIR, 'level1_top10_cities_by_count.csv'))
# Top/bottom cities by average rating (only include cities with >= 5 restaurants)
city_rating_filtered = city_group[city_group['restaurants_count'] >= 5].sort_values('average_rating', ascending=False)
city_rating_filtered.head(10).to_csv(os.path.join(OUT_DIR, 'level1_top10_cities_by_rating.csv'))
city_rating_filtered.tail(10).to_csv(os.path.join(OUT_DIR, 'level1_bottom10_cities_by_rating.csv'))
# Plot restaurants per city (top 15) - full mode only
if FULL_MODE:
    plt.figure(figsize=(10,8))
    sns.barplot(x=city_group.head(15).restaurants_count.values, y=city_group.head(15).index, palette='magma')
    plt.xlabel('Number of restaurants')
    plt.title('Top 15 Cities by Number of Restaurants')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'level1_top15_cities_count.png'))
    plt.close()
# Plot city with highest number of restaurants
plt.figure(figsize=(6,3))
top_city = city_group.head(1)
sns.barplot(x=top_city.restaurants_count.values, y=top_city.index, palette='Blues')
plt.xlabel('Number of restaurants')
plt.title('City with Most Restaurants')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'level1_city_most_restaurants.png'))
plt.close()
# Plot city with highest average rating (min 5 restaurants) - full mode only
if FULL_MODE:
    plt.figure(figsize=(6,3))
    top_rating_city = city_rating_filtered.head(1)
    sns.barplot(x=top_rating_city.average_rating.values, y=top_rating_city.index, palette='Greens')
    plt.xlabel('Average Rating')
    plt.title('City with Highest Average Rating (≥5 restaurants)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'level1_city_highest_avg_rating.png'))
    plt.close()

# Task: Level 1 - Price range distribution
if 'Price range' in df.columns:
    price_counts = df['Price range'].value_counts().sort_index()
    price_pct = (price_counts / len(df) * 100).round(2)
    price_df = pd.DataFrame({'count': price_counts, 'percent': price_pct})
    price_df.to_csv(os.path.join(OUT_DIR, 'level1_price_range_distribution.csv'))
    # Plot
    plt.figure(figsize=(8,5))
    sns.barplot(x=price_df.index.astype(str), y=price_df['percent'], palette='coolwarm')
    plt.xlabel('Price range')
    plt.ylabel('Percent of restaurants (%)')
    plt.title('Price Range Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'level1_price_range_distribution.png'))
    plt.close()
    # Pie chart for price range (full mode only)
    if FULL_MODE:
        plt.figure(figsize=(6,6))
        plt.pie(price_df['percent'], labels=price_df.index.astype(str), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('coolwarm', len(price_df)))
        plt.title('Price Range Distribution (Pie)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'level1_price_range_pie.png'))
        plt.close()
else:
    print('Price range column not found')

# Task: Level 1 - Online delivery analysis
if 'Has Online delivery' in df.columns:
    online_counts = df['Has Online delivery'].value_counts()
    online_pct = (online_counts / len(df) * 100).round(2)
    online_summary = pd.DataFrame({'count': online_counts, 'percent': online_pct})
    online_summary.to_csv(os.path.join(OUT_DIR, 'level1_online_delivery_summary.csv'))
    if FULL_MODE:
        # Bar plot for online delivery percentage
        plt.figure(figsize=(6,4))
        sns.barplot(x=online_summary.index, y=online_summary['percent'], palette='Set2')
        plt.ylabel('Percent of restaurants (%)')
        plt.xlabel('Has Online Delivery')
        plt.title('Online Delivery Availability')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'level1_online_delivery_bar.png'))
        plt.close()
        # Pie chart for online delivery
        plt.figure(figsize=(5,5))
        plt.pie(online_summary['percent'], labels=online_summary.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2', len(online_summary)))
        plt.title('Online Delivery Availability (Pie)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'level1_online_delivery_pie.png'))
        plt.close()
    # Compare average ratings
    df_online = df[df['Has Online delivery'].str.strip().str.lower() == 'yes']
    df_offline = df[df['Has Online delivery'].str.strip().str.lower() != 'yes']
    avg_online = df_online['Aggregate rating'].mean()
    avg_offline = df_offline['Aggregate rating'].mean()
    pd.DataFrame({'group': ['online', 'offline'], 'avg_rating': [avg_online, avg_offline]}).to_csv(os.path.join(OUT_DIR, 'level1_online_vs_offline_rating.csv'), index=False)
    # Bar plot for average ratings by delivery
    plt.figure(figsize=(6,4))
    sns.barplot(x=['Online Delivery', 'No Online Delivery'], y=[avg_online, avg_offline], palette='Set1')
    plt.ylabel('Average Rating')
    plt.title('Average Ratings: Online Delivery vs Not')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'level1_online_vs_offline_rating_bar.png'))
    plt.close()
else:
    print('Has Online delivery column not found')

# Mark Level 1 tasks summary (full mode only)
if FULL_MODE:
    with open(os.path.join(OUT_DIR, 'level1_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('Level 1 summary\n')
        f.write('Top cuisines saved to level1_top_cuisines.csv and level1_top10_cuisines.csv\n')
        f.write('City analysis saved to level1_city_analysis.csv\n')
        f.write('Price range distribution saved to level1_price_range_distribution.csv\n')
        f.write('Online delivery summary saved to level1_online_delivery_summary.csv and level1_online_vs_offline_rating.csv\n')

# Task: Level 3 - Reviews keywords & length vs rating
# Note: This dataset doesn't include full review text column. We'll check for 'Reviews' or 'Review' columns.
review_cols = [c for c in df.columns if 'review' in c.lower()]
if review_cols:
    print('Found review columns:', review_cols)
    reviews = df[review_cols[0]].dropna().astype(str)
    reviews_len = reviews.str.len()
    # Simple keyword frequency
    words = reviews.str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.split().explode()
    stopwords = set(['the','and','is','to','a','of','it','for','in','was','on','with','that','this','they','but'])
    words = words[~words.isin(stopwords)]
    top_words = words.value_counts().head(20)
    top_words.to_csv(os.path.join(OUT_DIR, 'level3_top_review_keywords.csv'))
    # Bar plot for top review keywords
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_words.values, y=top_words.index, palette='flare')
    plt.xlabel('Frequency')
    plt.title('Top 20 Review Keywords')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'level3_top_review_keywords.png'))
    plt.close()
    # Correlate review length with Aggregate rating (if ratings present)
    if 'Aggregate rating' in df.columns:
        merged = pd.DataFrame({'review_len': reviews_len, 'rating': df.loc[reviews.index, 'Aggregate rating']}).dropna()
        corr, p = pearsonr(merged['review_len'], merged['rating']) if len(merged) > 1 else (np.nan, np.nan)
        with open(os.path.join(OUT_DIR, 'level3_review_length_rating_corr.txt'), 'w') as f:
            f.write(f'pearson_corr={corr}, pvalue={p}\n')
        # Scatter plot: review length vs rating
        plt.figure(figsize=(7,5))
        sns.scatterplot(x=merged['review_len'], y=merged['rating'], alpha=0.5)
        plt.xlabel('Review Length')
        plt.ylabel('Aggregate Rating')
        plt.title('Review Length vs. Rating')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'level3_review_length_vs_rating.png'))
        plt.close()
else:
    print('No review text column found — skipping review keyword analysis')

# Additional Level 3 Task: Restaurant Reviews (external reviews.csv)
# If a separate reviews.csv file exists, perform positive/negative keyword extraction
# and review length statistics + correlation with rating.
reviews_file = os.path.join(ROOT, '..', 'Data analysis dataset', 'reviews.csv')
if os.path.exists(reviews_file):
    try:
        rev_df = pd.read_csv(reviews_file)
        # Expect at minimum: a text column and optionally a rating column.
        # Heuristically pick first text-like column.
        text_col = None
        for cand in rev_df.columns:
            if rev_df[cand].dtype == object and cand.lower() not in ['city','cuisines']:
                text_col = cand
                break
        if text_col is None:
            print('reviews.csv found but no suitable text column detected.')
        else:
            txt = rev_df[text_col].dropna().astype(str)
            # Basic tokenization
            tokens = (txt.str.lower()
                        .str.replace(r'[^a-z\s]', ' ', regex=True)
                        .str.split()
                        .explode())
            stopwords = set(['the','and','is','to','a','of','it','for','in','was','on','with','that','this','they','but','at','as','are','be','from'])
            tokens = tokens[~tokens.isin(stopwords)]
            # Simple sentiment lexicons (can be expanded)
            positive_words = set(['good','great','excellent','amazing','tasty','fresh','friendly','delicious','love','perfect','nice','fast','clean','wonderful'])
            negative_words = set(['bad','slow','rude','cold','dirty','worst','awful','terrible','poor','stale','unfriendly','overpriced','bland'])
            pos_counts = tokens[tokens.isin(positive_words)].value_counts().head(20)
            neg_counts = tokens[tokens.isin(negative_words)].value_counts().head(20)
            if not pos_counts.empty:
                pos_counts.to_csv(os.path.join(OUT_DIR, 'reviews_top_positive_keywords.csv'), header=['count'])
                plt.figure(figsize=(8,4))
                sns.barplot(x=pos_counts.values, y=pos_counts.index, palette='Greens')
                plt.xlabel('Frequency')
                plt.title('Top Positive Keywords')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, 'reviews_top_positive_keywords.png'))
                plt.close()
            if not neg_counts.empty:
                neg_counts.to_csv(os.path.join(OUT_DIR, 'reviews_top_negative_keywords.csv'), header=['count'])
                plt.figure(figsize=(8,4))
                sns.barplot(x=neg_counts.values, y=neg_counts.index, palette='Reds')
                plt.xlabel('Frequency')
                plt.title('Top Negative Keywords')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, 'reviews_top_negative_keywords.png'))
                plt.close()
            # Review length stats & correlation
            lengths = txt.str.len()
            stats = lengths.describe()[['count','mean','std','min','25%','50%','75%','max']]
            stats.to_csv(os.path.join(OUT_DIR, 'reviews_length_stats.csv'))
            # Histogram / distribution plot
            plt.figure(figsize=(8,4))
            sns.histplot(lengths, bins=30, kde=True, color='steelblue')
            plt.xlabel('Review Length (characters)')
            plt.title('Review Length Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, 'reviews_length_distribution.png'))
            plt.close()
            # Correlation with rating if rating column exists
            rating_col = None
            for cand in rev_df.columns:
                if 'rating' in cand.lower():
                    rating_col = cand
                    break
            if rating_col is not None:
                merged = pd.DataFrame({'length': lengths, 'rating': rev_df.loc[lengths.index, rating_col]}).dropna()
                if len(merged) > 1:
                    corr_rev, p_rev = pearsonr(pd.to_numeric(merged['length'], errors='coerce'), pd.to_numeric(merged['rating'], errors='coerce'))
                else:
                    corr_rev, p_rev = (np.nan, np.nan)
                with open(os.path.join(OUT_DIR, 'reviews_length_rating_corr.txt'), 'w') as f:
                    f.write(f'pearson_corr={corr_rev}, pvalue={p_rev}\n')
                plt.figure(figsize=(7,4))
                sns.scatterplot(x=merged['length'], y=merged['rating'], alpha=0.5)
                plt.xlabel('Review Length (chars)')
                plt.ylabel('Rating')
                plt.title('Review Length vs Rating')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, 'reviews_length_vs_rating.png'))
                plt.close()
            # Summary file for review analysis
            with open(os.path.join(OUT_DIR, 'reviews_summary.txt'), 'w') as f:
                f.write('Review Analysis Summary\n')
                f.write(f'Total reviews used: {int(stats["count"])}\n')
                f.write(f'Average length (chars): {stats["mean"]:.2f}\n')
                f.write(f'Positive keywords captured: {len(pos_counts)}\n')
                f.write(f'Negative keywords captured: {len(neg_counts)}\n')
                f.write(f'Correlation length-rating: {corr_rev if not np.isnan(corr_rev) else "NA"}\n')
    except Exception as e:
        print('Failed to process reviews.csv:', e)
else:
    print('reviews.csv not found or skipped')

# Task: Level 3 - Votes analysis
# Highest and lowest votes
votes = df[['Restaurant ID', 'Restaurant Name', 'Votes', 'Aggregate rating']].copy()
votes_sorted = votes.sort_values('Votes', ascending=False)
if FULL_MODE:
    votes_sorted.head(20).to_csv(os.path.join(OUT_DIR, 'level3_top20_by_votes.csv'), index=False)
    votes_sorted.tail(20).to_csv(os.path.join(OUT_DIR, 'level3_bottom20_by_votes.csv'), index=False)

# Correlation between votes and rating
votes_nonnull = votes.dropna(subset=['Votes', 'Aggregate rating'])
if len(votes_nonnull) > 1:
    corr_votes_rating, pval = pearsonr(votes_nonnull['Votes'], votes_nonnull['Aggregate rating'])
else:
    corr_votes_rating, pval = (np.nan, np.nan)
with open(os.path.join(OUT_DIR, 'level3_votes_rating_corr.txt'), 'w') as f:
    f.write(f'pearson_corr={corr_votes_rating}, pvalue={pval}\n')
# Scatter plot: votes vs rating
plt.figure(figsize=(7,5))
sns.scatterplot(x=votes_nonnull['Votes'], y=votes_nonnull['Aggregate rating'], alpha=0.5)
plt.xlabel('Votes')
plt.ylabel('Aggregate Rating')
plt.title('Votes vs. Aggregate Rating')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'level3_votes_vs_rating.png'))
plt.close()

# Task: Level 3 - Price range vs delivery/table booking
cols_of_interest = ['Price range', 'Has Online delivery', 'Has Table booking', 'Aggregate rating']
pr = df[cols_of_interest].copy()
# Normalize online/table booking to lowercase
for c in ['Has Online delivery', 'Has Table booking']:
    if c in pr.columns:
        pr[c] = pr[c].astype(str).str.strip().str.lower()
# Group by price range
if 'Price range' in pr.columns:
    pr_summary = pr.groupby('Price range').agg(count=('Aggregate rating','count'), avg_rating=('Aggregate rating','mean'), pct_online=('Has Online delivery', lambda x: (x=='yes').sum()/len(x)*100 if len(x)>0 else np.nan), pct_table_booking=('Has Table booking', lambda x: (x=='yes').sum()/len(x)*100 if len(x)>0 else np.nan))
    pr_summary.to_csv(os.path.join(OUT_DIR, 'level3_price_range_vs_delivery_table.csv'))
    # Grouped bar: price range vs online delivery/table booking
    plt.figure(figsize=(8,5))
    width = 0.35
    x = np.arange(len(pr_summary.index))
    plt.bar(x - width/2, pr_summary['pct_online'], width, label='Online Delivery')
    plt.bar(x + width/2, pr_summary['pct_table_booking'], width, label='Table Booking')
    plt.xticks(x, pr_summary.index.astype(str))
    plt.xlabel('Price Range')
    plt.ylabel('Percent of Restaurants (%)')
    plt.title('Online Delivery & Table Booking by Price Range')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'level3_price_range_vs_services.png'))
    plt.close()

# Final deliverables list (full mode only)
if FULL_MODE:
    open(os.path.join(OUT_DIR, 'README_generated.txt'), 'w').write('Generated outputs:\n' + '\n'.join(os.listdir(OUT_DIR)))
# Post-run pruning if minimal mode: remove any non-core lingering artifacts
if not FULL_MODE:
    for fname in os.listdir(OUT_DIR):
        if fname.endswith('.csv') and fname not in CORE_CSV:
            try: os.remove(os.path.join(OUT_DIR, fname))
            except OSError: pass
        if fname.endswith('.png') and fname not in CORE_PNG:
            try: os.remove(os.path.join(OUT_DIR, fname))
            except OSError: pass
        if fname.endswith('.txt') and 'rating_corr' in fname and fname.startswith('level3_votes'):
            # Keep correlation? Decide: optional -> remove in minimal mode
            try: os.remove(os.path.join(OUT_DIR, fname))
            except OSError: pass
print('Analysis complete. Outputs in', OUT_DIR)
