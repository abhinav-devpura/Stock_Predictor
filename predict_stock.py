import os

import numpy as np
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from textblob import TextBlob
import praw

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. Create a .env file in the "
            "project root with REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT "
            "(see reddit.com/prefs/apps for credentials)."
        )
    return value


# This is used to access the reddit API
reddit = praw.Reddit(
    client_id=_require_env("REDDIT_CLIENT_ID"),
    client_secret=_require_env("REDDIT_CLIENT_SECRET"),
    user_agent=_require_env("REDDIT_USER_AGENT"),
)

# Used to get the sentiment score polarity
def score_sentiment(post):
    text = post["title"] + " " + post["body"]
    return TextBlob(text).sentiment.polarity

# This is used to get all of the yFinance data for any given ticker
ticker = "NVDA"
data = yf.download(ticker, period = "10y" , auto_adjust=True)
data.columns = data.columns.get_level_values(0)
data = data.reset_index()

# Format to string for clean merging
data["Date"] = pd.to_datetime(data["Date"]).dt.date.astype(str)

# This is used to get a list of all relevent reddit posts
def get_reddit_posts(ticker, limit=5000):
    subreddits = ["stocks", "StockMarket", "investing", "news"]
    posts = []
    for sub in subreddits:
        for post in reddit.subreddit(sub).search(ticker, limit=limit, sort="relevance"):
            posts.append({
                "title": post.title,
                "body": post.selftext,
                "date": str(post.created_utc)[:10]  # UNIX timestamp → date snippet
            })
    return posts

# This is used to get the sentiment from all of those posts
def scrape_reddit_sentiment(ticker):
    posts = get_reddit_posts(ticker)
    data = []
    for post in posts:
        if post["body"]:
            sentiment = score_sentiment(post)
            data.append({
                "date": pd.to_datetime(float(post["date"]), unit="s").date().isoformat(),
                "sentiment": sentiment
            })
    return pd.DataFrame(data)


sentiment_df = scrape_reddit_sentiment(ticker)
sentiment_daily = sentiment_df.groupby("date").mean().reset_index()

# This is used to merge the quantitative data with the sentiment data
merged_df = data.merge(sentiment_daily, left_on="Date", right_on="date", how="left")
merged_df.drop(columns=["date"], inplace=True)

# Feature engineering — computed on the full contiguous daily price series,
# before any row filtering, so rolling windows span actual consecutive
# trading days rather than whatever rows happened to survive a sentiment filter.
merged_df["Return"] = merged_df["Close"].pct_change()
merged_df["MA5"] = merged_df["Close"].rolling(window=5).mean()
merged_df["MA10"] = merged_df["Close"].rolling(window=10).mean()
merged_df["Momentum"] = merged_df["Close"] - merged_df["Close"].shift(5)
merged_df["Volatility"] = merged_df["Return"].rolling(window=5).std()
merged_df["Sentiment"] = merged_df["sentiment"].ffill(limit=5)
merged_df["Target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)

# Prepare inputs
features = ["Return", "MA5", "MA10", "Momentum", "Volatility", "Sentiment"]

# Drop rows with no usable data only now: rolling-window warm-up, no
# sentiment within the forward-fill window, or no next-day price for Target.
merged_df.dropna(subset=features + ["Target"], inplace=True)

X = merged_df[features]
Y = merged_df["Target"]

# Walk-forward cross-validation: several chronological folds instead of a
# single split, each scored against a same-fold majority-class baseline so
# raw accuracy numbers can't be mistaken for skill.
tscv = TimeSeriesSplit(n_splits=5)
fold_accuracies = []
baseline_accuracies = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    Y_train_cv, Y_test_cv = Y.iloc[train_idx], Y.iloc[test_idx]

    fold_model = RandomForestClassifier(class_weight="balanced", random_state=10)
    fold_model.fit(X_train_cv, Y_train_cv)
    fold_pred = fold_model.predict(X_test_cv)
    fold_acc = accuracy_score(Y_test_cv, fold_pred)

    majority_class = Y_train_cv.mode()[0]
    baseline_pred = np.full(len(Y_test_cv), majority_class)
    baseline_acc = accuracy_score(Y_test_cv, baseline_pred)

    fold_accuracies.append(fold_acc)
    baseline_accuracies.append(baseline_acc)
    print(f"Fold {fold}: model accuracy={fold_acc:.3f}, majority-class baseline={baseline_acc:.3f}")

print(f"\nMean walk-forward accuracy: model={np.mean(fold_accuracies):.3f}, "
      f"baseline={np.mean(baseline_accuracies):.3f}")

# Final chronological holdout split for a detailed report
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.3)

model = RandomForestClassifier(class_weight="balanced", random_state=10)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

majority_class = Y_train.mode()[0]
baseline_pred = np.full(len(Y_test), majority_class)

print("\nModel Accuracy:", accuracy_score(Y_test, y_pred))
print("Majority-class Baseline Accuracy:", accuracy_score(Y_test, baseline_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

data_test = merged_df.iloc[len(X_train):].copy()
data_test["Predicted"] = y_pred

print(Y.value_counts())

print(sentiment_daily["date"].min(), sentiment_daily["date"].max())
print(sentiment_daily["date"].nunique())

