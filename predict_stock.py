import yfinance as yf
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from textblob import TextBlob
import praw

# This is used to access the reddit API
reddit = praw.Reddit(
    client_id = "8ArtuteIFTcyL0tW27Py5A",
    client_secret = "3tUD3HMFHQbKtuQnqpndmpNgfzDA2Q",
    user_agent = "Yourboyfatjoe10"
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
        for post in reddit.subreddit(sub).search(ticker, limit=limit, sort="relevence"):
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
data.reset_index(inplace=True)
data["Date"] = pd.to_datetime(data["Date"]).dt.date.astype(str) 
sentiment_daily = sentiment_daily.reset_index(drop=True) 
merged_df = data.merge(sentiment_daily, left_on="Date", right_on="date", how="left")
merged_df.drop(columns=["date"], inplace=True)
merged_df.dropna(subset=["sentiment"], inplace=True) 


# Feature engineering
merged_df["Return"] = merged_df["Close"].pct_change()
merged_df["MA5"] = merged_df["Close"].rolling(window=5).mean()
merged_df["MA10"] = merged_df["Close"].rolling(window=10).mean()
merged_df["Momentum"] = merged_df["Close"] - merged_df["Close"].shift(5)
merged_df["Volatility"] = merged_df["Return"].rolling(window=5).std()
merged_df["Sentiment"] = merged_df["sentiment"].fillna(method="ffill", limit=5)
merged_df["Target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)


# Prepare inputs
features = ["Return", "MA5", "MA10", "Momentum", "Volatility", "Sentiment"]
X = merged_df[features]
Y = merged_df["Target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.3)

model = RandomForestClassifier(class_weight="balanced", random_state=10)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

data_test = merged_df.iloc[len(X_train):].copy()
data_test["Predicted"] = y_pred

print(Y.value_counts())

print(sentiment_daily["date"].min(), sentiment_daily["date"].max())
print(sentiment_daily["date"].nunique())

