import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

LOG_PATTERN = re.compile(
    r'(?P<timestamp>\S+ \S+) ERROR (?P<service>\w+) (?P<function>\w+) (?P<error>\w+) (?P<message>.*)'
)

def parse_logs():
    records = []
    with open("app.log", "r") as file:
        for line in file:
            match = LOG_PATTERN.match(line)
            if match:
                records.append(match.groupdict())
    return records

def main():
    logs = parse_logs()
    df = pd.DataFrame(logs)

    print("\n=== DATAFRAME ===")
    print(df)

        # --- ROOT CAUSE RANKING ---
    ranking = (
        df.groupby(["service", "function"])
          .size()
          .reset_index(name="occurrences")
          .sort_values(by="occurrences", ascending=False)
    )

    print("\n=== ROOT CAUSE RANKING ===")
    print(ranking)

    top = ranking.iloc[0]
    print("\n🔥 MOST LIKELY ROOT CAUSE 🔥")
    print(f"Service  : {top['service']}")
    print(f"Function : {top['function']}")
    print(f"Failures : {top['occurrences']}")


    # --- ML PART STARTS HERE ---
    texts = df["error"] + " " + df["message"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    k = min(3, len(df))
    model = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = model.fit_predict(X)

    print("\n=== CLUSTERED ERRORS ===")
    print(df[["service", "function", "error", "message", "cluster"]])

if __name__ == "__main__":
    main()
