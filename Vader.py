import re, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
RJ_URL = "https://www.gutenberg.org/files/1513/1513-0.txt"
MD_URL = "https://www.gutenberg.org/files/2701/2701-0.txt"
RJ_FILE = "pg1513.txt"
MD_FILE = "pg2701.txt"
OUT_DIR = Path("outputs")

# ----------------------------
# UTILITIES
# ----------------------------
def maybe_download(url, dest):
    """Download file if not found locally"""
    if os.path.exists(dest): return
    try:
        import requests
        print(f"📥 Downloading {url}...")
        text = requests.get(url, timeout=20).text
        open(dest, "w", encoding="utf-8").write(text)
    except Exception as e:
        print("⚠️ Could not download:", e)

def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    if start and end:
        return text[start.end():end.start()]
    return text

# ----------------------------
# SENTIMENT
# ----------------------------
def get_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✅ Using VADER Sentiment Analyzer")
        return SentimentIntensityAnalyzer()
    except:
        print("⚠️ VADER not found, using simple lexicon fallback")
        pos = set("love joy happy peace kind sweet hope brave noble fair".split())
        neg = set("hate death fear pain sorrow anger dark cruel sin".split())
        def score(t):
            w = re.findall(r"[a-z']+", t.lower())
            pos_ct = sum(1 for x in w if x in pos)
            neg_ct = sum(1 for x in w if x in neg)
            total = len(w)+1
            return {"compound": (pos_ct - neg_ct)/np.sqrt(total),
                    "pos": pos_ct/total, "neg": neg_ct/total, "neu": 1 - (pos_ct+neg_ct)/total}
        class Fallback:
            def polarity_scores(self, t): return score(t)
        return Fallback()

# ----------------------------
# TEXT SEGMENTATION
# ----------------------------
def chunk_text(text, work, size=2000):
    words = re.findall(r"\S+", text)
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if len(chunk) < 300: break
        chunks.append({"work":work, "segment_id": f"{work[:2]}_{i//size}", "text":chunk})
        i += size
    return chunks

def segment_rj(text):  # Try ACT/SCENE if available
    parts = re.split(r"\n\s*SCENE\s+[IVXLC]+\.*[^\n]*\n", text)
    if len(parts) < 3:
        return chunk_text(text, "Romeo and Juliet")
    return [{"work":"Romeo and Juliet", "segment_id":f"Scene_{i}", "text":p} for i,p in enumerate(parts) if len(p.strip())>200]

def segment_md(text):  # Try CHAPTERS
    parts = re.split(r"\n\s*CHAPTER\s+[^\n]*\n", text)
    if len(parts) < 5:
        return chunk_text(text, "Moby-Dick")
    return [{"work":"Moby-Dick", "segment_id":f"Chapter_{i}", "text":p} for i,p in enumerate(parts) if len(p.strip())>300]

# ----------------------------
# ANALYSIS
# ----------------------------
def analyze(segments, analyzer):
    rows = []
    for s in segments:
        sc = analyzer.polarity_scores(s["text"])
        rows.append({"work":s["work"], "segment_id":s["segment_id"], **sc})
    return pd.DataFrame(rows)

def summary_table(df):
    return df.groupby("work").agg(
        segments=('segment_id','count'),
        mean_compound=('compound','mean'),
        std_compound=('compound','std'),
        pct_positive=('compound',lambda s:(s>0.05).mean()*100),
        pct_negative=('compound',lambda s:(s<-0.05).mean()*100)
    ).reset_index()

def top_extremes(df, work, n=3):
    sub = df[df.work==work].sort_values("compound")
    return sub.head(n), sub.tail(n)

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    maybe_download(RJ_URL, RJ_FILE)
    maybe_download(MD_URL, MD_FILE)

    rj = strip_gutenberg(open(RJ_FILE,encoding='utf-8',errors='ignore').read())
    md = strip_gutenberg(open(MD_FILE,encoding='utf-8',errors='ignore').read())

    analyzer = get_analyzer()
    rj_segments = segment_rj(rj)
    md_segments = segment_md(md)

    df = pd.concat([analyze(rj_segments, analyzer),
                    analyze(md_segments, analyzer)], ignore_index=True)

    # Summary
    print("\n📊 Overall Sentiment Summary:")
    print(summary_table(df).round(3).to_string(index=False))

    # Extremes
    for work in df.work.unique():
        lows, highs = top_extremes(df, work)
        print(f"\n🔻 Most negative segments — {work}")
        for _,r in lows.iterrows():
            print(f"  {r.segment_id} → compound={r.compound:.3f}")
        print(f"\n🔺 Most positive segments — {work}")
        for _,r in highs.iterrows():
            print(f"  {r.segment_id} → compound={r.compound:.3f}")

    # Sentiment distribution plot
    plt.figure(figsize=(8,4))
    for work in df.work.unique():
        plt.hist(df[df.work==work].compound, bins=25, alpha=0.5, label=work, density=True)
    plt.legend()
    plt.title("Sentiment distribution")
    plt.xlabel("Compound sentiment")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sentiment_distribution.png")
    print(f"\n📈 Saved histogram → {OUT_DIR/'sentiment_distribution.png'}")

    # Common word sentiment correlation (quick version)
    def get_words(text):
        return [w.lower() for w in re.findall(r"[A-Za-z']+", text) if len(w)>2]

    STOP = set("the a an and or of in to for from by is was are be i you he she they".split())
    from collections import Counter
    def extract_words(segments, df):
        pos_texts = [s["text"] for s in segments if df.loc[df.segment_id==s["segment_id"],"compound"].mean()>0.05]
        neg_texts = [s["text"] for s in segments if df.loc[df.segment_id==s["segment_id"],"compound"].mean()<-0.05]
        pos_counts = Counter(w for t in pos_texts for w in get_words(t) if w not in STOP)
        neg_counts = Counter(w for t in neg_texts for w in get_words(t) if w not in STOP)
        return pos_counts, neg_counts

    rj_pos, rj_neg = extract_words(rj_segments, df)
    md_pos, md_neg = extract_words(md_segments, df)

    common_pos = sorted(set(rj_pos) & set(md_pos))
    common_neg = sorted(set(rj_neg) & set(md_neg))

    print("\n💬 Common positive words:", ", ".join(common_pos[:20]), "...")
    print("💭 Common negative words:", ", ".join(common_neg[:20]), "...")

    print("\n✅ Done! Plots saved; everything else shown above.")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()
