import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
RJ_LOCAL = "/content/pg1513.txt"   # Romeo & Juliet
MD_LOCAL = "/content/pg2701.txt"   # Moby-Dick
RJ_URL = "https://www.gutenberg.org/files/1513/1513-0.txt"
MD_URL = "https://www.gutenberg.org/files/2701/2701-0.txt"
OUT_DIR = Path("outputs")

# ----------------------------
# Utilities
# ----------------------------
def maybe_download(url: str, dest: str):

    if os.path.exists(dest):
        return
    try:
        import requests
        print(f"Downloading {url} -> {dest} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest, "w", encoding="utf-8") as f:
            f.write(r.text)
    except Exception as e:
        print(f"Could not download {url}: {e}")
        print("Please place the file manually and rerun.")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def strip_gutenberg(text: str) -> str:
    start_re = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
    end_re = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
    start = start_re.search(text)
    end = end_re.search(text)
    if start and end and start.end() < end.start():
        return text[start.end():end.start()]
    return text

# ----------------------------
# Sentiment analyzer
# ----------------------------
def get_sentiment_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("Using VADER sentiment.")
        return SentimentIntensityAnalyzer()
    except Exception:
        print("VADER not found. Using a simple fallback lexicon (less accurate).")
        pos_words = set("""
            love loving loved joy joyful joyous happy happiness hope hopeful tender kind mercy peace
            sweet light delight praise gentle grace bless blessed brave honor noble fair friend
            friendship beauty beautiful triumph calm smile laughter
        """.split())
        neg_words = set("""
            hate hated hating sorrow sad sorrowful grief grievous pain painful death deadly die
            bleak blood bloody fear fearful rage fury angry anger wrath despair alone lonely dark
            darkness cold cruel curse cursed jealous jealousy shame guilty guilt sin sinful wound
            wounded kill killing murder murdered doom tragic tragedy monster
        """.split())

        def score(text):
            words = re.findall(r"[A-Za-z']+", text.lower())
            if not words:
                return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
            pos = sum(1 for w in words if w in pos_words)
            neg = sum(1 for w in words if w in neg_words)
            compound = (pos - neg) / (np.sqrt(len(words)) + 1e-9)
            total = max(pos+neg, 1)
            pos_p = pos/total
            neg_p = neg/total
            neu_p = 1.0 - min(1.0, pos_p+neg_p)
            return {"neg":neg_p,"neu":neu_p,"pos":pos_p,"compound":compound}

        class FallbackAnalyzer:
            def polarity_scores(self, text):
                return score(text)
        return FallbackAnalyzer()

# ----------------------------
# Segmentation
# ----------------------------
def chunk_text(text, work, chunk_size=2000, overlap=200):
    words = re.findall(r"\S+", text)
    segments = []
    i = 0
    seg = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) < 300:
            break
        chunk = " ".join(chunk_words)
        segments.append({"work":work, "segment_id": f"CHUNK_{seg}", "text": chunk})
        seg += 1
        i += chunk_size - overlap
    return segments

def segment_rj(text):
    # Split by ACT/SCENE headings if present, else fallback to chunks
    segments = []
    t = text
    acts = re.split(r"\n\s*ACT\s+[IVXLC]+\s*\n", t, flags=re.I)
    act_id = 0
    for act_chunk in acts:
        scene_parts = re.split(r"\n\s*SCENE\s+[IVXLC]+\.*[^\n]*\n", act_chunk, flags=re.I)
        scene_id = 0
        for scene in scene_parts:
            clean = scene.strip()
            if len(clean) < 200:
                continue
            segments.append({"work":"Romeo and Juliet","segment_id": f"ACT{act_id}_SCENE{scene_id}","text": clean})
            scene_id += 1
        act_id += 1
    if len(segments) < 5:
        segments = chunk_text(text, "Romeo and Juliet", chunk_size=1800, overlap=150)
    return segments

def segment_md(text):
    parts = re.split(r"\n\s*CHAPTER\s+[^\n]*\n", text, flags=re.I)
    segments = []
    for i, p in enumerate(parts):
        clean = p.strip()
        if len(clean) < 400:
            continue
        segments.append({"work":"Moby-Dick","segment_id": f"CHAPTER_{i}","text": clean})
    if len(segments) < 10:
        segments = chunk_text(text, "Moby-Dick", chunk_size=2500, overlap=200)
    return segments

# ----------------------------
# Analysis
# ----------------------------
def analyze_segments(segments, analyzer):
    rows = []
    for s in segments:
        scores = analyzer.polarity_scores(s["text"])
        rows.append({
            "work": s["work"],
            "segment_id": s["segment_id"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"],
            "n_chars": len(s["text"]),
            "n_words": len(re.findall(r"[A-Za-z']+", s["text"]))
        })
    return pd.DataFrame(rows)

def add_order_and_smooth(df):
    df = df.copy()
    df["order"] = df.groupby("work").cumcount()
    df["compound_smooth"] = df.groupby("work")["compound"].transform(
        lambda s: s.rolling(window=5, center=True, min_periods=1).mean()
    )
    return df

def top_extremes(df, work, k=5):
    d = df[df["work"]==work].sort_values("compound")
    return d.head(k)[["work","segment_id","compound"]].copy(), d.tail(k)[["work","segment_id","compound"]].copy()

def excerpts(segments, extremes_df, max_chars=300):
    seg_map = {s["segment_id"]: s["text"] for s in segments}
    rows = []
    for _, row in extremes_df.iterrows():
        text = seg_map.get(row["segment_id"], "")
        ex = re.sub(r"\s+", " ", text)[:max_chars] + ("..." if len(text)>max_chars else "")
        r = row.to_dict()
        r["excerpt"] = ex
        rows.append(r)
    return pd.DataFrame(rows)

def tokenize(text):
    return [w.lower() for w in re.findall(r"[A-Za-z']+", text)]

STOP = set("""
the a an and or but of in on with to for from by it is was be are were this that these those i you he she they we him her them my our your their not as at if then so than o oh
""".split())

def word_sentiment_association(segments, df_scores, topn=30):
    seg_map = {s["segment_id"]: s["text"] for s in segments}
    labels = {}
    for _, r in df_scores.iterrows():
        if r["segment_id"] in seg_map:
            labels[r["segment_id"]] = 1 if r["compound"]>0.05 else (-1 if r["compound"]<-0.05 else 0)

    from collections import Counter
    counts = {1: Counter(), -1: Counter(), 0: Counter()}
    total = {1:0, -1:0, 0:0}
    for sid, text in seg_map.items():
        lab = labels.get(sid, 0)
        words = [w for w in tokenize(text) if w not in STOP and len(w)>2]
        counts[lab].update(words)
        total[lab] += len(words)

    vocab = set(counts[1].keys()) | set(counts[-1].keys())
    data = []
    for w in vocab:
        pos = counts[1][w] + 1
        neg = counts[-1][w] + 1
        pos_total = total[1] + len(vocab)
        neg_total = total[-1] + len(vocab)
        log_odds = np.log(pos/pos_total) - np.log(neg/neg_total)
        data.append((w, log_odds, counts[1][w]+counts[-1][w]))
    df_words = pd.DataFrame(data, columns=["word","log_odds_pos_vs_neg","count"])
    top_pos = df_words.sort_values("log_odds_pos_vs_neg", ascending=False).head(topn)
    top_neg = df_words.sort_values("log_odds_pos_vs_neg").head(topn)
    return top_pos, top_neg, df_words

# ----------------------------
# Plotting
# ----------------------------
def plot_trajectory(df, work, title, out_path: Path):
    d = df[df["work"]==work].sort_values("order")
    plt.figure(figsize=(10,5))
    plt.plot(d["order"], d["compound"], label="compound")
    plt.plot(d["order"], d["compound_smooth"], label="compound (rolling mean, w=5)")
    plt.xlabel("Segment order")
    plt.ylabel("Sentiment (compound)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_hist_comparison(df, out_path: Path):
    d1 = df[df["work"]=="Romeo and Juliet"]["compound"]
    d2 = df[df["work"]=="Moby-Dick"]["compound"]
    plt.figure(figsize=(10,5))
    plt.hist(d1, bins=30, alpha=0.5, label="Romeo & Juliet", density=True)
    plt.hist(d2, bins=30, alpha=0.5, label="Moby-Dick", density=True)
    plt.xlabel("Compound sentiment")
    plt.ylabel("Density")
    plt.title("Distribution of segment sentiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure files exist (local or download)
    if not os.path.exists(RJ_LOCAL):
        maybe_download(RJ_URL, RJ_LOCAL)
    if not os.path.exists(MD_LOCAL):
        maybe_download(MD_URL, MD_LOCAL)

    rj_text = strip_gutenberg(read_text(RJ_LOCAL))
    md_text = strip_gutenberg(read_text(MD_LOCAL))

    rj_segments = segment_rj(rj_text)
    md_segments = segment_md(md_text)

    analyzer = get_sentiment_analyzer()

    df_rj = analyze_segments(rj_segments, analyzer)
    df_md = analyze_segments(md_segments, analyzer)
    df_all = pd.concat([df_rj, df_md], ignore_index=True)
    df_all = add_order_and_smooth(df_all)

    # Summary
    summary = df_all.groupby("work").agg(
        segments=("segment_id","count"),
        mean_compound=("compound","mean"),
        median_compound=("compound","median"),
        std_compound=("compound","std"),
        pct_positive=("compound", lambda s: (s>0.05).mean()*100),
        pct_negative=("compound", lambda s: (s<-0.05).mean()*100),
        pct_neutral=("compound", lambda s: ((s>=-0.05)&(s<=0.05)).mean()*100),
        total_words=("n_words","sum")
    ).reset_index()

    # Extremes + excerpts
    rj_negs, rj_poss = top_extremes(df_all, "Romeo and Juliet")
    md_negs, md_poss = top_extremes(df_all, "Moby-Dick")
    rj_neg_ex = excerpts(rj_segments, rj_negs)
    rj_pos_ex = excerpts(rj_segments, rj_poss)
    md_neg_ex = excerpts(md_segments, md_negs)
    md_pos_ex = excerpts(md_segments, md_poss)

    # Save CSVs
    summary.to_csv(OUT_DIR / "summary_stats.csv", index=False)
    df_all.to_csv(OUT_DIR / "segment_sentiment.csv", index=False)
    pd.concat([rj_neg_ex.assign(kind="most_negative"), rj_pos_ex.assign(kind="most_positive")]).to_csv(
        OUT_DIR / "romeo_juliet_extremes.csv", index=False)
    pd.concat([md_neg_ex.assign(kind="most_negative"), md_pos_ex.assign(kind="most_positive")]).to_csv(
        OUT_DIR / "moby_dick_extremes.csv", index=False)

    # Word–sentiment association + commonalities
    rj_top_pos, rj_top_neg, rj_word_table = word_sentiment_association(rj_segments, df_rj)
    md_top_pos, md_top_neg, md_word_table = word_sentiment_association(md_segments, df_md)
    rj_word_table.to_csv(OUT_DIR / "rj_word_sentiment_table.csv", index=False)
    md_word_table.to_csv(OUT_DIR / "md_word_sentiment_table.csv", index=False)

    rj_pos_set = set(rj_top_pos.head(200)["word"])
    rj_neg_set = set(rj_top_neg.head(200)["word"])
    md_pos_set = set(md_top_pos.head(200)["word"])
    md_neg_set = set(md_top_neg.head(200)["word"])
    common_pos = sorted(rj_pos_set & md_pos_set)[:50]
    common_neg = sorted(rj_neg_set & md_neg_set)[:50]
    pd.DataFrame({
        "common_positive_words": pd.Series(common_pos, dtype="string"),
        "common_negative_words": pd.Series(common_neg, dtype="string")
    }).to_csv(OUT_DIR / "common_sentiment_words.csv", index=False)

    # Plots
    plot_trajectory(df_all, "Romeo and Juliet",
                    "Romeo & Juliet sentiment over segments",
                    OUT_DIR / "rj_sentiment_curve.png")
    plot_trajectory(df_all, "Moby-Dick",
                    "Moby-Dick sentiment over segments",
                    OUT_DIR / "md_sentiment_curve.png")
    plot_hist_comparison(df_all, OUT_DIR / "sentiment_distribution_comparison.png")

    print("\nDone. Files written to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
