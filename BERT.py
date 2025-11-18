import os, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# Choose sentiment model here: "hf-sst2", "vader", or "textblob"
MODEL_NAME = "hf-sst2"

# -----------------------------
# Download texts (Gutenberg)
# -----------------------------
rj_url = "https://www.gutenberg.org/files/1513/1513-0.txt"
hm_url = "https://www.gutenberg.org/files/27761/27761-0.txt"
rj_file, hm_file = "/content/pg1513.txt", "/content/pg27761.txt"

def download(url, dest):
    import requests
    if not os.path.exists(dest):
        print(f"Downloading {dest} ...")
        text = requests.get(url, timeout=40).text
        open(dest,"w",encoding="utf-8").write(text)

download(rj_url, rj_file)
download(hm_url, hm_file)

def strip_gutenberg(text):
    s = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    e = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    if s and e and s.end()<e.start():
        return text[s.end():e.start()]
    return text

rj_text = strip_gutenberg(open(rj_file,encoding='utf-8',errors='ignore').read())
hm_text = strip_gutenberg(open(hm_file,encoding='utf-8',errors='ignore').read())

# -----------------------------
# Sentiment Backends
# -----------------------------
class SentimentBackend:
    def score(self, text:str)->dict: raise NotImplementedError

class VaderBackend(SentimentBackend):
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.an = SentimentIntensityAnalyzer()
    def score(self, text): return self.an.polarity_scores(text)

class TextBlobBackend(SentimentBackend):
    def __init__(self):
        from textblob import TextBlob; self.TB=TextBlob
    def score(self, text):
        pol=float(self.TB(text).sentiment.polarity)
        pos=max(0,pol); neg=max(0,-pol); neu=max(0,1-pos-neg)
        return {"pos":pos,"neg":neg,"neu":neu,"compound":pol}

class HFBackend(SentimentBackend):
    def __init__(self):
        from transformers import pipeline
        self.clf = pipeline("sentiment-analysis",
                            model="distilbert-base-uncased-finetuned-sst-2-english",
                            return_all_scores=True, truncation=True)
    def score(self, text):
        out=self.clf(text[:4000])[0]
        probs={d["label"].upper():d["score"] for d in out}
        pos,neg=probs.get("POSITIVE",0),probs.get("NEGATIVE",0)
        return {"pos":pos,"neg":neg,"neu":max(0,1-pos-neg),"compound":pos-neg}

def get_backend(name):
    if name=="vader": print("Using VADER"); return VaderBackend()
    if name=="textblob": print("Using TextBlob"); return TextBlobBackend()
    print("Using Hugging Face DistilBERT SST-2"); return HFBackend()

backend = get_backend(MODEL_NAME)

# -----------------------------
# Segmentation
# -----------------------------
def segment_play(text, title):
    parts = re.split(r"\n\s*SCENE\s+[IVXLC]+\.*[^\n]*\n", text, flags=re.I)
    if len(parts)<3:
        words=re.findall(r"\S+", text)
        chunks=[]; i=0; k=0
        while i<len(words):
            chunk=" ".join(words[i:i+2000])
            if len(chunk)<300: break
            chunks.append({"work":title,"segment_id":f"{title[:2].upper()}_{k}","text":chunk})
            i+=2000; k+=1
        return chunks
    segs=[]
    for i,p in enumerate(parts):
        clean=p.strip()
        if len(clean)>300: segs.append({"work":title,"segment_id":f"{title[:2].upper()}_SCENE_{i}","text":clean})
    return segs

rj_segments = segment_play(rj_text,"Romeo and Juliet")
hm_segments = segment_play(hm_text,"Hamlet")

# -----------------------------
# Analysis
# -----------------------------
def analyze_segments(segments, backend):
    rows=[]
    for s in segments:
        sc=backend.score(s["text"])
        rows.append({"work":s["work"],"segment_id":s["segment_id"],**sc})
    df=pd.DataFrame(rows)
    df["order"]=df.groupby("work").cumcount()
    return df

df_rj=analyze_segments(rj_segments,backend)
df_hm=analyze_segments(hm_segments,backend)
df=pd.concat([df_rj,df_hm],ignore_index=True)

# -----------------------------
# Summary + Print
# -----------------------------
summary=df.groupby("work").agg(
    segments=('segment_id','count'),
    mean_compound=('compound','mean'),
    std_compound=('compound','std'),
    pct_positive=('compound',lambda s:(s>0.05).mean()*100),
    pct_negative=('compound',lambda s:(s<-0.05).mean()*100)
).round(4).reset_index()

print("📊 Sentiment Summary (Hamlet vs Romeo & Juliet)")
print(summary.to_string(index=False))

# Extremes
def show_extremes(df,work,n=3):
    sub=df[df.work==work].sort_values("compound")
    lows,highs=sub.head(n),sub.tail(n)
    print(f"\n🔻 Most negative segments — {work}")
    for _,r in lows.iterrows():
        print(f" {r.segment_id}: compound={r.compound:.3f}")
    print(f"\n🔺 Most positive segments — {work}")
    for _,r in highs.iterrows():
        print(f" {r.segment_id}: compound={r.compound:.3f}")

show_extremes(df,"Hamlet")
show_extremes(df,"Romeo and Juliet")

# Direct comparison
rj_mean, hm_mean = df[df.work=="Romeo and Juliet"].compound.mean(), df[df.work=="Hamlet"].compound.mean()
print(f"\n⚖️ Mean compound:\n  Hamlet = {hm_mean:.4f}\n  Romeo & Juliet = {rj_mean:.4f}\n  Δ = {hm_mean - rj_mean:+.4f}")

# -----------------------------
# Plots (inline)
# -----------------------------
plt.figure(figsize=(9,4))
for w in ["Romeo and Juliet","Hamlet"]:
    plt.hist(df[df.work==w].compound,bins=25,alpha=0.5,density=True,label=w)
plt.legend(); plt.title("Sentiment distribution"); plt.xlabel("Compound sentiment"); plt.ylabel("Density")
plt.show()

plt.figure(figsize=(9,4))
for w in ["Romeo and Juliet","Hamlet"]:
    d=df[df.work==w].sort_values("order")
    smooth=d.compound.rolling(5,center=True,min_periods=1).mean()
    plt.plot(d.order,smooth,label=w)
plt.legend(); plt.title("Sentiment trajectory (smoothed)"); plt.xlabel("Segment order"); plt.ylabel("Compound sentiment")
plt.show()
