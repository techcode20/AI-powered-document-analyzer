import time, os, json, re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from groq import Groq

# Set your Groq API key here OR use environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")

_nlp_model = None
_zero_shot_pipeline = None

def _get_nlp():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model

def _get_zero_shot():
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        _zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli", device=-1)
    return _zero_shot_pipeline

def _groq(prompt, max_tokens=600):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=max_tokens,
            messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[groq entities] {e}")
        return ""

def extract_entities(text: str) -> dict:
    start = time.time()
    sample = text[:3000]
    prompt = """Extract named entities from the text below.
Return ONLY a JSON object. No explanation. No markdown. Just the JSON.

Rules:
- names: real person names only
- organizations: companies, colleges, institutions
- locations: countries, cities, states ONLY (never tools/libraries)
- dates: dates, years, durations
- amounts: numbers, percentages, money
- other: anything else

Text:
""" + sample + """
JSON (fill with real values from text):
{"names":[],"organizations":[],"locations":[],"dates":[],"amounts":[],"other":[]}"""

    raw = _groq(prompt, 700)
    buckets = {"names":[],"organizations":[],"locations":[],"dates":[],"amounts":[],"other":[]}

    try:
        s = raw.find("{")
        e = raw.rfind("}") + 1
        if s != -1 and e > s:
            parsed = json.loads(raw[s:e])
            for k in buckets:
                for item in parsed.get(k, []):
                    if isinstance(item, str) and item.strip():
                        buckets[k].append({"text": item.strip(), "label": k})
    except Exception as ex:
        print(f"[entities parse error] {ex} — raw: {raw[:200]}")
        nlp_model = _get_nlp()
        doc = nlp_model(text[:50000])
        MAP = {
            "PERSON":"names","ORG":"organizations",
            "GPE":"locations","LOC":"locations",
            "DATE":"dates","TIME":"dates",
            "MONEY":"amounts","CARDINAL":"amounts",
            "PERCENT":"amounts","QUANTITY":"amounts"
        }
        seen = set()
        for ent in doc.ents:
            v = ent.text.strip()
            if not v or v.lower() in seen: continue
            seen.add(v.lower())
            buckets[MAP.get(ent.label_, "other")].append(
                {"text": v, "label": ent.label_})

    total = sum(len(v) for v in buckets.values())
    return {
        "entities": buckets, "total_count": total,
        "confidence": 0.92,
        "processing_time_ms": round((time.time()-start)*1000, 2)
    }


def extract_keywords(text: str, top_n: int = 15) -> dict:
    start = time.time()
    clean = str(text)
    sentences = [s.strip() for s in clean.replace("\n", ". ").split(". ")
                 if len(s.strip()) > 5]
    if len(sentences) < 2:
        sentences = [clean]
    try:
        # Extended stopwords — removes common document/resume noise words
        EXTRA_STOPS = {
            "using","used","use","also","one","two","three","may","well",
            "will","would","could","like","make","made","get","got","good",
            "work","worked","working","year","years","month","months","time",
            "summary","objective","experience","education","skills","project",
            "projects","certificate","certificates","certification","resume",
            "name","email","phone","address","linkedin","github","profile",
            "responsible","responsibility","developed","development","managed",
            "created","implemented","designed","built","ability","knowledge",
            "strong","excellent","proficient","familiar","understanding",
            "include","includes","including","based","related","various",
            "multiple","different","new","current","present","team","company"
        }
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        all_stops = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPS))
        vec = TfidfVectorizer(stop_words=all_stops,
                              max_features=500, ngram_range=(1,2),
                              min_df=1, max_df=0.85)
        mat = vec.fit_transform(sentences)
        names = vec.get_feature_names_out()
        scores = mat.sum(axis=0).A1
        top = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)[:top_n]
        mx = top[0][1] if top else 1.0
        keywords = [{"word": w, "score": round(float(s)/float(mx), 4)}
                    for w, s in top]
    except Exception as e:
        print(f"[keywords] {e}")
        keywords = []
    return {"keywords": keywords,
            "processing_time_ms": round((time.time()-start)*1000, 2)}


TOPIC_LABELS = [
    "Technology","Finance & Business","Healthcare & Medicine",
    "Legal & Compliance","Science & Research","Education",
    "Politics & Government","Sports","Entertainment & Media",
    "Environment & Climate","Human Resources","Marketing & Sales",
]

def classify_topic(text: str) -> dict:
    start = time.time()
    sample = str(text)[:1500]
    # Use Groq for topic classification — faster and no caching issues
    import os, json
    from groq import Groq
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
    labels_str = ", ".join(TOPIC_LABELS)
    prompt = f"""Classify this document into one or more of these topic categories:
{labels_str}

Return ONLY a JSON array of objects sorted by relevance score (highest first):
[{{"label": "Category Name", "score": 0.95}}, ...]

Include ALL categories with their scores (0.0 to 1.0).
Text: {sample}

JSON:"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=400,
            messages=[{{"role":"user","content":prompt}}])
        raw = r.choices[0].message.content.strip()
        s = raw.find("["); e = raw.rfind("]")+1
        scores = json.loads(raw[s:e])
        # Ensure all labels present
        found = {{item["label"] for item in scores}}
        for lbl in TOPIC_LABELS:
            if lbl not in found:
                scores.append({{"label": lbl, "score": 0.01}})
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        return {{
            "top_topic": scores[0]["label"],
            "all_scores": [{{"label": s["label"], "score": round(float(s["score"]),4)}} for s in scores],
            "confidence": round(float(scores[0]["score"]),4),
            "processing_time_ms": round((time.time()-start)*1000,2)
        }}
    except Exception as ex:
        print(f"[topic groq] {{ex}}, falling back to zero-shot")
    # Fallback to zero-shot
    clf = _get_zero_shot()
    result = clf(sample, candidate_labels=TOPIC_LABELS, multi_label=True)
    scores = [{"label": l, "score": round(float(s), 4)}
              for l, s in zip(result["labels"], result["scores"])]
    return {
        "top_topic": scores[0]["label"],
        "all_scores": scores,
        "confidence": scores[0]["score"],
        "processing_time_ms": round((time.time()-start)*1000, 2)
  }
  

ENTITY_NOISE = {"wi-fi","wifi","isbn","cip","publisher","author","copyright","printed","edition"}

def clean_other_entities(others):
    return [e for e in others if e.lower() not in ENTITY_NOISE and len(e) > 2]
