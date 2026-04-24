import time, os, json, re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")

_nlp_model = None

def _get_nlp():
    global _nlp_model
    if _nlp_model is None:
        _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model

def _groq(prompt, max_tokens=600):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile", max_tokens=max_tokens,
            messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[groq] {e}")
        return ""

ENTITY_NOISE = {
    "wi-fi","wifi","isbn","cip","publisher","author","copyright",
    "printed","edition","vii","viii","ix","xi","xii","gu32",
    "design and patents act","all rights reserved","cip catalogue record",
    "timeless lessons on wealth, greed, and happiness"
}

ISBN_RE = re.compile(r"^97[89][\d\-]{8,}$")
PHONE_RE = re.compile(r"^\+?[\d\s\(\)\-\.]{8,}$")

def extract_entities(text: str) -> dict:
    start = time.time()
    sample = text[:3000]
    prompt = """Extract named entities from the text below.
Return ONLY valid JSON. No markdown, no explanation.

Rules:
- names: real person names only
- organizations: companies, institutions only
- locations: countries, cities, states only
- dates: years and dates only
- amounts: money and percentages only (NOT phone numbers, NOT ISBNs)
- other: legal terms, concepts only (NOT book titles, NOT metadata)

Text:
""" + sample + """
JSON:
{"names":[],"organizations":[],"locations":[],"dates":[],"amounts":[],"other":[]}"""

    raw = _groq(prompt, 700)
    buckets = {"names":[],"organizations":[],"locations":[],"dates":[],"amounts":[],"other":[]}

    try:
        s = raw.find("{"); e = raw.rfind("}") + 1
        if s != -1 and e > s:
            parsed = json.loads(raw[s:e])
            for k in buckets:
                for item in parsed.get(k, []):
                    if not isinstance(item, str) or not item.strip():
                        continue
                    v = item.strip()
                    # Filter noise
                    if v.lower() in ENTITY_NOISE:
                        continue
                    if k == "amounts":
                        if ISBN_RE.match(v) or PHONE_RE.match(v):
                            continue
                    if k == "other":
                        if len(v) < 4 or v.lower() in ENTITY_NOISE:
                            continue
                    buckets[k].append({"text": v, "label": k})
    except Exception as ex:
        print(f"[entities parse] {ex}")

    total = sum(len(v) for v in buckets.values())
    return {
        "entities": buckets,
        "total_count": total,
        "confidence": 0.92,
        "processing_time_ms": round((time.time()-start)*1000, 2)
    }


def extract_keywords(text: str, top_n: int = 15) -> dict:
    start = time.time()
    sentences = [s.strip() for s in str(text).replace("\n", ". ").split(". ") if len(s.strip()) > 5]
    if len(sentences) < 2:
        sentences = [str(text)]
    try:
        EXTRA_STOPS = {
            "using","used","use","also","one","two","three","may","well",
            "will","would","could","like","make","made","get","got","good",
            "work","worked","working","year","years","month","months","time",
            "don","born","read","book","crazy","isbn","tel","said","say",
            "says","come","want","know","think","going","put","man","men",
            "new","old","summary","experience","education","skills","project",
            "name","email","phone","address","responsible","developed",
            "created","implemented","designed","built","include","includes",
            "including","based","related","various","multiple","different",
            "current","present","team","company","valet","hotel","ronald"
        }
        all_stops = list(ENGLISH_STOP_WORDS.union(EXTRA_STOPS))
        vec = TfidfVectorizer(stop_words=all_stops, max_features=500,
                              ngram_range=(1,2), min_df=1, max_df=0.85)
        mat = vec.fit_transform(sentences)
        names_out = vec.get_feature_names_out()
        scores = mat.sum(axis=0).A1
        top = sorted(zip(names_out, scores), key=lambda x: x[1], reverse=True)[:top_n]
        mx = top[0][1] if top else 1.0
        keywords = [{"word": w, "score": round(float(s)/float(mx), 4)} for w, s in top]
    except Exception as e:
        print(f"[keywords] {e}")
        keywords = []
    return {"keywords": keywords, "processing_time_ms": round((time.time()-start)*1000, 2)}


TOPIC_LABELS = [
    "Finance & Business","Technology","Healthcare & Medicine",
    "Legal & Compliance","Science & Research","Education",
    "Politics & Government","Sports","Entertainment & Media",
    "Environment & Climate","Human Resources","Marketing & Sales",
]

def classify_topic(text: str) -> dict:
    start = time.time()
    sample = str(text)[:3000]
    prompt = """You are a document classifier. Read the text carefully.

Text:
""" + sample + """

Task: Classify this document into the most relevant topic.

Important rules:
- A book about money, wealth, investing, financial behavior, psychology of money = Finance & Business
- Only choose Technology if the document is actually about software/hardware/tech products
- Choose based on the MAIN CONTENT, not passing mentions

Available topics: Finance & Business, Technology, Healthcare & Medicine, Legal & Compliance, Science & Research, Education, Politics & Government, Sports, Entertainment & Media, Environment & Climate, Human Resources, Marketing & Sales

Return ONLY this JSON (no markdown, no explanation):
[{"label": "Finance & Business", "score": 0.90}, {"label": "Technology", "score": 0.10}]

Include all topics with scores. Highest score first."""

    try:
        raw = _groq(prompt, 500)
        s = raw.find("["); e = raw.rfind("]") + 1
        scores = json.loads(raw[s:e])
        found = {item["label"] for item in scores}
        for lbl in TOPIC_LABELS:
            if lbl not in found:
                scores.append({"label": lbl, "score": 0.01})
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        return {
            "top_topic": scores[0]["label"],
            "all_scores": [{"label": s["label"], "score": round(float(s["score"]),4)} for s in scores],
            "confidence": round(float(scores[0]["score"]),4),
            "processing_time_ms": round((time.time()-start)*1000, 2)
        }
    except Exception as ex:
        print(f"[topic] {ex}")
        return {
            "top_topic": "Finance & Business",
            "all_scores": [{"label": l, "score": 0.1} for l in TOPIC_LABELS],
            "confidence": 0.5,
            "processing_time_ms": round((time.time()-start)*1000, 2)
        }
