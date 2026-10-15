"""Newsdesk pipeline: builds data.json from the free news feeds, with summaries from a local Ollama model.

Steps: fetch feeds -> keep articles <= 2 days old -> sort into topics (rules) -> drop paywalled articles
-> read article text + image -> preview + summary via Ollama -> write data.json.

Usage: python3 pipeline/newsdesk.py [--out data.json] [--model gemma3:4b] [--limit 45]
Only uses the Python standard library and curl.
"""
import argparse, concurrent.futures as cf, datetime, html, json, os, re, subprocess, sys, time, urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

HERE = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128 Safari/537.36"
SRC = {"ntv": "n-tv", "ts": "tagesschau", "anth": "Anthropic", "oai": "OpenAI", "cnbc": "CNBC",
       "bi": "Business Insider", "ollama": "Ollama"}
GERMAN = {"n-tv", "tagesschau"}
MAX_AGE = datetime.timedelta(days=2)
NOW = datetime.datetime.now(datetime.timezone.utc)
ATOM = "{http://www.w3.org/2005/Atom}"

COMPANIES = {
    "Anthropic": r"Anthropic|\bClaude\b|Mythos", "OpenAI": r"Open ?AI|ChatGPT|Chat-GPT|Codex|\bGPT-",
    "Google": r"Google|Alphabet|Gemini|DeepMind", "Microsoft": r"Microsoft|Copilot|Azure",
    "Nvidia": r"Nvidia|NVIDIA", "Apple": r"\bApple\b|iPhone|\bSiri\b|Mac Studio|MacBook",
    "Ollama": r"Ollama", "Groq": r"\bGroq\b|\bGrok\b|\bxAI\b",
}
AI = r"\bAI\b|A\.I\.|\bKI\b|KI-|künstliche Intelligenz|artificial intelligence|chatbot|Chatbot|\bagent|Agent|model|Modell|GPT|Gemini|Claude|Copilot|Muse|LLM|data cent|Rechenzentr|GPU|AI chip|KI-Chip"
TECH = r"Nvidia|NVIDIA|Microsoft|\bApple\b|Alphabet|Google|Amazon|\bMeta\b|TSMC|Broadcom|\bAMD\b|Oracle|Samsung|\bSAP\b|ASML|Infineon|\bIntel\b|Micron|SK Hynix|CoreWeave|Palantir|Nebius|\bDell\b|\bArm\b|Qualcomm|Marvell|Halbleiter|[Cc]hip|semiconductor|Tech-|[Tt]ech stocks|AI stocks|KI-Aktien|KI-Werte|Techwerte|Nasdaq|neocloud"
MARKET = r"stock|shares|Aktie|Börse|Nasdaq|Kurs|investor|Anleger|earnings|profit|Gewinn|Quartal|quarter|\bIPO\b|Börsengang|sell-off|Ausverkauf|[Rr]ally|market|Markt|Dax|DAX|S&P|Wall Street"
ECOMMERCE = r"E-Commerce|e-commerce|Onlinehandel|Online-Handel|Onlineshop|online shopping|online retail|Online-Apotheke|Versandhandel|Versandapotheke|\bTemu\b|\bShein\b|Zalando|Shopify|\beBay\b|\bEtsy\b|Alibaba|DocMorris|Prime Day|Black Friday|Cyber Monday|Amazon (?:retail|Prime|marketplace)"
# OpenAI publishes many customer case studies; they are not news.
OPENAI_CASE_STUDY = r"^How |\b(with|using) (ChatGPT|Codex|OpenAI|GPT)|\b(cuts|boosts|saves|scales|turns|brings)\b"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def curl(url, out=None, timeout=25):
    args = ["curl", "-sL", "-m", str(timeout), "-A", UA, url] + (["-o", out] if out else [])
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout + 5).stdout.decode("utf8", "ignore")
    except subprocess.TimeoutExpired:
        return ""


def clean(s, n=None):
    s = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()
    return s if not n or len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


def iso(s):
    if not s:
        return ""
    for parse in (lambda v: parsedate_to_datetime(v),
                  lambda v: datetime.datetime.fromisoformat(v.replace("Z", "+00:00")),
                  lambda v: datetime.datetime.strptime(v.strip(), "%b %d, %Y").replace(tzinfo=datetime.timezone.utc)):
        try:
            d = parse(s)
            return (d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)).isoformat()
        except Exception:
            pass
    return ""


# ---------- 1. Feeds ----------
CATEGORIES = {"Announcements", "Product", "Policy", "Features", "Research", "Societal Impacts", "Interpretability",
              "Alignment", "Economic Research", "Case Study", "Announcement"}


def parse_anthropic(page):
    out = []
    for m in re.finditer(r'<a[^>]+href="(/news/[a-z0-9\-]+)"[^>]*>(.*?)</a>', page, re.S):
        parts = [html.unescape(p.strip()) for p in re.split(r"<[^>]+>", m.group(2)) if p.strip()]
        date = next((p for p in parts if re.match(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$", p)), "")
        texts = [p for p in parts if p != date and len(p) > 3 and p not in CATEGORIES]
        if not texts:
            continue
        title = max(texts[:2], key=len) if len(texts) > 1 and len(texts[0]) < 25 else texts[0]
        desc = next((t for t in texts if t != title), "")
        out.append(dict(title=title, teaser=clean(desc, 320), url="https://www.anthropic.com" + m.group(1), published=iso(date)))
    return out


def parse_feed(xml_text):
    root = ET.fromstring(xml_text)
    out = []
    for it in root.iter("item"):
        out.append(dict(title=clean(it.findtext("title")), url=(it.findtext("link") or "").strip(),
                        teaser=clean(it.findtext("description"), 320), published=iso(it.findtext("pubDate"))))
    for e in root.iter(ATOM + "entry"):
        link = e.find(ATOM + "link")
        body = html.unescape(e.findtext(ATOM + "summary") or e.findtext(ATOM + "content") or "")
        bullets = re.search(r"summary-list.*?</ul>", body, re.S)
        out.append(dict(title=clean(e.findtext(ATOM + "title")), url=link.get("href") if link is not None else e.findtext(ATOM + "id"),
                        teaser=clean(bullets.group(0) if bullets else body, 320),
                        published=iso(e.findtext(ATOM + "published") or e.findtext(ATOM + "updated"))))
    return out


def fetch_feeds():
    feeds = [l.split() for l in open(os.path.join(HERE, "feeds.txt")) if l.strip() and not l.startswith("#")]
    def one(entry):
        key, url = entry
        src = SRC.get(key.split("_")[0])
        body = curl(url)
        try:
            rows = parse_anthropic(body) if key.startswith("anth") else parse_feed(body)
        except Exception as e:
            log("feed error", key, e)
            rows = []
        return [dict(r, source=src) for r in rows]
    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(one, feeds))
    seen, items = set(), []
    for rows in results:
        for r in rows:
            if not r["title"] or not r["url"] or r["url"] in seen or not r["published"]:
                continue
            if NOW - datetime.datetime.fromisoformat(r["published"]) > MAX_AGE:
                continue
            seen.add(r["url"])
            items.append(r)
    log(f"{len(items)} articles from the last 2 days")
    return items


# ---------- 2. Topics ----------
def classify(a):
    text = a["title"] + " " + a["teaser"]
    src = a["source"]
    topics = []
    if src == "OpenAI" and re.search(OPENAI_CASE_STUDY, a["title"]):
        return []
    if re.search(COMPANIES["Anthropic"], text) or src == "Anthropic":
        topics.append("claude")
    if src in ("Anthropic", "OpenAI", "Ollama") or (any(re.search(p, text) for p in COMPANIES.values()) and re.search(AI, text)):
        topics.append("ki")
    if src in ("CNBC", "n-tv", "tagesschau", "Business Insider") and re.search(TECH, text) and re.search(MARKET, text):
        topics.append("aktien")
    if re.search(ECOMMERCE, text):
        topics.append("ecommerce")
    return topics


# ---------- 3. Paywall + article text ----------
def read_article(a):
    page = curl(a["url"])
    paid = False
    if a["source"] in ("CNBC", "Business Insider"):
        acc = re.search(r'"isAccessibleForFree"\s*:\s*"?(true|false)', page, re.I)
        own = re.search(r'"contentClassification"\s*:\s*\[([^\]]*)\]', page)
        paid = (len(page) < 20000 or "/pro/" in a["url"] or (acc and acc.group(1).lower() == "false")
                or bool(own and "premium" in own.group(1)))
    img = re.search(r'<meta[^>]+(?:property|name)="(?:og:image|twitter:image)"[^>]+content="([^"]+)"', page) \
        or re.search(r'<meta[^>]+content="([^"]+)"[^>]+(?:property|name)="(?:og:image|twitter:image)"', page)
    body = re.sub(r"(?is)<(script|style|nav|footer|header|aside|figure)[^>]*>.*?</\1>", " ", page)
    paras = [clean(p) for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", body)]
    text = " ".join(p for p in paras if len(p) > 60)
    text = re.sub(r"^.*?bevorzugen", "", text)  # n-tv share-bar residue
    return dict(paid=paid, image=html.unescape(img.group(1)) if img else "", text=text)


# ---------- 4. Summaries via Ollama ----------
SCHEMA = {
    "type": "object",
    "properties": {
        "preview": {"type": "string"},
        "sections": {"type": "array", "items": {"type": "object",
                     "properties": {"h": {"type": "string"}, "t": {"type": "string"}}, "required": ["h", "t"]}},
    },
    "required": ["preview", "sections"],
}


def target_sentences(text):
    # A summary stays at roughly a quarter of the original; never a retelling. Max 15 sentences.
    return min(15, int(len(text) / 4 / 130))


def summarize(a, text, model, host):
    german = a["source"] in GERMAN
    n = target_sentences(text)
    lang = "German" if german else "English"
    long_part = (f"Then write a longer summary of about {n} sentences, split into 2 to 4 sections. Each section has a short, "
                 f"specific heading (h) and flowing narrative text (t), like a magazine story."
                 if n >= 3 else "Do not write a longer summary: return an empty sections list.")
    prompt = (
        f"Summarize the news article below for a private news reader. Write everything in {lang}.\n"
        f"Rules: use your own words, do not copy sentences, no quotes longer than five words. Only state facts that are in "
        f"the article, keep all numbers exactly as given, do not add opinions or outside knowledge.\n"
        f"First write a preview of exactly two sentences with the key news.\n{long_part}\n"
        f"Return JSON with the keys preview and sections.\n\nTITLE: {a['title']}\n\nARTICLE:\n{text[:12000]}"
    )
    req = dict(model=model, stream=False, format=SCHEMA, messages=[{"role": "user", "content": prompt}],
               options=dict(temperature=0.3, num_ctx=8192))
    r = urllib.request.Request(host + "/api/chat", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        out = json.loads(json.loads(resp.read())["message"]["content"])
    preview = " ".join(sentences(clean(out.get("preview", "")))[:2])
    # The model does not reliably respect the length; cut to the target so a summary never becomes a retelling.
    sections, left = [], n
    for s in out.get("sections", []):
        sents = sentences(clean(s.get("t", "")))[:left]
        if not sents:
            break
        sections.append(dict(h=clean(s.get("h", "")), t=" ".join(sents)))
        left -= len(sents)
    return preview, (sections if n >= 3 else [])


def sentences(text):
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„'])", text) if x.strip()]


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data.json"))
    ap.add_argument("--model", default=os.environ.get("NEWSDESK_MODEL", "gemma3:4b"))
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--limit", type=int, default=45)
    ap.add_argument("--dry", action="store_true", help="only list the selected articles, no summaries")
    args = ap.parse_args()

    items = []
    for a in fetch_feeds():
        topics = classify(a)
        if topics:
            items.append(dict(a, topics=topics))
    # Claude and KI first, then the rest; newest first inside each group
    items.sort(key=lambda a: a["published"], reverse=True)
    items.sort(key=lambda a: 0 if {"claude", "ki"} & set(a["topics"]) else 1)
    items = items[: args.limit]
    log(f"{len(items)} articles match a topic")
    if args.dry:
        for a in items:
            print(a["source"][:5], a["published"][5:16], ",".join(a["topics"]), "|", a["title"][:90])
        return

    with cf.ThreadPoolExecutor(8) as ex:
        pages = list(ex.map(read_article, items))

    articles, t0 = [], time.time()
    for i, (a, p) in enumerate(zip(items, pages), 1):
        if p["paid"]:
            log("paywall, skipped:", a["title"])
            continue
        text = p["text"] if len(p["text"]) > len(a["teaser"]) else a["teaser"]
        try:
            preview, sections = summarize(a, text, args.model, args.host)
        except Exception as e:
            log("summary failed, using feed teaser:", a["title"], e)
            preview, sections = a["teaser"], []
        if not preview:
            preview = a["teaser"] or " ".join(sentences(text)[:2])
        full = a["title"] + " " + text
        cos = [k for k, pat in COMPANIES.items() if re.search(pat, full)] if {"claude", "ki"} & set(a["topics"]) else []
        art = dict(source=a["source"], title=a["title"], url=a["url"], published=a["published"], topics=a["topics"],
                   companies=cos, preview=preview, image=p["image"])
        if sections:
            art["summary"] = sections
        articles.append(art)
        log(f"[{i}/{len(items)}] {time.time() - t0:.0f}s  {a['source']}: {a['title'][:70]}")

    doc = dict(updatedAt=NOW.isoformat(timespec="seconds"), generator=f"ollama:{args.model}", articles=articles)
    with open(args.out, "w") as f:
        json.dump(doc, f, ensure_ascii=False)
    log(f"wrote {len(articles)} articles to {args.out}")


if __name__ == "__main__":
    main()
