"""
RAG Colosseum service: POST /query over a 50-passage knowledge base.

Pipeline
  1. Sanitize question (strip hidden / injected instructions).
  2. Hybrid retrieval: dense cosine (MiniLM) + BM25, plus entity/attribute boosts
     so confounder passages (Raft vs Paxos, Postgres vs MySQL...) are separated.
  3. Abstention: off-topic (low dense score), unknown named entity, or the
     passage does not contain the kind of value asked for (port, ms, bits ...).
  4. Extractive answer: best matching sentence(s); multi-entity / multi-attribute
     questions get one sentence per sub-question (cross-passage synthesis).
The answer is always copied from the KB, so injected instructions can never
change what it says.
"""
import json
import math
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

HERE = Path(__file__).parent
KB_PATH = Path(os.getenv("KB_PATH", HERE / "knowledge_base.json"))
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDER = os.getenv("EMBEDDER", "fastembed")  # "fastembed" (ONNX MiniLM) | "st" | "tfidf" (offline tests)
CACHE_DIR = os.getenv("MODEL_CACHE", str(HERE / "model_cache"))
DENSE_MIN = float(os.getenv("DENSE_MIN", "0.08" if EMBEDDER == "tfidf" else "0.27"))
UNKNOWN_ENTITY_OVERRIDE = float(os.getenv("UNKNOWN_ENTITY_OVERRIDE", "0.72"))

KB = json.loads(KB_PATH.read_text())
N = len(KB)

# ---------------------------------------------------------------- text utils
STOP = set("""a an the of to in on for and or is are was were be been being by with as at from
that this these those it its into than then which what who whom whose when where why how
does do did can could should would will shall may might must not no yes if any all both
each per via about above over under between across also there their they them we you your
i me my our us he she his her has have had such so very just only more most other some
use uses used using default value number many much size kind type tell give name say
please according kb knowledge base passage""".split())

ALIASES = {
    "sgd": "stochastic gradient descent", "adam": "adamw", "lsm": "log-structured merge tree",
    "wal": "write-ahead log", "2pc": "two-phase commit", "3pc": "three-phase commit",
    "mlp": "multi-layer perceptron feedforward", "svm": "support vector machine",
    "svms": "support vector machines", "hlc": "hybrid logical clock", "hlcs": "hybrid logical clocks",
    "mdns": "multicast dns", "postgres": "postgresql", "pg": "postgresql", "mongo": "mongodb",
    "ecc": "elliptic curve cryptography", "aes": "advanced encryption standard",
    "tcp": "transmission control protocol", "dns": "domain name system", "tls": "transport layer security",
    "bgp": "border gateway protocol", "ospf": "open shortest path first", "kdf": "pbkdf2",
    "argon2": "secure credential storage argon2", "bcrypt": "", "lb": "", "swim": "swim", "ecdsa": "elliptic curve ecdsa", "jwt": "json web tokens",
    "jwts": "json web tokens", "pkce": "proof key for code exchange", "oauth2": "oauth",
    "http2": "http/2", "http3": "http/3", "h2": "http/2", "h3": "http/3", "cnn": "convolutional",
    "resnet": "residual convolutional resnet", "waf": "web application firewall",
    "aes-gcm": "aes gcm", "chacha20": "chacha20-poly1305", "chacha": "chacha20-poly1305",
    "bn": "batch normalization", "batchnorm": "batch normalization", "layernorm": "layer normalization",
    "ln": "layer normalization", "lr": "learning rate", "ttl": "time-to-live", "rto": "retransmission",
    "gossip": "gossip epidemic", "paxos": "multi-paxos", "b-tree": "b+ tree", "btree": "b+ tree",
    "innodb": "mysql innodb", "sstable": "sstable", "logreg": "logistic regression",
    "gateway": "api gateway", "dht": "consistent hashing", "hrw": "rendezvous hashing highest random weight",
    "dynamodb": "dynamo", "cockroach": "cockroachdb", "e2e": "", "rsa": "rsa",
}

GENERIC_CAPS = set("""what which how when why who whom whose does do did is are was were can could should
would will in on the a an if for according under between compare explain name list please also
both describe give tell given suppose assume consider note i in at by from with as of to and or
yes no true false ok okay hi hello hey kb question answer""".split())

INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|prompt|rule|above|previous|context|guideline)s?"
    r"|system\s*prompt|developer\s*mode|jailbreak|you\s+are\s+now|act\s+as\s+(if|though|an?\s+(ai|assistant|chatbot|unrestricted|different))|pretend|roleplay"
    r"|(respond|reply|answer|output|print|say|return|write)\s+(only\s+)?(with\s+)?(the\s+)?(word|phrase|string|text|exactly|\"|')"
    r"|new\s+instructions?|admin\s+(override|command)|\bassistant\s*:|\bsystem\s*:|\buser\s*:"
    r"|reveal\s+(your|the)\s+(system|prompt|instructions|hidden)|secret\s+(word|password|phrase)|canary|instead\s+(say|output|answer|respond)"
    r"|\bprint\b.{0,20}\b(hacked|pwned|password)|important\s*(note|notice)?\s*(to|for)\s+(the\s+)?(ai|assistant|model|llm)",
    re.I,
)
ZERO_WIDTH = re.compile(r"[​-‏ - ⁠-⁯﻿]")
HIDDEN_BLOCKS = re.compile(r"<!--.*?-->|<[^>]{0,200}>|\[\[.*?\]\]|\{\{.*?\}\}", re.S)


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[+\-/.][a-z0-9]+)*\+?", text.lower())


def content_tokens(text):
    return [t for t in tokenize(text) if t not in STOP and len(t) > 1]


def stem(t):
    for suf in ("ing", "es", "s", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t


def stems(text):
    return set(stem(t) for t in content_tokens(text))


def split_sentences(p):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if s.strip()]


def sanitize(q: str) -> str:
    q = ZERO_WIDTH.sub("", q or "")
    q = HIDDEN_BLOCKS.sub(" ", q)
    # drop bracketed / parenthesised asides that carry injected instructions
    q = re.sub(r"[\(\[\{][^\)\]\}]*[\)\]\}]", lambda m: " " if INJECTION_RE.search(m.group()) else m.group(), q)
    parts = re.split(r"(?<=[.?!;:\n])\s+|\n+|\s[-–—]{1,2}\s", q)
    kept = [p for p in parts if p.strip() and not INJECTION_RE.search(p)]
    if not kept:  # everything looked like an injection: keep the part before the trigger
        m = INJECTION_RE.search(q)
        kept = [q[: m.start()]] if m else [q]
    out = " ".join(kept)
    return re.sub(r"\s+", " ", out).strip()


def expand_aliases(q: str) -> str:
    extra = [ALIASES[t] for t in tokenize(q) if t in ALIASES and ALIASES[t]]
    return q + (" " + " ".join(extra) if extra else "")


# ---------------------------------------------------------------- attributes
# (question trigger, sentence pattern, strong?) strong attributes are used for abstention
ATTRS = {
    "port": (r"\bports?\b|listen|bind", r"\bport \d+", True),
    "lr": (r"learning[\s-]*rate|\blr\b", r"learning rate", True),
    "quorum": (r"quorum|how many (nodes|acceptors|replicas|members|participants)", r"quorum", True),
    "bits": (r"\bbits?\b|key (size|length)|-bit\b|how (big|long|large)\b.*\bkeys?\b|key strength", r"\d+ bits|\d+-bit", True),
    "hash": (r"hash(ing)? (function|algorithm|primitive)|digest|\bsha\b|blake|which hash", r"SHA-?\d|BLAKE2", True),
    "ms": (r"time-?outs?|interval|latency|how long|duration|\bms\b|millisecond|heartbeat|deadline|\bwait|lease|"
           r"skew|drift|penalty window|grant window|dead interval|rto|keepalive|within how", r"\d+ ?ms\b", True),
    "batch": (r"batch(es)? size|mini-?batch(es)? (size|of)|how many (samples|examples|instances)|batch sizes?",
              r"\d+ (samples|training examples|instances)|batch sizes? of \d+|mini-batches of \d+", True),
    "retries": (r"retr(y|ies)", r"retr", True),
    "layers": (r"\b(layers?|blocks?|depth|stages?)\b", r"\b\d+\b[^.]*\b(layers?|blocks?)\b", False),
    "ratio": (r"threshold|ratio|probability|fraction|utili[sz]ation|saturation|metric|cutoff|accrual|skew|"
              r"burst|capacity limit|variance", r"\b0\.\d+", False),
    "chunk": (r"chunks?|groups?|records|entries|bytes|support vectors|per (fsync|frame|flush|transmission)",
              r"\b\d+ (records?|record chunks|log entries|bytes|support vectors)", False),
    "tokens": (r"window length|sequence length|tokens? per|how many tokens", r"\d+ tokens", False),
}


def question_attrs(q):
    ql = q.lower()
    return [k for k, (trig, _, _) in ATTRS.items() if re.search(trig, ql)]


def has_attr(text, attr):
    return re.search(ATTRS[attr][1], text, re.I) is not None


# ---------------------------------------------------------------- KB indexes
SENTS = [split_sentences(p) for p in KB]
SUBJECT = [SENTS[i][0] for i in range(N)]
HEAD = [" ".join(SUBJECT[i].split()[:6]) for i in range(N)]  # topic phrase of each passage
HEAD_STEMS = [stems(h) for h in HEAD]
_head_df = Counter(t for hs in HEAD_STEMS for t in hs)
RARE_HEAD = {t for t, c in _head_df.items() if c <= 3}
KB_TOKENS = set(t for p in KB for t in tokenize(p))
KB_TOKENS |= set(part for t in list(KB_TOKENS) for part in re.split(r"[+\-/.]", t) if part)

# entity vocabulary: capitalised / technical terms that appear in the KB
ENTITY_VOCAB = set()
for p in KB:
    for sent in split_sentences(p):
        for k, w in enumerate(re.findall(r"[A-Za-z][A-Za-z0-9+\-/.]*[A-Za-z0-9+]", sent)):
            techy = re.search(r"[A-Z].*[A-Z]|\d|^[A-Z][a-z]+[A-Z]", w)
            if techy or (w[0].isupper() and k > 0 and w.lower() not in STOP):
                ENTITY_VOCAB.add(w.lower())
LOWER_WORDS = set(w for p in KB for w in re.findall(r"(?<![A-Za-z])[a-z][a-z0-9+\-/.]*", p))
ENTITY_VOCAB = {e for e in ENTITY_VOCAB if e not in LOWER_WORDS or re.search(r"\d", e)}
ENTITY_VOCAB |= {"etcd", "raft", "saga", "sagas", "paxos", "multi-paxos", "redis", "memcached", "postgresql",
                 "mongodb", "cassandra", "swim", "pbkdf2", "adamw", "chacha20-poly1305", "oauth", "quic", "tls"}
ENTITY_VOCAB -= GENERIC_CAPS | {"first", "policy", "post", "proof", "tree", "forgery", "linux", "json", "http"}


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.docs = [content_tokens(d) for d in docs]
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q):
        qt = content_tokens(q)
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in qt:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out[i] = s
        return out


BM = BM25(KB)


class Embedder:
    def __init__(self):
        flat = [s for ss in SENTS for s in ss]
        if EMBEDDER == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)).fit(KB + flat)
            self._enc = lambda xs: self.vec.transform(xs).toarray()
        elif EMBEDDER == "fastembed":
            from fastembed import TextEmbedding
            self.model = TextEmbedding(MODEL_NAME, cache_dir=CACHE_DIR, threads=1)
            self._enc = lambda xs: np.stack(list(self.model.embed(list(xs), batch_size=64)))
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME, device="cpu")
            self._enc = lambda xs: self.model.encode(xs, batch_size=64, normalize_embeddings=True,
                                                     show_progress_bar=False)
        self.lock = threading.Lock()
        self.P = self.norm(self._enc(KB))
        self.S = [self.norm(self._enc(ss)) for ss in SENTS]

    @staticmethod
    def norm(m):
        m = np.asarray(m, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def encode(self, q):
        with self.lock:
            return self.norm(self._enc([q]))[0]


EMB = Embedder()


# ---------------------------------------------------------------- retrieval
def question_entities(q):
    """(known entities, unknown capitalised terms) mentioned in the question."""
    known, unknown = [], []
    words = re.findall(r"[A-Za-z][A-Za-z0-9+\-/.']*[A-Za-z0-9+]|[A-Za-z]", q)
    for i, w in enumerate(words):
        w = re.sub(r"'s$", "", w).rstrip(".")
        lw = w.lower()
        if lw in GENERIC_CAPS or lw in STOP:
            continue
        techy = bool(re.search(r"[A-Z].*[A-Z]|\d[A-Za-z]|[A-Za-z]\d|^[A-Z][a-z]+[A-Z]", w))
        capital = w[0].isupper() and i > 0
        if lw in ALIASES:
            known.append(lw)
            continue
        if lw in ENTITY_VOCAB:
            known.append(lw)
            continue
        if not (techy or capital):
            continue
        in_kb = lw in KB_TOKENS or any(
            (len(lw) >= 4 and (t.startswith(lw) or lw.startswith(t)) and len(t) >= 4) for t in ENTITY_VOCAB
        ) or all(part in KB_TOKENS for part in re.split(r"[+\-/.]", lw) if part)
        (known if in_kb else unknown).append(lw)
    return known, unknown


def entity_in(ent, text):
    tl = text.lower()
    keys = [ent]
    if ALIASES.get(ent):
        keys.append(ALIASES[ent])
        keys += [w for w in ALIASES[ent].split() if w in ENTITY_VOCAB]
    for k in keys:
        if len(k) >= 2 and re.search(r"(?<![a-z0-9])" + re.escape(k), tl):
            return True
    return False


def rank(q_raw):
    q = expand_aliases(q_raw)
    qv = EMB.encode(q)
    dense = np.clip(EMB.P @ qv, 0, 1)
    bm = BM.scores(q)
    bmn = bm / bm.max() if bm.max() > 0 else bm
    known, unknown = question_entities(q_raw)
    attrs = question_attrs(q_raw)
    final = 0.65 * dense + 0.35 * bmn
    qs = stems(q)
    qvals = set(re.findall(r"\b\d+(?:\.\d+)?(?:ms)?\b", q_raw)) - {"1", "2", "3"}
    for i in range(N):
        boost = min(0.3, 0.15 * len(qs & HEAD_STEMS[i] & RARE_HEAD))
        if qvals:
            pv = set(re.findall(r"\b\d+(?:\.\d+)?(?:ms)?\b", KB[i]))
            boost += 0.4 * len(qvals & pv)
        for e in known:
            if entity_in(e, HEAD[i]):
                boost += 0.2
            elif entity_in(e, SUBJECT[i]):
                boost += 0.08
            elif entity_in(e, KB[i]):
                boost += 0.03
        strong = [a for a in attrs if ATTRS[a][2]]
        if strong:
            boost += 0.06 if any(has_attr(KB[i], a) for a in strong) else -0.08
        final[i] += boost
    order = np.argsort(-final, kind="stable")
    return order, final, dense, known, unknown, attrs, qv


def best_sentence(pid, qv, attrs, q):
    sents, S = SENTS[pid], EMB.S[pid]
    sims = S @ qv
    qt = stems(q) - HEAD_STEMS[pid]
    best, best_s = 0, -9
    for j, s in enumerate(sents):
        sc = float(sims[j])
        sc += 0.1 * len(qt & stems(s))
        if attrs:
            sc += 0.6 * sum(has_attr(s, a) for a in attrs)
        if re.search(r"\d", s) and re.search(r"how|what|which|value|many|much", q, re.I):
            sc += 0.1
        if j == 0 and not re.search(r"\bwhat (is|are)\b.*\?$|define|explain|purpose", q, re.I):
            sc -= 0.05
        if sc > best_s:
            best, best_s = j, sc
    return sents[best]


def abstain(conf=0.1):
    return {"answer": None, "sources": [], "scores": [], "confidence": round(conf, 4)}


@lru_cache(maxsize=2048)
def answer_question(question: str, top_k: int):
    q = sanitize(question)
    if len(content_tokens(q)) == 0:
        return abstain(0.05)

    order, final, dense, known, unknown, attrs, qv = rank(q)
    top = int(order[0])
    top_dense = float(dense[top])
    strong = [a for a in attrs if ATTRS[a][2]]

    # --- abstention rules
    subj_hit = any(entity_in(e, HEAD[top]) for e in known)
    if top_dense < (DENSE_MIN * 0.5 if subj_hit else DENSE_MIN):
        return abstain(min(0.35, top_dense))
    if unknown and top_dense < UNKNOWN_ENTITY_OVERRIDE:
        # named something (e.g. Kafka, RMSprop) that is not in the KB at all
        return abstain(0.15)
    if known and not any(entity_in(e, KB[top]) for e in known):
        return abstain(0.2)
    if strong and not any(has_attr(KB[top], a) for a in strong):
        return abstain(0.2)

    # --- answer synthesis
    answer_pids, sentences = [top], []
    # multi-entity question -> one sub-answer per entity ("Raft and Multi-Paxos", "Redis vs Memcached")
    ent_pids = []
    for e in dict.fromkeys(known):
        cands = [int(i) for i in order[:8] if entity_in(e, HEAD[int(i)])]
        if cands:
            ent_pids.append(cands[0])
    ent_pids = list(dict.fromkeys(ent_pids))
    if len(ent_pids) >= 2:
        answer_pids = [p for p in order if int(p) in ent_pids][:3]
        answer_pids = [int(p) for p in answer_pids]
    for pid in answer_pids:
        if strong and not any(has_attr(KB[pid], a) for a in strong):
            continue
        # one sentence per requested attribute (e.g. "port and idle timeout")
        groups = [[a] for a in attrs if any(has_attr(s, a) for s in SENTS[pid])] or [attrs]
        for g in groups[:3]:
            s = best_sentence(pid, qv, g, q)
            if s not in sentences:
                sentences.append(s)
    if not sentences:
        return abstain(0.2)
    # reverse lookups ("which database listens on 27017?") need the passage's name in the answer
    if not known and SUBJECT[answer_pids[0]] not in sentences:
        sentences.insert(0, SUBJECT[answer_pids[0]])

    # keep strongest attribute-matching sentences only (max 3)
    answer = " ".join(sentences[:4])

    # sources: answer passages first, then the rest of the ranking, descending scores
    k = max(1, min(int(top_k or 3), N))
    srcs = list(dict.fromkeys(answer_pids + [int(i) for i in order]))[:k]
    srcs.sort(key=lambda i: -final[i])
    scores = [round(float(np.tanh(max(final[i], 0.0))), 4) for i in srcs]
    conf = float(np.clip(0.5 + 0.5 * np.tanh(final[top]), 0.65, 0.99))
    return {"answer": answer, "sources": srcs, "scores": scores, "confidence": round(conf, 4)}


# ---------------------------------------------------------------- API
app = FastAPI(title="RAG Colosseum")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def always_cors(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"  # even when no Origin header was sent
    return resp


@app.get("/")
def root():
    return {"status": "ok", "passages": N, "endpoint": "POST /query", "embedder": EMBEDDER}


from collections import deque
import time as _time
QUERY_LOG = deque(maxlen=200)


@app.get("/log")
def get_log():
    return list(QUERY_LOG)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    question = str(body.get("question") or body.get("query") or "")
    try:
        top_k = int(body.get("top_k", 3))
    except Exception:
        top_k = 3
    try:
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(answer_question, question, top_k)
    except Exception:
        result = abstain(0.0)
    entry = {"t": _time.strftime("%H:%M:%S"), "body": body, "response": dict(result)}
    QUERY_LOG.append(entry)
    print("QUERY", json.dumps(entry), flush=True)
    return JSONResponse(dict(result))
