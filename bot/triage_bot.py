# Issue Triage Bot - STANDALONE drop-in version.
#
# Copy this whole bot/ folder + .github/workflows/issue-triage-bot.yml into
# ANY repository (for example your fork of kubernetes/kubernetes), commit,
# and every newly opened issue gets a triage comment + kind/* label.
# It has no dependency on the FastAPI/React project: model training,
# prediction and duplicate search all live in this one file, using the
# bundled dataset (bot/data/kubernetes_issues.csv.gz, 9,862 real issues).
#
# Local test (no GitHub writes):
#   ISSUE_TITLE="..." ISSUE_BODY="..." DRY_RUN=1 python3 bot/triage_bot.py
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BOT_DIR = Path(__file__).resolve().parent
DATA_PATH = BOT_DIR / 'data' / 'kubernetes_issues.csv.gz'
MODEL_PATH = BOT_DIR / 'models' / 'triage_model.joblib'
API = 'https://api.github.com'
# duplicate links point at the repo the training data came from
DATASET_REPO = os.getenv('DATASET_REPO', 'kubernetes/kubernetes')
THRESHOLD = float(os.getenv('DUPLICATE_THRESHOLD', '0.65'))
LABEL_COLORS = {'bug': 'd73a4a', 'feature': 'a2eeef', 'documentation': '0075ca',
                'regression': 'e4e669', 'failing-test': 'f9d0c4', 'cleanup': 'cfd3d7',
                'support': 'd876e3', 'flake': 'fbca04'}

# --------------------------------------------------------------------------
# preprocessing - same logic as the notebook / main project
# --------------------------------------------------------------------------
KIND_ORDER = ['kind/bug', 'kind/feature', 'kind/documentation', 'kind/regression',
              'kind/failing-test', 'kind/cleanup', 'kind/support', 'kind/flake']
STOP_WORDS = set("a about above after again against all am an and any are as at be because been before being below between both but by could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself let me more most my nor of on once only or other our ours ourselves out over own s same she should so some such t than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why with would you your yours yourself yourselves".split())
_TOKEN_RE = re.compile(r'\b[a-z]+\b')


def get_kind(labels_str):
    labs = [l.strip() for l in str(labels_str).split(',') if l.strip()]
    for k in KIND_ORDER:
        if k in labs:
            return k.replace('kind/', '')
    return None


def clean_text(text):
    text = re.sub(r'<details>.*?</details>', ' ', str(text), flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = _TOKEN_RE.findall(text)
    return ' '.join(t for t in tokens if t not in STOP_WORDS and len(t) > 2)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class Model:
    def __init__(self):
        self.vectorizer = None
        self.clf = None
        self.df = None
        self.matrix = None

    def train(self):
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
        df = df.dropna(subset=['title']).copy()
        df['body'] = df['body'].fillna('')
        df['labels'] = df['labels'].fillna('')
        df['kind'] = df['labels'].apply(get_kind)
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        df['clean_text'] = (df['title'].fillna('') + ' ' + df['body']).apply(clean_text)
        df = df[df['clean_text'].str.split().str.len() >= 3].reset_index(drop=True)
        self.df = df
        self.vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.9,
                                          ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(df['clean_text'])
        labelled = df[df['kind'].notna()]
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.clf.fit(self.vectorizer.transform(labelled['clean_text']), labelled['kind'])
        return self

    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'vectorizer': self.vectorizer, 'clf': self.clf,
                     'numbers': self.df['number'].tolist(),
                     'titles': self.df['title'].tolist(),
                     'matrix': self.matrix}, MODEL_PATH)

    def load_or_train(self):
        if MODEL_PATH.exists():
            try:
                b = joblib.load(MODEL_PATH)
                self.vectorizer, self.clf = b['vectorizer'], b['clf']
                self.matrix = b['matrix']
                self.df = pd.DataFrame({'number': b['numbers'], 'title': b['titles']})
                return self
            except Exception:
                pass
        self.train()
        self.save()
        return self

    def predict(self, title, body):
        vec = self.vectorizer.transform([clean_text(f'{title} {body or ""}')])
        kind = str(self.clf.predict(vec)[0])
        proba = self.clf.predict_proba(vec)[0]
        probabilities = {str(c): round(float(p), 4)
                         for c, p in zip(self.clf.classes_, proba)}
        sims = cosine_similarity(vec, self.matrix).flatten()
        order = np.argsort(sims)[::-1][:5]
        duplicates = [{'number': int(self.df.loc[i, 'number']),
                       'title': str(self.df.loc[i, 'title']),
                       'score': round(float(sims[i]), 3)}
                      for i in order if sims[i] >= THRESHOLD]
        similar = [{'number': int(self.df.loc[i, 'number']),
                    'title': str(self.df.loc[i, 'title']),
                    'score': round(float(sims[i]), 3)} for i in order]
        arr = vec.toarray()[0]
        names = np.array(self.vectorizer.get_feature_names_out())
        top_terms = [str(names[i]) for i in np.argsort(arr)[::-1][:8] if arr[i] > 0]
        return {'predicted_kind': kind, 'confidence': probabilities.get(kind, 0.0),
                'probabilities': dict(sorted(probabilities.items(), key=lambda kv: -kv[1])),
                'duplicates': duplicates, 'similar': similar,
                'top_terms': top_terms, 'threshold': THRESHOLD}


# --------------------------------------------------------------------------
# comment + GitHub writes
# --------------------------------------------------------------------------
def build_comment(r):
    lines = ['## 🤖 Issue Triage Bot', '']
    lines.append(f"**Predicted type:** `kind/{r['predicted_kind']}` "
                 f"(confidence {r['confidence'] * 100:.0f}%, model: Random Forest)")
    lines += ['', '**Top predictions:**', '', '| Type | Confidence |', '|---|---|']
    for k, p in list(r['probabilities'].items())[:3]:
        lines.append(f'| kind/{k} | {p * 100:.1f}% |')
    lines.append('')
    if r['duplicates']:
        lines.append(f"**Possible duplicates** (similarity ≥ {r['threshold']}):")
        lines.append('')
        for d in r['duplicates']:
            lines.append(f"- [{d['score']:.3f}] [#{d['number']} — {d['title']}]"
                         f"(https://github.com/{DATASET_REPO}/issues/{d['number']})")
        lines.append('')
    elif r['similar']:
        top = r['similar'][0]
        lines.append(f"No duplicates above {r['threshold']}. Closest existing issue: "
                     f"[#{top['number']} — {top['title']}]"
                     f"(https://github.com/{DATASET_REPO}/issues/{top['number']}) "
                     f"at {top['score']:.3f}.")
        lines.append('')
    if r['top_terms']:
        lines.append(f"**Key signals:** {', '.join(r['top_terms'])}")
        lines.append('')
    lines.append(f"I've applied the `kind/{r['predicted_kind']}` label — "
                 "a maintainer can change it if it's wrong.")
    lines.append('')
    lines.append('<sub>Automated triage: TF-IDF + Random Forest trained on '
                 '9,862 kubernetes/kubernetes issues (89.3% accuracy). '
                 'Predictions are suggestions, not decisions.</sub>')
    return '\n'.join(lines)


def _api(method, path, payload, token):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={'Accept': 'application/vnd.github+json',
                 'User-Agent': 'issue-triage-bot',
                 'Authorization': f'Bearer {token}',
                 'Content-Type': 'application/json',
                 'X-GitHub-Api-Version': '2022-11-28'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode('utf-8') or 'null')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', errors='ignore')[:300]


def main():
    title = os.environ['ISSUE_TITLE']
    body = os.getenv('ISSUE_BODY') or ''
    number = int(os.getenv('ISSUE_NUMBER', '0'))
    repo = os.getenv('REPO_FULL_NAME', DATASET_REPO)
    token = os.getenv('GITHUB_TOKEN', '')
    dry_run = os.getenv('DRY_RUN') == '1'

    result = Model().load_or_train().predict(title, body)
    comment = build_comment(result)
    with open(os.getenv('COMMENT_OUT', 'bot_comment.md'), 'w') as f:
        f.write(comment)

    if dry_run:
        print(comment)
        print(f"\n[DRY_RUN] would post the above on {repo}#{number} "
              f"and apply label kind/{result['predicted_kind']}")
        return
    if not token:
        raise RuntimeError('GITHUB_TOKEN is required to post (or set DRY_RUN=1)')
    status, resp = _api('POST', f'/repos/{repo}/issues/{number}/comments',
                        {'body': comment}, token)
    if status not in (200, 201):
        raise RuntimeError(f'comment failed ({status}): {resp}')
    kind = result['predicted_kind']
    _api('POST', f'/repos/{repo}/labels',
         {'name': f'kind/{kind}', 'color': LABEL_COLORS.get(kind, 'ededed')}, token)
    status, _ = _api('POST', f'/repos/{repo}/issues/{number}/labels',
                     {'labels': [f'kind/{kind}']}, token)
    print(f'commented on {repo}#{number}; label kind/{kind} applied: '
          f'{status in (200, 201)}')


if __name__ == '__main__':
    main()
