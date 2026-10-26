import json, collections, glob, sys
# Usage: spread.py CQA_DIR QASPER_DIR (ConditionalQA's v1_0, and the Qasper
# pages the Qasper harness caches).
cqa, qasper = sys.argv[1], sys.argv[2]
# ConditionalQA: section path of every content line.
docs = {d['url']: d for d in json.load(open(f'{cqa}/documents.json'))}
def paths(doc):
    stack, out = [], {}
    for line in doc['contents']:
        tag = line[1:3]
        if tag in ('h1','h2','h3','h4'):
            lvl = int(tag[1]); stack = [s for s in stack if s[0] < lvl] + [(lvl, line)]
        else:
            out.setdefault(line, tuple(s[1] for s in stack))
    return out
def classify(secs):
    secs = list(set(secs))
    if len(secs) <= 1: return 'one section'
    anc = any(a != b and b[:len(a)] == a for a in secs for b in secs)
    sib = any(a != b and len(a) == len(b) and a[:-1] == b[:-1] for a in secs for b in secs)
    return 'section + its ancestor' if anc else ('siblings' if sib else 'unrelated sections')
for split in ['dev', 'train']:
    qs = json.load(open(f'{cqa}/{split}.json'))
    c = collections.Counter(); missing = 0
    for q in qs:
        if q['not_answerable'] or not q['evidences']: c['no evidence'] += 1; continue
        p = paths(docs[q['url']]); secs = []
        for e in q['evidences']:
            if e in p: secs.append(p[e])
            else: missing += 1
        c[classify(secs)] += 1
    print('ConditionalQA', split, len(qs), dict(c), 'evidence lines not found', missing)
# Qasper: by section of each cited paragraph, best annotator = the one citing most sections.
rows = []
for f in sorted(glob.glob(f'{qasper}/test-*.json')):
    rows += [r['row'] for r in json.load(open(f))['rows']]
c = collections.Counter()
for row in rows:
    names = row['full_text']['section_name']; paras = row['full_text']['paragraphs']
    where = {}
    for n, ps in zip(names, paras):
        path = tuple(x.strip() for x in (n or '').split(':::') if x.strip())
        for p in ps: where.setdefault(' '.join(p.split()), path)
    for answers in row['qas']['answers']:
        best = None
        for a in answers['answer']:
            if a['unanswerable']: continue
            secs = [where[' '.join(e.split())] for e in a['evidence'] if not e.startswith('FLOAT') and ' '.join(e.split()) in where]
            if secs and (best is None or len(set(secs)) > len(set(best))): best = secs
        if best: c[classify(best)] += 1
print('Qasper test', dict(c))
