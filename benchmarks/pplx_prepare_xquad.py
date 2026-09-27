"""Prepare the fixed XQuAD-R query IDs for the pplx Greek replay.

The published corpus stays in PAMIN_EVAL_HOME/xquad-r; no dataset is vendored.
"""

import argparse
import json
from pathlib import Path

LANGUAGES = ('ar', 'de', 'el', 'en', 'es', 'hi', 'ru', 'th', 'tr', 'vi', 'zh')
EXPECTED_FINGERPRINT = '24ad7f1862182925'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-home', type=Path, required=True)
    home = parser.parse_args().eval_home
    documents = []
    asked = {}
    answers = {}

    for language in LANGUAGES:
        corpus = json.loads((home / 'xquad-r' / f'{language}.json').read_text())
        paragraphs = (p for article in corpus['data'] for p in article['paragraphs'])
        for position, paragraph in enumerate(paragraphs):
            for index, sentence in enumerate(paragraph['sentences']):
                documents.append({
                    'key': f'{language}:{position}:{index}',
                    'text': sentence.lstrip('\ufeff'),
                    'language': language,
                })
            for question in paragraph['qas']:
                question_id = question['id']
                start = question['answers'][0]['answer_start']
                index = next(
                    i for i, (begin, end) in enumerate(paragraph['sentence_breaks'])
                    if begin <= start < end
                )
                asked.setdefault(question_id, {})[language] = question['question']
                answers.setdefault(question_id, {})[language] = f'{language}:{position}:{index}'

    queries = []
    for position, question_id in enumerate(sorted(answers)):
        if len(answers[question_id]) != len(LANGUAGES):
            continue
        language = LANGUAGES[position % len(LANGUAGES)]
        queries.append({
            'id': question_id,
            'language': language,
            'text': asked[question_id][language],
            'answers': answers[question_id],
        })

    fingerprint = 0xCBF29CE484222325
    for document in documents:
        for byte in (document['key'] + document['text']).encode():
            fingerprint = ((fingerprint ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    assert f'{fingerprint:016x}' == EXPECTED_FINGERPRINT
    assert len(documents) == 13_014 and len(queries) == 1_190
    assert len({document['key'] for document in documents}) == len(documents)
    destination = home / 'pplx-xquad-manifest.json'
    destination.write_text(json.dumps({'documents': documents, 'queries': queries}, ensure_ascii=False))
    print(f'{destination}: {len(documents)} documents, {len(queries)} paired queries')


if __name__ == '__main__':
    main()
