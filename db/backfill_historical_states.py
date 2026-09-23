"""Split historical states that shared a vote_data key (one-off backfill).

Until Sep 2026 the scraper stored the German Democratic Republic under DEU, South
Yemen under YEM and Zanzibar under TZA. When both states of a pair voted, only one
vote was kept (GDR over FRG, South over North Yemen, Tanganyika over Zanzibar), and
Serbia and Montenegro (2003-06) was not stored at all. This rebuilds exactly those
keys from the scraper's 2025-03-24 export, which kept one column per UN name:

    git show 7034ee9^:pipeline_output/UN_VOTING_DATA_RAW_WITH_TAGS_2025-03-24.csv > raw.csv
    python db/backfill_historical_states.py raw.csv           # dry run: writes the diff
    python db/backfill_historical_states.py raw.csv --apply   # updates both vote tables

Every other key is left as stored. Re-running is safe: once applied it reports 0 records.
"""
import argparse
import collections
import csv
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from un_data_pipeline.turso_http import get_turso_connection  # noqa: E402

VOTES = ('YES', 'NO', 'ABSTAIN')
# (UN names whose vote marks an affected record, {UN name: key} rebuilt in it)
FAMILIES = [
    (('GERMAN DEMOCRATIC REPUBLIC', 'GERMANY, FEDERAL REPUBLIC OF'),
     {'GERMANY, FEDERAL REPUBLIC OF': 'DEU', 'GERMAN DEMOCRATIC REPUBLIC': 'DDR'}),
    (('DEMOCRATIC YEMEN', 'SOUTHERN YEMEN'),
     {'YEMEN': 'YEM', 'DEMOCRATIC YEMEN': 'YMD', 'SOUTHERN YEMEN': 'YMD'}),
    (('ZANZIBAR',), {'TANGANYIKA': 'TZA', 'ZANZIBAR': 'ZAN'}),
    (('SERBIA AND MONTENEGRO',), {'SERBIA AND MONTENEGRO': 'SRB'}),
]


def rebuilt_keys(raw_row):
    """{key: vote or None} for the families that voted in this raw record."""
    keys = {}
    for triggers, names in FAMILIES:
        if not any(raw_row.get(n) in VOTES for n in triggers):
            continue
        for key in set(names.values()):
            votes = [raw_row[n] for n, k in names.items() if k == key and raw_row.get(n) in VOTES]
            if len(votes) > 1:
                raise ValueError(f"{key} has {len(votes)} votes in {raw_row['Link']}")
            keys[key] = votes[0] if votes else None
    return keys


def fetch(conn, table):
    rows, last = [], 0
    while page := conn.execute(f'SELECT id, Link, vote_data FROM {table} '
                               'WHERE id > ? ORDER BY id LIMIT 1000', [last]).fetchall():
        rows += page
        last = page[-1][0]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('raw_csv')
    parser.add_argument('--apply', action='store_true', help='write the changes to Turso')
    parser.add_argument('--out', default='historical_states_changes.csv')
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv, dtype=str)
    names = [c for c in raw.columns if raw[c].isin(VOTES).any()]
    raw = {row['Link']: row for row in raw.to_dict('records')}
    conn = get_turso_connection()
    updates, diff = collections.defaultdict(list), []
    for table in ('un_votes_with_sc', 'un_votes_unga'):
        for _, link, vote_data in fetch(conn, table):
            row = raw.get(link)
            new = rebuilt_keys(row) if row else {}
            votes = json.loads(vote_data)
            changed = {k: v for k, v in new.items() if votes.get(k) != v}
            if not changed:
                continue
            diff += [(table, link, row['Date'], row['Resolution'], k, votes.get(k), v)
                     for k, v in sorted(changed.items())]
            votes.update(changed)
            for vote in VOTES:  # each UN vote of the record stored exactly once
                if sum(v == vote for v in votes.values()) != sum(row[n] == vote for n in names):
                    sys.exit(f'{link}: {vote} total would differ from the raw record; nothing written')
            updates[table].append((json.dumps(votes, sort_keys=True), link))

    with open(args.out, 'w', newline='') as f:
        csv.writer(f).writerows([('table', 'Link', 'Date', 'Resolution', 'key', 'old', 'new')] + diff)
    for table, rows in updates.items():
        print(f'{table}: {len(rows)} records to update')
    print(collections.Counter(d[4] for d in diff if d[0] == 'un_votes_with_sc'))
    print(f'{len(diff)} key changes listed in {args.out}')
    if args.apply:
        for table, rows in updates.items():
            conn.executemany(f'UPDATE {table} SET vote_data = ? WHERE Link = ?', rows)
        print('Applied. Run again without --apply: it should report no records to update.')


if __name__ == '__main__':
    main()
