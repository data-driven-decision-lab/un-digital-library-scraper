"""Checks that topic_votes_yearly keeps UNBIS labels that contain commas.

Run from the repo root: python -m unittest discover tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from un_data_pipeline import dashboard_data_pipeline as dp  # noqa: E402


class TopicVotesTest(unittest.TestCase):
    def test_comma_labels_are_counted_whole(self):
        df = dp.pd.DataFrame([
            # 'HUMAN RIGHTS' is not a known item of this subcategory: it must stay an
            # item and must not swallow the INTERNATIONAL RELATIONS path after it.
            {'Year': 2025, 'Resolution': 'A/RES/80/1', 'USA': 'YES', 'FRA': 'NO',
             'tags': 'POLITICAL AND LEGAL QUESTIONS, POLITICAL CONDITIONS, INSTITUTIONS, MOVEMENTS, '
                     'HUMAN RIGHTS, POLITICAL AND LEGAL QUESTIONS, INTERNATIONAL RELATIONS, DIPLOMATIC RELATIONS'},
            {'Year': 2025, 'Resolution': 'A/RES/80/2', 'USA': 'YES', 'FRA': 'ABSTAIN',
             'tags': 'AGRICULTURE, FORESTRY AND FISHING, AGRICULTURAL SCIENCES, ENGINEERING AND EDUCATION'},
        ])
        got = {(r.Country, r.TopicTag): (r.YesVotes_Topic, r.NoVotes_Topic, r.AbstainVotes_Topic)
               for r in dp.generate_topic_votes(df).itertuples()}
        self.assertEqual({tag for _, tag in got}, {
            'POLITICAL AND LEGAL QUESTIONS', 'POLITICAL CONDITIONS, INSTITUTIONS, MOVEMENTS',
            'INTERNATIONAL RELATIONS', 'AGRICULTURE, FORESTRY AND FISHING',
            'AGRICULTURAL SCIENCES, ENGINEERING AND EDUCATION',
        })
        self.assertEqual(got[('FRA', 'POLITICAL CONDITIONS, INSTITUTIONS, MOVEMENTS')], (0, 1, 0))
        self.assertEqual(got[('FRA', 'AGRICULTURE, FORESTRY AND FISHING')], (0, 0, 1))
        self.assertEqual(got[('USA', 'POLITICAL AND LEGAL QUESTIONS')], (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
