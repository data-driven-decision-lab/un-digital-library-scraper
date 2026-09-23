"""Checks that scraper failures are recorded and surface as a non-zero exit.

Run from the repo root: python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

# The module needs GEMINI_API_KEY at import time and writes logs/ and ../data/
# relative to the cwd, so import it from a throwaway directory.
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_cwd = os.path.join(tempfile.mkdtemp(), "run")
os.makedirs(os.path.join(_cwd, "logs"))
os.chdir(_cwd)
os.environ.setdefault("GEMINI_API_KEY", "test")
sys.path.insert(0, _src)
from un_data_pipeline import scraper_pipeline as sp  # noqa: E402

REC = "https://digitallibrary.un.org/record/"


class RunStatusTest(unittest.TestCase):
    def test_waf_cookie_is_installed_in_every_new_browser_before_navigation(self):
        browsers = [mock.Mock(), mock.Mock()]
        for browser in browsers:
            browser.execute_cdp_cmd.return_value = {"success": True}
        with mock.patch.dict(os.environ, {"AWS_WAF_TOKEN": " test-token "}), \
             mock.patch.object(sp, "ChromeDriverManager"), \
             mock.patch.object(sp, "Service"), \
             mock.patch.object(sp.os, "chmod"), \
             mock.patch.object(sp.webdriver, "Chrome", side_effect=browsers):
            for browser in browsers:
                self.assertIs(sp.get_driver(), browser)
                browser.get.assert_not_called()
                browser.execute_cdp_cmd.assert_called_once_with("Network.setCookie", {
                    "name": "aws-waf-token", "value": "test-token",
                    "url": "https://digitallibrary.un.org/", "path": "/", "secure": True,
                })

    def test_local_browser_without_token_does_not_set_cookie(self):
        with mock.patch.dict(os.environ, {"AWS_WAF_TOKEN": ""}), \
             mock.patch.object(sp, "ChromeDriverManager"), \
             mock.patch.object(sp, "Service"), \
             mock.patch.object(sp.os, "chmod"), \
             mock.patch.object(sp.webdriver, "Chrome") as chrome:
            sp.get_driver()
            chrome.return_value.execute_cdp_cmd.assert_not_called()

    def test_cookie_failure_closes_browser_and_hides_sensitive_error(self):
        for outcome in [{"success": False}, RuntimeError("secret-token")]:
            browser = mock.Mock()
            if isinstance(outcome, Exception):
                browser.execute_cdp_cmd.side_effect = outcome
            else:
                browser.execute_cdp_cmd.return_value = outcome
            with mock.patch.dict(os.environ, {"AWS_WAF_TOKEN": "secret-token"}), \
                 mock.patch.object(sp, "ChromeDriverManager"), \
             mock.patch.object(sp, "Service"), \
                 mock.patch.object(sp.os, "chmod"), \
                 mock.patch.object(sp.webdriver, "Chrome", return_value=browser):
                with self.assertRaisesRegex(RuntimeError, "Could not configure") as error:
                    sp.get_driver()
            browser.quit.assert_called_once()
            self.assertNotIn("secret-token", str(error.exception))
            self.assertTrue(error.exception.__suppress_context__)

    def test_main_records_success_and_failure(self):
        for outcome, expected in [(None, ("success",)), (RuntimeError("boom"), ("failed", "boom"))]:
            def run():
                if outcome:
                    raise outcome
            with mock.patch.object(sp, "start_scraper_log"), \
                 mock.patch.object(sp, "finish_scraper_log") as finish, \
                 mock.patch.object(sp, "run_scraper", run):
                if outcome:
                    self.assertRaises(RuntimeError, sp.main)
                else:
                    sp.main()
            finish.assert_called_once_with(*expected)

    def test_turso_unreachable_fails_instead_of_rescraping_everything(self):
        conn = mock.Mock(side_effect=RuntimeError("Hrana: unexpected EOF"))
        with mock.patch.object(sp, "get_turso_connection", conn), \
             mock.patch.object(sp.time, "sleep"), \
             mock.patch.object(sp, "update_scraper_log"), \
             mock.patch.object(sp, "get_driver") as get_driver:
            self.assertRaises(RuntimeError, sp.run_scraper)
        self.assertEqual(conn.call_count, 3)
        get_driver.assert_not_called()

    def test_bot_challenge_is_named_in_the_error(self):
        driver = mock.Mock(page_source='<script src="https://x.token.awswaf.com/x/challenge.js">')
        with mock.patch.object(sp, "get_links_from_turso", return_value=set()), \
             mock.patch.object(sp, "get_driver", return_value=driver), \
             mock.patch.object(sp, "get_available_years", return_value=[]), \
             mock.patch.object(sp, "update_scraper_log"), \
             mock.patch.object(sp.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "AWS WAF bot challenge"):
                sp.run_scraper()

    def test_final_step_uploads_only_rows_missing_from_turso(self):
        rows = [{"Link": REC + "1", "Scrape_Year": 2026}, {"Link": REC + "2?ln=en", "Scrape_Year": 2026}]
        with mock.patch.object(sp, "get_links_from_turso", side_effect=[set(), {REC + "1"}]), \
             mock.patch.object(sp, "get_driver"), \
             mock.patch.object(sp, "get_available_years", return_value=[{"year": 2026, "count": 2}]), \
             mock.patch.object(sp, "select_year_facet", side_effect=lambda d, y: (True, d)), \
             mock.patch.object(sp, "collect_links_for_year", return_value=[REC + "1", REC + "2"]), \
             mock.patch.object(sp, "batch_scrape_resolutions", return_value=(rows, [])), \
             mock.patch.object(sp, "checkpoint_progress"), \
             mock.patch.object(sp, "clear_filters"), \
             mock.patch.object(sp, "update_scraper_log"), \
             mock.patch.object(sp.time, "sleep"), \
             mock.patch.object(sp, "process_and_upload_data") as upload:
            sp.run_scraper()
        uploaded = upload.call_args.args[0]
        self.assertEqual(list(uploaded["Link"]), [REC + "2?ln=en"])


if __name__ == "__main__":
    unittest.main()
