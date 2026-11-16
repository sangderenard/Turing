"""Disposable research case tests; no live service or credentials."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import accounts, ticket_research
from tests.test_issue_tickets import users

class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="research-tests-")
        self.addCleanup(self.temp.cleanup)
        self.accounts = accounts.Accounts(Path(self.temp.name) / "accounts.db")
        self.addCleanup(self.accounts._conn.close)
        users(self.accounts)
        self.service = ticket_research.Service(self.accounts)
        self.tokens = {i: "fixture-token-" + str(i) for i in range(1, 5)}

    def create(self):
        return self.service.create(self.tokens[1], {"title": "Measure the flow",
            "goal": "Find a reliable user flow and preserve the existing contracts.",
            "ticket_ids": []})["id"]

    def enter(self, case, *ids):
        for actor in ids:
            self.service.affirmation(self.tokens[actor], case, {"phrase": f"I will check the evidence {actor}"})

    def doc(self, case, actor, body="A design document"):
        self.service.document(self.tokens[actor], case, {
            "title": f"Document {actor}", "body": body, "source_kind": "text", "source_ref": None})

    def test_entry_documents_brief_and_promotion(self):
        case = self.create()
        self.assertTrue(self.service.snapshot(self.tokens[2], case)["requires_affirmation"])
        with self.assertRaises(ticket_research.ResearchError) as error:
            self.doc(case, 2)
        self.assertEqual(error.exception.status, 403)
        self.enter(case, 1, 2, 3)
        self.doc(case, 1)
        self.doc(case, 2, "Second document")
        state = self.service.snapshot(self.tokens[1], case)
        self.assertEqual(state["case"]["active_user_id"], 1)
        self.assertEqual(len(state["participants"]), 4)
        self.assertEqual(next(p for p in state["participants"] if p["id"] == 2)["document_title"], "Document 2")
        revision = state["case"]["revision"]
        self.service.invariants(self.tokens[2], case, {
            "invariants": ["Preserve the existing contract", "Measure before changing"], "revision": revision})
        revision += 1
        self.service.brief(self.tokens[2], case, {"user_ids": [1, 2], "revision": revision})
        revision += 1
        with self.assertRaises(ticket_research.ResearchError) as error:
            self.service.promote(self.tokens[2], case, {"revision": revision})
        self.assertEqual(error.exception.status, 403)
        self.assertEqual(self.service.promote(self.tokens[1], case, {"revision": revision})["phase"], "development")
        with self.assertRaises(ticket_research.ResearchError) as error:
            self.service.invariants(self.tokens[4], case, {
                "invariants": ["Unapproved"], "revision": revision + 1})
        self.assertEqual(error.exception.status, 403)
        self.assertEqual(self.service.snapshot(self.tokens[2], case)["case"]["phase"], "development")

    def test_tab_vote_tie_switches_and_survey_deltas(self):
        case = self.create()
        self.enter(case, 1, 2, 3, 4)
        self.doc(case, 1)
        self.doc(case, 2)
        version = self.service.snapshot(self.tokens[1], case)["case"]["revision"]
        self.service.propose(self.tokens[1], case, {"target_user_id": 2, "revision": version})
        self.service.invariants(self.tokens[1], case, {"invariants": ["Keep the interface stable"], "revision": version})
        self.service.vote(self.tokens[2], case, {"choice": "current"})
        self.service.vote(self.tokens[3], case, {"choice": "current"})
        self.assertIsNotNone(self.service.snapshot(self.tokens[1], case)["proposal"])
        self.service.vote(self.tokens[4], case, {"choice": "new"})
        state = self.service.snapshot(self.tokens[1], case)
        self.assertIsNone(state["proposal"])
        self.assertEqual(state["case"]["active_user_id"], 2)
        self.service.survey(self.tokens[1], case, {"axis": "clarity", "value": 2})
        self.service.survey(self.tokens[1], case, {"axis": "clarity", "value": 4})
        state = self.service.snapshot(self.tokens[1], case)
        self.assertEqual(state["surveys"]["clarity"]["mine"], 4)
        changes = [event for event in state["events"] if event["kind"] == "survey:clarity"]
        self.assertEqual(json.loads(changes[0]["before_json"]), 2)
        self.assertEqual(json.loads(changes[0]["after_json"]), 4)

    def test_private_file_requires_explicit_owned_copy(self):
        case = self.create()
        self.enter(case, 1, 2)
        with self.accounts.lock:
            db = self.accounts._conn
            db.execute("CREATE TABLE user_files(id TEXT PRIMARY KEY,owner_id INTEGER)")
            db.execute("INSERT INTO user_files VALUES(?,1)", ("a" * 32,))
            db.commit()
        with self.assertRaises(ticket_research.ResearchError) as error:
            self.service.document(self.tokens[2], case, {"title": "Private",
                "body": "A copy", "source_kind": "file_copy", "source_ref": "a" * 32})
        self.assertEqual(error.exception.status, 404)
        self.service.document(self.tokens[1], case, {"title": "Shared copy",
            "body": "Deliberately copied text", "source_kind": "file_copy", "source_ref": "a" * 32})
        state = self.service.snapshot(self.tokens[2], case)
        self.assertIn("Deliberately copied text", [p["document_body"] for p in state["participants"]])

if __name__ == "__main__":
    unittest.main()
