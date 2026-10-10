import json
from pathlib import Path

from skill_trigger_bench.evaluate import run_eval
from skill_trigger_bench.prompts import GENERIC_NEGATIVES, PromptFileError, PromptSet, generate, load_prompt_file
from skill_trigger_bench.runners import ClaudeRunner, ReplayRunner
from skill_trigger_bench.skills import find_skills

from .helpers import SkillDirCase, fixture

DESC = (
    "Summarize a CSV file. Use when the user asks what is in a CSV, or wants a quick profile of a spreadsheet export."
)


class EvalTests(SkillDirCase):
    def skill(self):
        return find_skills(self.make_skill("csv-summary", "csv-summary", DESC))[0]

    def test_rates_from_replayed_transcripts(self):
        runner = ReplayRunner(
            transcripts=[
                fixture("claude_trigger.jsonl"),
                fixture("claude_no_trigger.jsonl"),
                fixture("claude_no_trigger.jsonl"),
                fixture("claude_trigger.jsonl"),
            ]
        )
        ps = PromptSet(["a", "b"], ["c", "d"])
        r = run_eval(self.skill(), [], ps, runner)
        self.assertEqual(r.trigger_rate, 0.5)
        self.assertEqual(r.false_trigger_rate, 0.5)
        self.assertFalse(r.passed(0.8, 0.2))
        self.assertTrue(r.passed(0.5, 0.5))
        self.assertGreater(r.cost_usd, 0)

    def test_failed_run_counts_as_error(self):
        r = run_eval(self.skill(), [], PromptSet(["a"], []), ReplayRunner(transcripts=[]))
        self.assertEqual(r.errors, 1)
        self.assertIsNone(r.trigger_rate)
        self.assertFalse(r.passed(0, 1))

    def test_skill_not_loaded_flag(self):
        r = run_eval(self.skill(), [], PromptSet(["a"], []), ReplayRunner(transcripts=[fixture("api_error.jsonl")]))
        self.assertTrue(r.skill_not_loaded)

    def test_report_json_serializable(self):
        r = run_eval(
            self.skill(), [], PromptSet(["a"], []), ReplayRunner(transcripts=[fixture("claude_trigger.jsonl")])
        )
        d = json.loads(json.dumps(r.to_dict()))
        self.assertEqual(d["trigger_rate"], 1.0)
        self.assertEqual(d["results"][0]["skill_calls"], ["csv-summary"])

    def test_generate_prompts(self):
        other = find_skills(
            self.make_skill("bread", "bread", "Bake sourdough bread. Use when the user asks about baking.")
        )[0]
        ps = generate(self.skill(), [other], n=4)
        self.assertTrue(ps.generated)
        self.assertTrue(1 <= len(ps.should_trigger) <= 4)
        self.assertEqual(len(ps.should_not_trigger), 4)
        self.assertTrue(any("CSV" in p or "csv" in p for p in ps.should_trigger))
        self.assertTrue(any("sourdough" in p.lower() for p in ps.should_not_trigger))

    def test_generate_is_deterministic(self):
        other = find_skills(
            self.make_skill("bread", "bread", "Bake sourdough bread. Use when the user asks about baking.")
        )[0]
        a = generate(self.skill(), [other], n=10)
        b = generate(self.skill(), [other], n=10)
        self.assertEqual(a.should_trigger, b.should_trigger)
        self.assertEqual(a.should_not_trigger, b.should_not_trigger)

    def test_generate_covers_new_styles(self):
        ps = generate(self.skill(), [], n=10)
        self.assertEqual(len(ps.should_trigger), 10)
        joined = "\n".join(ps.should_trigger)
        self.assertIn(".csv", joined)  # prompts that mention a file by name
        self.assertTrue(any(p.endswith("pls") for p in ps.should_trigger))  # very short prompt
        self.assertTrue(any("goal" in p.lower() for p in ps.should_trigger))  # goal instead of a task
        self.assertTrue(any("past" in p.lower() for p in ps.should_trigger))  # pasted file/output + short ask

    def test_file_prompts_only_for_file_skills(self):
        bread = find_skills(
            self.make_skill("bread", "bread", "Bake sourdough bread. Use when the user asks about baking.")
        )[0]
        ps = generate(bread, [], n=10)
        self.assertNotIn(".csv", "\n".join(ps.should_trigger))

    def test_generic_negatives_have_near_misses(self):
        self.assertGreaterEqual(len(GENERIC_NEGATIVES), 10)
        joined = "\n".join(GENERIC_NEGATIVES)
        self.assertIn("JSON", joined)
        self.assertIn("SQL", joined)

    def test_prompt_file_yaml(self):
        p = self.root / "p.yaml"
        p.write_text("should_trigger:\n  - one\nshould_not_trigger:\n  - two\n  - three\n")
        ps = load_prompt_file(p)
        self.assertEqual((ps.should_trigger, ps.should_not_trigger), (["one"], ["two", "three"]))

    def test_prompt_file_json(self):
        p = self.root / "p.json"
        p.write_text(json.dumps({"should_trigger": ["x"]}))
        self.assertEqual(load_prompt_file(p).should_trigger, ["x"])

    def test_prompt_file_errors(self):
        p = self.root / "p.json"
        p.write_text(json.dumps({"should_trigger": [1]}))
        with self.assertRaises(PromptFileError):
            load_prompt_file(p)
        p.write_text("{}")
        with self.assertRaises(PromptFileError):
            load_prompt_file(p)

    def test_example_prompt_file_parses(self):
        root = Path(__file__).resolve().parent.parent
        ps = load_prompt_file(root / "examples" / "csv-summary.prompts.yaml")
        self.assertEqual(len(ps.should_trigger), 2)
        self.assertIn('{"a": 1', ps.should_not_trigger[1])

    def test_claude_command_is_sandboxed(self):
        cmd = ClaudeRunner(model="haiku").command("hi")
        self.assertEqual(cmd[cmd.index("--setting-sources") + 1], "project")
        self.assertIn("stream-json", cmd)
        self.assertIn("Bash", cmd[cmd.index("--disallowedTools") + 1])
        self.assertEqual(cmd[-2:], ["--model", "haiku"])

    def test_sandbox_copies_skills(self):
        d = self.make_skill("csv-summary", "csv-summary", DESC)
        box = ClaudeRunner.make_sandbox([d], ".claude/skills")
        try:
            self.assertTrue((box / ".claude/skills/csv-summary/SKILL.md").is_file())
        finally:
            import shutil

            shutil.rmtree(box)
