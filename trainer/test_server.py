import importlib.util, json, tempfile, unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "trainer_server", Path(__file__).with_name("server.py")
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class DatasetTests(unittest.TestCase):
    def test_disjoint_unique_splits(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"Example {i}"},
                    {"role": "assistant", "content": "NOTICE: a synthetic example."},
                ]
            }
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as d:
            server.split_custom("\n".join(json.dumps(r) for r in rows), Path(d))
            parts = [
                {
                    json.loads(s)["messages"][0]["content"]
                    for s in (Path(d) / (name + ".jsonl")).read_text().splitlines()
                }
                for name in ["train", "valid", "test"]
            ]
            self.assertEqual(sum(map(len, parts)), 20)
            self.assertFalse(
                parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]
            )

    def test_bad_data_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                server.split_custom("{}", Path(d))
            row = json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "same"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }
            )
            with self.assertRaises(ValueError):
                server.split_custom("\n".join([row] * 12), Path(d))


if __name__ == "__main__":
    unittest.main()
