import json
import subprocess
import sys


def test_doctor_json_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "model.scripts.doctor", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["python"].startswith("3.12")
    assert isinstance(payload["compute"]["cuda_available"], bool)
    assert payload["assets"]["model_cached"] is False

