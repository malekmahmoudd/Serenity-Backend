"""
Proves the production dependency set (requirements-onnx.txt) actually boots
the real app: imports every backend module main.py imports, confirms the
classifier loads via ONNX, confirms TensorFlow is not merely un-imported but
GENUINELY NOT INSTALLED, and runs real requests through the FastAPI app via
TestClient (no live server, no Docker needed for this check).

Run with the SLIM venv specifically:
    venv_slim_test\\Scripts\\python.exe verify_slim_boot.py
"""
import os
import sys

os.environ["SERENITY_ENC_KEY"] = "sVh0kQ2Zp7XcJ9nR4tB6yU8wE1oA3iL5mF7gD0hK2vM="
os.environ["SERENITY_JWT_SECRET"] = "slim-boot-test-secret"
os.environ["DATABASE_URL"] = "sqlite:///./slim_test.db"

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


print("=" * 62)
print("1. tensorflow is genuinely not installed")
print("=" * 62)
try:
    import tensorflow  # noqa: F401
    check("tensorflow is NOT installed", False, "(it imported successfully!)")
except ModuleNotFoundError:
    check("tensorflow is NOT installed", True)

print()
print("=" * 62)
print("2. full app import chain, exactly what main.py imports")
print("=" * 62)
try:
    import fastapi, sqlalchemy, cryptography, onnxruntime, sklearn  # noqa
    check("all direct dependencies import", True)
except Exception as e:
    check("all direct dependencies import", False, str(e))

try:
    from fastapi.testclient import TestClient
    import config  # noqa: F401
    import main
    check("main.py imports and FastAPI app builds", True)
except Exception as e:
    import traceback
    traceback.print_exc()
    check("main.py imports and FastAPI app builds", False, str(e))
    print(f"\n{len(FAILURES)} FAILED -- cannot continue")
    sys.exit(1)

print()
print("=" * 62)
print("3. classifier loaded via ONNX in this environment")
print("=" * 62)
import classifier
status = classifier.status()
check("classifier ready", status["ready"], str(status.get("error")))
check("backend is onnx (not keras -- keras would need TF, which isn't here)",
      status["backend"] == "onnx", f"(got {status['backend']})")

print()
print("=" * 62)
print("4. real requests through the actual FastAPI app")
print("=" * 62)
client = TestClient(main.app)

r = client.get("/health")
check("/health returns 200", r.status_code == 200, f"(got {r.status_code})")
body = r.json()
check("/health reports classifier ready", body["classifier"]["ready"] is True)
check("/health reports onnx backend",
      body["classifier"]["backend"] == "onnx", f"(got {body['classifier']['backend']})")

r2 = client.post("/chat", json={"message": "i can't stop checking the locks"})
check("/chat returns 200", r2.status_code == 200, f"(got {r2.status_code})")
body2 = r2.json()
check("/chat produces a real action", body2.get("action") in
      ("ASK", "RESOLVE", "ABSTAIN", "OOD", "SAFETY"), f"(got {body2.get('action')})")
check("/chat produces a real reply", bool(body2.get("gpt_response")))
print(f"    action={body2.get('action')}  reply={body2.get('gpt_response','')[:70]!r}")

r3 = client.post("/chat", json={"message": "i don't want to be here anymore"})
body3 = r3.json()
check("crisis path works in the slim environment",
      body3.get("action") == "SAFETY", f"(got {body3.get('action')})")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("ALL PASS -- the production (no-TensorFlow) dependency set genuinely "
      "boots and serves the real app")
