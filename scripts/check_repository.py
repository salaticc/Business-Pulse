"""Check the Git index for local artifacts and common credential patterns.

This small guard is not a complete secret-detection system. Values are never
printed. Run after staging; CI checks the committed checkout.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "telegram_token": re.compile(rb"\b\d{7,12}:[A-Za-z0-9_-]{30,}\b"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}


def main() -> None:
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "-z"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    paths = [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
    if not paths:
        raise SystemExit("No tracked/staged files; stage intended changes first.")
    errors, metadata = [], {}
    for name in paths:
        rel = Path(name)
        forbidden = any(p in {".venv", "venv", "__pycache__", ".pytest_cache", "local_state", ".ci-data"} for p in rel.parts)
        raw = rel.parts[:2] == ("data", "raw") and rel.name != "README.md"
        env = rel.name.startswith(".env") and rel.name != ".env.example"
        local = rel.suffix.lower() in {".zip", ".pyc", ".pkl", ".pickle", ".joblib", ".db", ".pem", ".key"} or rel.suffix.lower().startswith(".sqlite")
        if forbidden or raw or env or local or rel.parts[0] == "outputs":
            errors.append(f"{name}: local/private artifact must not be tracked")
            continue
        blob = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f":{name}"],
                              cwd=ROOT, check=True, capture_output=True).stdout
        if name in {"VERSION", "CHANGELOG.md"}:
            metadata[name] = blob.decode("utf-8-sig")
        if len(blob) > 10 * 1024 * 1024:
            errors.append(f"{name}: exceeds the 10 MiB repository limit")
        for kind, pattern in PATTERNS.items():
            if pattern.search(blob):
                errors.append(f"{name}: {kind} pattern detected (value withheld)")
    version = metadata.get("VERSION", "").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        errors.append("VERSION must contain X.Y.Z")
    if f"## {version} " not in metadata.get("CHANGELOG.md", ""):
        errors.append("VERSION missing from CHANGELOG")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Repository check passed: {len(paths)} tracked/staged files, version {version}.")


if __name__ == "__main__":
    main()
