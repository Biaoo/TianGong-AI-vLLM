"""Download a pinned public HF snapshot, verify every file, and resume partial files."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--include-prefix", action="append", default=[])
    parser.add_argument("--transport", choices=["curl", "hub", "modelscope"], default="curl")
    parser.add_argument("--mirror-model")
    parser.add_argument("--mirror-revision")
    parser.add_argument("--ipv4", action="store_true", help="Use IPv4 for curl downloads")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.model):
        parser.error("Expected a public owner/model identifier")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("A full immutable revision is required")
    if args.transport == "modelscope" and not (
        re.fullmatch(r"[\w.-]+/[\w.-]+", args.mirror_model or "")
        and re.fullmatch(r"[0-9a-f]{40}", args.mirror_revision or "")
    ):
        parser.error("ModelScope requires a public mirror model and immutable mirror revision")
    root = args.directory.resolve()
    if not args.verify_only:
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".snapshot.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another snapshot operation is already running") from exc
    manifest_path = root / "snapshot-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["model_id"], manifest["revision"]) != (args.model, args.revision):
            raise ValueError("Existing manifest belongs to a different snapshot")
        entries = manifest["files"]
        recorded = manifest.get("selection")
        requested = {"include": args.include_prefix, "exclude": args.exclude_prefix}
        if recorded is not None and recorded != requested:
            raise ValueError("Selection differs from the existing immutable manifest")
        if any(
            (args.include_prefix and not any(e["path"].startswith(p) for p in args.include_prefix))
            or any(e["path"].startswith(p) for p in args.exclude_prefix)
            for e in entries
        ):
            raise ValueError("Existing manifest contains files outside requested selection")
    else:
        if args.verify_only:
            raise ValueError("Missing snapshot manifest; verification never downloads")
        url = (
            f"https://huggingface.co/api/models/{args.model}/tree/{args.revision}"
            "?recursive=true&expand=false&limit=1000"
        )
        entries = []
        while url:
            with urlopen(url, timeout=60) as response:
                data = json.load(response)
                links = response.headers.get("Link", "")
            for item in data:
                if item["type"] != "file" or any(
                    item["path"].startswith(p) for p in args.exclude_prefix
                ):
                    continue
                if args.include_prefix and not any(
                    item["path"].startswith(p) for p in args.include_prefix
                ):
                    continue
                entries.append(
                    {
                        "path": item["path"],
                        "size": item["size"],
                        "sha256": item.get("lfs", {}).get("oid"),
                        "git_oid": item["oid"],
                    }
                )
            match = re.search(r'<(https://huggingface.co/[^>]+)>; rel="next"', links)
            url = match[1] if match else None
        if not entries:
            raise ValueError("No files matched the requested snapshot selection")
        manifest = {
            "model_id": args.model,
            "revision": args.revision,
            "selection": {"include": args.include_prefix, "exclude": args.exclude_prefix},
            "files": entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def valid(path, entry):
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return False
        if entry["sha256"]:
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        digest.update(path.read_bytes())
        return digest.hexdigest() == entry["git_oid"]

    def fetch(entry):
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe snapshot path")
        if not valid(path, entry):
            if args.verify_only:
                raise ValueError(f"Missing or corrupt: {entry['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if args.transport == "hub":
                # Optional dependency exists only in the pinned download container.
                # The host implementation and integrity checks remain stdlib-only.
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    repo_id=args.model,
                    filename=entry["path"],
                    revision=args.revision,
                    local_dir=root,
                    token=False,
                    force_download=path.exists(),
                )
                if not valid(path, entry):
                    raise ValueError(f"Checksum mismatch: {entry['path']}")
                if not entry["sha256"]:
                    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                print(f"Verified {entry['path']}", flush=True)
                return
            partial = path.with_name(path.name + ".part")
            url = (
                f"https://huggingface.co/{args.model}/resolve/{args.revision}/"
                f"{quote(entry['path'])}"
            )
            if args.transport == "modelscope" and entry["path"] != ".gitattributes":
                # Mirror bytes must still pass the original HF hash, including
                # Git-blob SHA1 for metadata. Repository-specific .gitattributes
                # is deliberately fetched from the original repository.
                url = (
                    f"https://modelscope.cn/models/{args.mirror_model}/resolve/"
                    f"{args.mirror_revision}/{quote(entry['path'])}"
                )
            # curl handles a changing signed CDN URL without logging its query string.
            subprocess.run(
                [
                    "curl",
                    *(["--ipv4"] if args.ipv4 else []),
                    "--http1.1",
                    "-fLsS",
                    "--retry",
                    "8",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "30",
                    "--speed-time",
                    "120",
                    "--speed-limit",
                    "1024",
                    "-C",
                    "-",
                    "-o",
                    str(partial),
                    url,
                ],
                check=True,
            )
            if not valid(partial, entry):
                raise ValueError(f"Checksum mismatch: {entry['path']}")
            partial.replace(path)
        if not entry["sha256"]:
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Verified {entry['path']}", flush=True)

    print(
        f"Snapshot: {len(entries)} files, {sum(x['size'] for x in entries)/2**30:.2f} GiB",
        flush=True,
    )
    if args.manifest_only:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, entries))
    if not args.verify_only:
        if args.transport == "modelscope":
            manifest["verified_mirror"] = {
                "provider": "modelscope",
                "model_id": args.mirror_model,
                "revision": args.mirror_revision,
            }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Snapshot fully verified", flush=True)


if __name__ == "__main__":
    main()
