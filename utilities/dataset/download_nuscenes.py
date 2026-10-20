# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Download and extract the full nuScenes v1.0 dataset (trainval and test).

Adapted from https://github.com/li-xl/nuscenes-download (f2fc4fd). The nuScenes
account comes from an env file. A download that stops resumes where it was,
each archive must match its MD5, and an archive that is extracted is recorded
in <dest>/.download/, so a second run skips it.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

COGNITO_URL = "https://cognito-idp.us-east-1.amazonaws.com/"
COGNITO_CLIENT_ID = "7fq5jvs5ffs1c50hd3toobb3b9"
ARCHIVE_API = ("https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/"
               "{name}?region={region}&project=nuScenes")
TOKEN_LIFETIME = 50 * 60  # the ID token expires after one hour
CHUNK = 8 << 20
RETRIES = 5
PROGRESS_EVERY = 60.0

# Metadata first, so a partial run already has the tables.
ARCHIVES = {
    "v1.0-trainval_meta.tgz": "537d3954ec34e5bcb89a35d4f6fb0d4a",
    "v1.0-test_meta.tgz": "b0263f5c41b780a5a10ede2da99539eb",
    "v1.0-trainval01_blobs.tgz": "cbf32d2ea6996fc599b32f724e7ce8f2",
    "v1.0-trainval02_blobs.tgz": "aeecea4878ec3831d316b382bb2f72da",
    "v1.0-trainval03_blobs.tgz": "595c29528351060f94c935e3aaf7b995",
    "v1.0-trainval04_blobs.tgz": "b55eae9b4aa786b478858a3fc92fb72d",
    "v1.0-trainval05_blobs.tgz": "1c815ed607a11be7446dcd4ba0e71ed0",
    "v1.0-trainval06_blobs.tgz": "7273eeea36e712be290472859063a678",
    "v1.0-trainval07_blobs.tgz": "46674d2b2b852b7a857d2c9a87fc755f",
    "v1.0-trainval08_blobs.tgz": "37524bd4edee2ab99678909334313adf",
    "v1.0-trainval09_blobs.tgz": "a7fcd6d9c0934e4052005aa0b84615c0",
    "v1.0-trainval10_blobs.tgz": "31e795f2c13f62533c727119b822d739",
    "v1.0-test_blobs.tgz": "e065445b6019ecc15c70ad9d99c47b33",
}

TRANSIENT = (urllib.error.URLError, http.client.HTTPException, OSError)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def load_credentials(env_file: Path) -> tuple[str, str]:
    """`NUSCENES_EMAIL` and `NUSCENES_PASSWORD` from a KEY=VALUE file that only its owner can read."""
    if not env_file.is_file():
        sys.exit(f"{env_file} does not exist; write NUSCENES_EMAIL=... and NUSCENES_PASSWORD=... to it.")
    if env_file.stat().st_mode & 0o077:
        sys.exit(f"{env_file} is readable by other users; run: chmod 600 {env_file}")
    values = {}
    for line in env_file.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if not sep or key.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    missing = [k for k in ("NUSCENES_EMAIL", "NUSCENES_PASSWORD") if not values.get(k)]
    if missing:
        sys.exit(f"{env_file} has no {' or '.join(missing)}.")
    return values["NUSCENES_EMAIL"], values["NUSCENES_PASSWORD"]


def login(email: str, password: str) -> str:
    body = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": COGNITO_CLIENT_ID,
        "AuthParameters": {"USERNAME": email, "PASSWORD": password},
        "ClientMetadata": {},
    }).encode()
    request = urllib.request.Request(COGNITO_URL, data=body, headers={
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            reply = json.load(response)
    except urllib.error.HTTPError as e:
        sys.exit(f"nuScenes login failed ({e.code}): {e.read().decode(errors='replace')}")
    if "AuthenticationResult" not in reply:
        sys.exit(f"nuScenes login gave no token: {reply}")
    return reply["AuthenticationResult"]["IdToken"]


class Account:
    """Signed download URLs, with a login token that is renewed before it expires."""

    def __init__(self, email: str, password: str, region: str):
        self.email, self.password, self.region = email, password, region
        self._token, self._token_time = "", 0.0

    def token(self) -> str:
        if not self._token or time.monotonic() - self._token_time > TOKEN_LIFETIME:
            self._token, self._token_time = login(self.email, self.password), time.monotonic()
        return self._token

    def archive_url(self, name: str) -> str:
        request = urllib.request.Request(ARCHIVE_API.format(name=name, region=self.region),
                                         headers={"Authorization": f"Bearer {self.token()}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)["url"]


def archive_size(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as r:
        return int(r.headers["Content-Range"].rpartition("/")[2])


def fetch(url: str, part: Path) -> None:
    """Write `url` to `part`, continuing from the bytes `part` already holds."""
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # nothing left to fetch
            return
        raise
    with response:
        if have and response.status != 206:
            have = 0
        total = have + int(response.headers.get("Content-Length", 0))
        if have:
            log(f"{part.name}: resuming at {gib(have)} of {gib(total)}")
        start, last, done = time.monotonic(), time.monotonic(), have
        with part.open("ab" if have else "wb") as f:
            while chunk := response.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > PROGRESS_EVERY:
                    last = time.monotonic()
                    rate = (done - have) / (last - start) / 2**20
                    log(f"{part.name}: {gib(done)} of {gib(total)} ({100 * done / total:.0f}%), {rate:.0f} MiB/s")
    if part.stat().st_size != total:
        raise OSError(f"got {part.stat().st_size} of {total} bytes")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(get_url: Callable[[], str], part: Path, md5: str) -> None:
    """Fetch with retries, asking for a fresh signed URL each time, then check the MD5."""
    for attempt in range(1, RETRIES + 1):
        try:
            fetch(get_url(), part)
            break
        except TRANSIENT as e:
            if attempt == RETRIES:
                raise
            wait = 30 * attempt
            log(f"{part.name}: {e}; retry {attempt} of {RETRIES - 1} in {wait} s")
            time.sleep(wait)
    log(f"{part.name}: checking MD5")
    if (got := md5_of(part)) != md5:
        part.unlink()
        raise RuntimeError(f"{part.name}: MD5 {got} is not {md5}; the partial file is removed, run again")


def extract(archive: Path, dest: Path) -> None:
    subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, required=True,
                        help="the nuScenes dataroot; samples/, sweeps/ and the tables land here")
    parser.add_argument("--env-file", type=Path, default=Path("~/.config/nuscenes.env"),
                        help="KEY=VALUE file with NUSCENES_EMAIL and NUSCENES_PASSWORD (default: %(default)s)")
    parser.add_argument("--region", choices=["us", "asia"], default="us")
    parser.add_argument("--archives", nargs="+", choices=list(ARCHIVES), metavar="NAME",
                        help="only these archives (default: all, in the order of ARCHIVES)")
    parser.add_argument("--keep-archives", action="store_true", help="keep each .tgz after it is extracted")
    parser.add_argument("--check", action="store_true",
                        help="log in, print the size of each archive, and stop without downloading")
    args = parser.parse_args()

    dest = args.dest.expanduser().resolve()
    state = dest / ".download"
    names = [n for n in ARCHIVES if n in (args.archives or ARCHIVES)]
    account = Account(*load_credentials(args.env_file.expanduser()), region=args.region)

    if args.check:
        total = 0
        for name in names:
            size = archive_size(account.archive_url(name))
            total += size
            done = "  (done)" if (state / f"{name}.done").exists() else ""
            print(f"{name:28} {gib(size):>10}{done}")
        print(f"{'total':28} {gib(total):>10}")
        return

    state.mkdir(parents=True, exist_ok=True)
    for name in names:
        done, archive, part = state / f"{name}.done", state / name, state / f"{name}.part"
        if done.exists():
            log(f"{name}: already extracted")
            continue
        if not archive.exists():
            log(f"{name}: downloading")
            download(lambda: account.archive_url(name), part, ARCHIVES[name])
            part.rename(archive)
        log(f"{name}: extracting to {dest}")
        extract(archive, dest)
        done.touch()
        if not args.keep_archives:
            archive.unlink()
        log(f"{name}: done")
    log("all archives extracted")


if __name__ == "__main__":
    main()
