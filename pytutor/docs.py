import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import click

from pytutor.config import DATA_DIR

PYDOC_ARCHIVE_URL = "https://docs.python.org/3/archives/python-{version}-docs-text.zip"
LATEST_DOCS_URL = "https://docs.python.org/3/"
LATEST_VERSION_RE = re.compile(r"3\.\d+\.\d+")


def latest_docs_version() -> str:
    """Resolve the latest stable Python docs minor version (e.g. ``3.14``)."""
    html = urllib.request.urlopen(LATEST_DOCS_URL, timeout=30).read().decode()
    match = LATEST_VERSION_RE.search(html)
    if not match:
        raise RuntimeError(
            f"Could not determine the latest Python docs version from {LATEST_DOCS_URL}"
        )
    return ".".join(match.group(0).split(".")[:2])


def update_python_docs(
    version: str,
    data_dir: Path = DATA_DIR,
    progress=None,
) -> None:
    """Download python-{version}-docs-text.zip and replace data_dir with its content.

    ``progress`` is an optional callback ``(downloaded_bytes, total_bytes)``
    called as chunks are written. When omitted, a ``click.progressbar`` is
    rendered for CLI use.
    """
    url = PYDOC_ARCHIVE_URL.format(version=version)
    click.echo(f"Downloading Python docs {version} from {url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / f"python-{version}-docs-text.zip"

        try:
            with urllib.request.urlopen(url) as resp:
                total = int(resp.headers.get("Content-Length", "0") or 0)
                chunk_size = 1024 * 256

                with open(zip_path, "wb") as f:
                    if progress is not None:
                        done = 0
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            done += len(chunk)
                            progress(done, total)
                    elif total > 0:
                        with click.progressbar(
                            length=total, label=f"Downloading {zip_path.name}"
                        ) as bar:
                            while True:
                                chunk = resp.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                                bar.update(len(chunk))
                    else:
                        f.write(resp.read())
        except Exception as e:
            click.echo(click.style(f"Error: failed to download docs: {e}", fg="red"))
            raise click.Abort()

        click.echo("Extracting documentation...")
        extract_dir = tmpdir_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            click.echo(click.style(f"Error: failed to extract docs: {e}", fg="red"))
            raise click.Abort()

        candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
        target_src = None
        for p in candidates:
            if p.name.startswith("python-") and p.name.endswith("-docs-text"):
                target_src = p
                break
        if target_src is None:
            if len(candidates) == 1:
                target_src = candidates[0]
            else:
                click.echo(
                    click.style(
                        "Error: could not locate extracted docs directory.", fg="red"
                    )
                )
                raise click.Abort()

        if data_dir.exists():
            click.echo(f"Removing old docs at {data_dir}")
            shutil.rmtree(data_dir)
        click.echo(f"Installing new docs to {data_dir}")
        shutil.move(str(target_src), str(data_dir))

        click.echo("Python documentation updated successfully.")
