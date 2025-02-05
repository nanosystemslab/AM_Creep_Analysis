"""Nox sessions."""
import os 
import shutil
from pathlib import Path
import nox
from nox.sessions import Session

session = nox.session

package = "AM_Creep_Analysis"
python_versions = ["3.12", "3.11", "3.10", "3.9"]
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = ("docs-build", "docs", "mypy")

@session(python=python_versions)
def mypy(session: Session) -> None:
    """Run mypy static type checks."""
    session.install("mypy")
    session.run("mypy", "src/")

@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.install(".")
    session.install("sphinx", "sphinx-argparse", "furo", "myst-parser")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)

@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".")
    session.install(
        "sphinx", "sphinx-autobuild", "sphinx-argparse", "furo", "myst-parser"
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
