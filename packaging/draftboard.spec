# PyInstaller spec for the standalone `draftboard` binary.
#
#   .venv/bin/pyinstaller packaging/draftboard.spec --noconfirm
#
# Two things here are not the defaults and both matter:
#
#  * The modules under src/ import each other flatly (`from common import ...`).
#    PyInstaller analyses imports statically and never sees the sys.path insert
#    that makes that work, so every module has to be named as a hidden import
#    and src/ has to be on the analysis path.
#  * The dashboard shells out to run_pipeline.py / refresh_news.py for `F`.
#    A frozen app has no .py files next to it, so those scripts are bundled as
#    data and the code resolves them through sys._MEIPASS.
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "fantasydraft"

# Every top-level module name the flat imports can reach.
FLAT_MODULES = sorted(p.stem for p in SRC.glob("*.py") if p.stem != "__init__")

a = Analysis(
    [str(SRC / "draft_tui.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[
        (str(ROOT / "config" / "weights.yaml"), "config"),
        (str(ROOT / "config" / "news.yaml"), "config"),
        (str(ROOT / "config" / "team_context.yaml"), "config"),
        # Bundled so the in-app fetch can still run them.
        *[(str(SRC / f"{m}.py"), "scripts") for m in
          ("run_pipeline", "refresh_news")],
    ],
    hiddenimports=[
        *FLAT_MODULES,
        "textual.widgets",
        "textual.app",
        "pyarrow.parquet",
        "pandas._libs.tslibs.base",
        "scipy.special.cython_special",
        "sklearn.utils._typedefs",
        "sklearn.neighbors._partition_nodes",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim the heaviest things a terminal app can never use. Matplotlib and Qt
    # get pulled in transitively by scikit-learn/scipy and add ~100MB.
    excludes=["matplotlib", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "IPython", "jupyter", "notebook", "pytest", "sphinx"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="draftboard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # it IS a terminal app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
