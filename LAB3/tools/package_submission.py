"""Package source, README and experiment evidence without datasets/checkpoints."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'data', 'tmp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'output' / '实验三_GitHub上传.zip')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file() or EXCLUDED.intersection(relative.parts):
                continue
            if path.suffix.lower() in {'.pt', '.pth', '.pyc', '.tmp', '.zip'}:
                continue
            archive.write(path, relative.as_posix())
            count += 1
    print('Packaged %d files: %s (%.2f MB)' % (count, args.output, args.output.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
