"""Download the official CIFAR-10 archive in parallel, then verify its MD5."""
import argparse
import concurrent.futures
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request

URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
SIZE = 170498071
MD5 = 'c58f30108f718f92721af3b95e74349a'


def download(directory, workers=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'cifar-10-python.tar.gz'
    if archive.exists() and hashlib.md5(archive.read_bytes()).hexdigest() == MD5:
        print('Official CIFAR-10 archive already verified.', flush=True)
        return archive
    parts = directory / '.download-parts'
    parts.mkdir(exist_ok=True)
    chunk_size = 8 * 1024 * 1024
    ranges = [(i, start, min(start + chunk_size, SIZE) - 1)
              for i, start in enumerate(range(0, SIZE, chunk_size))]

    def fetch(item):
        index, start, end = item
        path = parts / ('%03d.part' % index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={'Range': 'bytes=%d-%d' % (start, end)})
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected_range = 'bytes %d-%d/%d' % (start, end, SIZE)
                    if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                        raise RuntimeError('Server did not honor the requested byte range')
                    with path.open('wb') as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                if path.stat().st_size != end - start + 1:
                    raise RuntimeError('Incomplete data segment')
                print('Downloaded segment %d/%d' % (index + 1, len(ranges)), flush=True)
                return path
            except Exception as error:
                print('Segment %d attempt %d failed: %s' % (index + 1, attempt + 1, error), flush=True)
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(fetch, ranges))
    temporary = archive.with_suffix('.verified.tmp')
    digest = hashlib.md5()
    with temporary.open('wb') as output:
        for path in paths:
            with path.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
    if digest.hexdigest() != MD5:
        raise RuntimeError('Archive MD5 mismatch; remove .download-parts and retry')
    temporary.replace(archive)
    for path in paths:
        path.unlink()
    parts.rmdir()
    print('CIFAR-10 archive verified: ' + MD5, flush=True)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='../data')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    download(args.data_dir, args.workers)
