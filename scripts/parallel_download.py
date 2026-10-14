#!/usr/bin/env python3
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_chunk(url, start, end, out_path, chunk_idx):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) != (end - start + 1):
                    raise ValueError(f"Incomplete chunk {chunk_idx}: expected {end - start + 1}, got {len(data)}")
                with open(out_path, 'r+b') as f:
                    f.seek(start)
                    f.write(data)
                return chunk_idx, len(data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1 + attempt)

def parallel_download(url, dest_path, num_workers=16):
    print(f"Resolving URL: {url}")
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=15) as resp:
        content_length = int(resp.headers.get('Content-Length', 0))
        final_url = resp.geturl()

    print(f"Content Length: {content_length} bytes ({content_length / (1024**3):.2f} GiB)")
    if content_length == 0:
        raise ValueError("Could not determine content length")

    # Pre-allocate destination file
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) != content_length:
        print(f"Pre-allocating destination file: {dest_path}")
        with open(dest_path, 'wb') as f:
            f.truncate(content_length)

    chunk_size = 16 * 1024 * 1024  # 16 MB chunks
    chunks = []
    start = 0
    idx = 0
    while start < content_length:
        end = min(start + chunk_size - 1, content_length - 1)
        chunks.append((idx, start, end))
        idx += 1
        start = end + 1

    total_chunks = len(chunks)
    print(f"Downloading {total_chunks} chunks using {num_workers} parallel threads...")
    
    t0 = time.time()
    downloaded_bytes = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(download_chunk, final_url, s, e, dest_path, c_idx): (c_idx, s, e)
            for c_idx, s, e in chunks
        }
        for future in as_completed(future_map):
            c_idx, bytes_written = future.result()
            downloaded_bytes += bytes_written
            completed += 1
            elapsed = time.time() - t0
            speed = (downloaded_bytes / (1024 * 1024)) / max(elapsed, 0.001)
            pct = (downloaded_bytes / content_length) * 100.0
            print(f"\rProgress: {completed}/{total_chunks} chunks ({pct:.1f}%) | {speed:.2f} MB/s | Elapsed: {elapsed:.0f}s", end='', flush=True)

    print(f"\nDownload finished in {time.time() - t0:.1f}s!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: parallel_download.py <URL> <DEST_PATH> [NUM_WORKERS]")
        sys.exit(1)
    url = sys.argv[1]
    dest = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    parallel_download(url, dest, workers)
