#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_chunk(url, start, end, out_path, chunk_idx):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'Mozilla/5.0'})
    max_retries = 8
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
                expected_len = end - start + 1
                if len(data) != expected_len:
                    raise ValueError(f"Incomplete chunk {chunk_idx}: expected {expected_len}, got {len(data)}")
                with open(out_path, 'r+b') as f:
                    f.seek(start)
                    f.write(data)
                return chunk_idx, len(data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.0 + attempt * 1.5)

def parallel_download(url, dest_path, num_workers=24):
    print(f"Resolving URL: {url}")
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content_length = int(resp.headers.get('Content-Length', 0))
        final_url = resp.geturl()

    print(f"Content Length: {content_length} bytes ({content_length / (1024**3):.2f} GiB)")
    if content_length == 0:
        raise ValueError("Could not determine content length")

    state_path = f"{dest_path}.chunks.json"
    done_chunks = set()
    if os.path.exists(state_path) and os.path.exists(dest_path) and os.path.getsize(dest_path) == content_length:
        try:
            with open(state_path, 'r') as sf:
                done_chunks = set(json.load(sf))
            print(f"Resuming download: {len(done_chunks)} chunks already completed.")
        except Exception:
            done_chunks = set()
    else:
        print(f"Pre-allocating destination file: {dest_path}")
        with open(dest_path, 'wb') as f:
            f.truncate(content_length)

    chunk_size = 16 * 1024 * 1024  # 16 MB chunks
    chunks_to_download = []
    start = 0
    idx = 0
    while start < content_length:
        end = min(start + chunk_size - 1, content_length - 1)
        if idx not in done_chunks:
            chunks_to_download.append((idx, start, end))
        idx += 1
        start = end + 1

    total_chunks = idx
    already_done_bytes = sum(
        min((c_idx + 1) * chunk_size, content_length) - c_idx * chunk_size
        for c_idx in done_chunks
    )

    if not chunks_to_download:
        print(f"All {total_chunks} chunks already downloaded and verified.")
        if os.path.exists(state_path):
            os.remove(state_path)
        return

    print(f"Downloading {len(chunks_to_download)} remaining chunks (out of {total_chunks}) using {num_workers} parallel threads...")
    
    t0 = time.time()
    downloaded_bytes = already_done_bytes
    completed = len(done_chunks)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(download_chunk, final_url, s, e, dest_path, c_idx): c_idx
            for c_idx, s, e in chunks_to_download
        }
        for future in as_completed(future_map):
            c_idx, bytes_written = future.result()
            downloaded_bytes += bytes_written
            completed += 1
            done_chunks.add(c_idx)
            
            # Periodically or every chunk persist progress
            try:
                with open(state_path, 'w') as sf:
                    json.dump(list(done_chunks), sf)
            except Exception:
                pass

            elapsed = time.time() - t0
            speed = ((downloaded_bytes - already_done_bytes) / (1024 * 1024)) / max(elapsed, 0.001)
            pct = (downloaded_bytes / content_length) * 100.0
            print(f"\rProgress: {completed}/{total_chunks} chunks ({pct:.1f}%) | {speed:.2f} MB/s | Elapsed: {elapsed:.0f}s", end='', flush=True)

    if os.path.exists(state_path):
        os.remove(state_path)
    print(f"\nDownload finished in {time.time() - t0:.1f}s!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: parallel_download.py <URL> <DEST_PATH> [NUM_WORKERS]")
        sys.exit(1)
    url = sys.argv[1]
    dest = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    parallel_download(url, dest, workers)
