import hashlib
import pickle
import os
import requests
import tarfile

from tqdm import tqdm
from typing import Optional
from zipfile import ZipFile


def download(url: str, filename: str, chunk_size=1024, checksum: Optional[str] = None, force: bool = False):
    if not force and os.path.exists(filename):
        return

    if "drive.google.com" in url:
        from gdown import download as gdownload

        gdownload(url, filename, quiet=False)
    else:
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(filename, "wb") as file:
            desc = "Downloading %s" % os.path.basename(filename)
            with tqdm(desc=desc, total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)

    if checksum is not None:
        sha256 = hashlib.sha256()
        with open(filename, "rb") as file:
            while True:
                data = file.read(1 << 16)
                if not data:
                    break
                sha256.update(data)

        if sha256.hexdigest() != checksum:
            raise ValueError("Checksum mismatch.")


def unzip(source: str, path: str):
    with ZipFile(source, "r") as zip_ref:
        desc = "Extracting %s" % os.path.basename(source)
        for file in tqdm(desc=desc, iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=path)


def untar(source: str, path: str, mode: str = "r"):
    with tarfile.open(source, mode) as tar:
        desc = "Extracting %s" % os.path.basename(source)
        for file in tqdm(desc=desc, iterable=tar.getmembers(), total=len(tar.getmembers())):
            tar.extract(member=file, path=path)


def unpickle(file, encoding="latin1"):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding=encoding)
    return dict
