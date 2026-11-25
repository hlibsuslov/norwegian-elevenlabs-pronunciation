"""Fetch the named upstream dictionary releases into the local data directory."""
from .common import data_dir
from .setup_lexicon import download

RESOURCES = {
    "ukrainian_mfa.dict": (
        "https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/ukrainian/mfa/ukrainian_mfa.dict",
        "876f28a49f56036005e804daf53a70b3d2573bce3f690c615976dde7982917b6",
    ),
    "english_uk_mfa.dict": (
        "https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_uk_mfa-v3.1.0/english_uk_mfa.dict",
        "a4104e3ef5f98eb86baa11b032da77059dc6180227ddbca31a0c6fa3d5c1802c",
    ),
}


if __name__ == "__main__":
    from filelock import FileLock
    directory = data_dir()/"multilingual"
    directory.mkdir(parents=True,exist_ok=True)
    with FileLock(directory/"setup.lock",timeout=60):
        for name,(url,checksum) in RESOURCES.items():
            print(download(url,directory/name,checksum))
