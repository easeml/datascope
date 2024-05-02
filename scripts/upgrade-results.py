#!/usr/bin/env python3

import argparse
import re
import os

from glob import glob
from pathlib import Path
from tqdm import tqdm


REPLACEMENTS = {
    "iteration": "seed",
    "trainsize": "dataset_trainsize",
    "dataset-trainsize": "dataset_trainsize",
    "valsize": "dataset_valsize",
    "dataset-valsize": "dataset_valsize",
    "testsize": "dataset_testsize",
    "dataset-testsize": "dataset_testsize",
    "numfeatures": "dataset_numfeatures",
    "dataset-numfeatures": "dataset_numfeatures",
    "checkpoints": "numcheckpoints",
    "UCI": "UCI-Adult",
    "FolkUCI": "Folk-UCI-Adult",
    "FashionMNIST": "Fashion-MNIST",
    "TwentyNewsGroups": "Twenty-Newsgroups",
    "DataPerfVision": "DataPerf-Vision",
    "CifarN": "CIFAR-N",
}

# Matchers will replace using regex. Do not replace if surrounded by hyphens or alphanumeric characters.
MATCHERS = {
    re.compile(r"(?<![a-zA-Z0-9-])" + old + r"(?![a-zA-Z0-9-])"): (old, new) for old, new in REPLACEMENTS.items()
}

FILETYPES = ["*.csv", "*.yaml"]


def main(rootdir: str, dry_run: bool = False) -> None:
    # Assemble all names of files that could contain result content.
    filenames = []
    for filetype in FILETYPES:
        filenames.extend(glob(os.path.join(rootdir, "**", filetype), recursive=True))

    # # Perform replacements of the text.
    for filename in tqdm(filenames, desc="Replacing file content"):
        with open(filename, "r") as f:
            content = f.read()

        for matcher, (old, new) in MATCHERS.items():
            # Replace using regex. Do not replace if surrounded by hyphens or alphanumeric characters.
            if dry_run:
                for m in matcher.finditer(content):
                    a, b = max(m.start(), 0), min(m.end(), len(content))
                    substring = content[a:b]
                    tqdm.write(
                        "File %s: Replacing '%s' with '%s'."
                        % (filename, substring.replace("\n", "\\n"), matcher.sub(new, substring).replace("\n", "\\n"))
                    )
            else:
                content = matcher.sub(new, content)

        with open(filename, "w") as f:
            f.write(content)

    # Go through all filenames and find the ones that contain keywords in their name.
    filepaths = [Path(x) for x in glob(os.path.join(rootdir, "**"), recursive=True)]
    filepaths = [x for x in filepaths if any(m.search(x.name) for m in MATCHERS.keys())]
    filepaths = sorted(filepaths, key=lambda x: len(x.parts), reverse=True)
    for old_path in tqdm(filepaths, desc="Renaming files"):
        new_path = old_path
        for matcher, (old, new) in MATCHERS.items():
            new_path = new_path.with_name(matcher.sub(new, new_path.name))
        if dry_run:
            tqdm.write("File %s: Renaming to %s." % (old_path, new_path))
        else:
            old_path.rename(new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgrade results files.")
    parser.add_argument("rootdir", type=str, help="The root directory to search for files.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files.")
    args = parser.parse_args()

    main(args.rootdir, args.dry_run)
