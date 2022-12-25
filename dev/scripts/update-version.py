#!/usr/bin/env python3

import argparse
import re
import sys

from typing import Tuple, List


def update_version(versionfile: str, major: bool, minor: bool, patch: bool) -> Tuple[str, str]:
    pre_version, post_version = "", ""
    with open(versionfile, "r") as f:
        line = f.readline()
        matches = re.findall(r'\s*__version__ = "([0-9]+)\.([0-9]+)\.([0-9]+)"\s*', line)
        if len(matches) != 1:
            raise ValueError(
                "The provided version file is not properly formatted."
                'Expected format is `__version__ = "A.B.C"`'
                "where A, B and C are major, minor and patch version numbers respectively."
            )
        version: List[int] = [int(x) for x in matches[0]]
        pre_version = "%d.%d.%d" % tuple(version)

        if major:
            version[0] += 1
        if minor:
            version[1] += 1
        if patch:
            version[2] += 1

        post_version = "%d.%d.%d" % tuple(version)

    if pre_version != post_version:
        with open(versionfile, "w") as f:
            f.write('__version__ = "%s"\n' % post_version)

    return pre_version, post_version


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "versionfile",
        help="The version.py file containing the version string. "
        'It must contain a single line formatted as `__version__ = "A.B.C"` '
        "where A, B and C are major, minor and patch version numbers respectively.",
    )

    parser.add_argument("--major", action="store_true", default=False, help="Increment the major version number.")
    parser.add_argument("--minor", action="store_true", default=False, help="Increment the minor version number.")
    parser.add_argument("--patch", action="store_true", default=False, help="Increment the patch version number.")

    args = parser.parse_args()
    pre, post = update_version(versionfile=args.versionfile, major=args.major, minor=args.minor, patch=args.patch)
    if pre != post:
        print("Original version: %s; New version: %s" % (pre, post), file=sys.stderr)
    print(post, file=sys.stdout)
