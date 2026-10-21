"""Run offloading CPU contracts without importing serving-only dependencies.

This installs only the sglang package namespace. Tested modules and torch are
loaded normally; no offloading implementation is mocked. The optional source
root is useful when running an isolated copy of the offloading package.
"""

import argparse
import pathlib
import sys
import types
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path)
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()
    test_dir = pathlib.Path(__file__).resolve().parent
    root = test_dir.parents[3]
    package = types.ModuleType("sglang")
    package.__path__ = [str(root / "python" / "sglang")]
    if args.source_root is not None:
        package.__path__.append(str(args.source_root.resolve() / "python" / "sglang"))
    sys.modules["sglang"] = package
    suite = unittest.defaultTestLoader.discover(str(test_dir), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(main())
