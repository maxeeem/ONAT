from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Deprecated WSC evaluation entrypoint.")
    parser.parse_args()
    print("wsc_eval.py has been retired.")
    print("Reason: the previous version used a mocked tie-breaker and did not run a valid end-to-end ONA evaluation.")
    print("Use benchmark.py for executable, non-mocked ablations on the local constructed benchmark.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
