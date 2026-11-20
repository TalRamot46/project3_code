"""Locations used by the test-doc generator. Override with environment variables."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
# solver cache (pickles, untracked): results/fitting/cache in the repo root
CACHE = os.environ.get("FITGEN_CACHE", os.path.normpath(os.path.join(HERE, "..", "..", "..", "results", "fitting", "cache")))
# default output: project3_docs/fitting/tests next to project3_code (…/GitHub/project3_docs)
DOCS_OUT = os.environ.get("FITGEN_OUT", os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "project3_docs", "fitting", "tests")))
os.makedirs(CACHE, exist_ok=True)
