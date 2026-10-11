"""Targeted stopped-seed path/full-process scope archive regressions."""
import sys
sys.dont_write_bytecode=True
from round23_fixtures import run
run("paths")
