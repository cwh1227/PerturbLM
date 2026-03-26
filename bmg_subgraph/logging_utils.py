"""Logging setup helpers."""

import logging


def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("bmg_subgraph")
