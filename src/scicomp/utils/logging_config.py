"""Logging Configuration."""

import logging


def setup_logging() -> None:
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
