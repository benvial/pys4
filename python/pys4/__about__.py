#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: The pyS4 Developers
# This file is part of pyS4
# License: GPLv3


"""Package metadata."""

import importlib.metadata as metadata


def _get_metadata(metadata):
    """
    Get package metadata.

    Returns
    -------
        tuple: A tuple of package metadata including version, author,
            description, and a dictionary of additional metadata.
    """
    try:
        data = metadata.metadata("pys4")
        __version__ = metadata.version("pys4")
        __author__ = data.get("author")
        __description__ = data.get("summary")
    except Exception:
        data = dict(License="unknown")
        __version__ = "unknown"
        __author__ = "unknown"
        __description__ = "unknown"
    return __version__, __author__, __description__, data


__version__, __author__, __description__, data = _get_metadata(metadata)
