#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import os
import time

import numpy as np

from pys4 import _S4


def test_simple():
    import pys4
    from pys4 import _S4

import pytest
from types import SimpleNamespace
import importlib.metadata as importlib_metadata
import pys4  # replace with the actual module name

# --- TESTS ---


def test_get_metadata_success(monkeypatch):
    # Simulate metadata and version
    class FakeMetadata:
        def metadata(self, package_name):
            return {"author": "John Doe", "summary": "Test package"}

        def version(self, package_name):
            return "1.2.3"

    monkeypatch.setattr(importlib_metadata, "metadata", FakeMetadata().metadata)
    monkeypatch.setattr(importlib_metadata, "version", FakeMetadata().version)

    version, author, desc, data = pys4.__about__._get_metadata(importlib_metadata)

    assert version == "1.2.3"
    assert author == "John Doe"
    assert desc == "Test package"
    assert data["author"] == "John Doe"
    assert data["summary"] == "Test package"


def test_get_metadata_failure(monkeypatch):
    # Simulate an exception in metadata access
    def raise_exception(*args, **kwargs):
        raise Exception("package not found")

    monkeypatch.setattr(importlib_metadata, "metadata", raise_exception)
    monkeypatch.setattr(importlib_metadata, "version", raise_exception)

    version, author, desc, data = pys4.__about__._get_metadata(importlib_metadata)

    assert version == "unknown"
    assert author == "unknown"
    assert desc == "unknown"
    assert data["License"] == "unknown"
