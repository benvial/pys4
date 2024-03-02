#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


import subprocess
from contextlib import suppress
from pathlib import Path

from setuptools import Command, setup
from setuptools.command.build import build


class CustomCommand(Command):
    def initialize_options(self):
        self.bdist_dir = None

    def finalize_options(self):
        with suppress(Exception):
            self.bdist_dir = Path(self.get_finalized_command("bdist_wheel").bdist_dir)

    def run(self):
        try:
            proc = subprocess.Popen(
                [
                    "make",
                    "install-S4",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc.wait()
            (stdout, stderr) = proc.communicate()

        except calledProcessError as err:
            print("Error ocurred: " + err.stderr)


class CustomBuild(build):
    sub_commands = [("build_custom", None)] + build.sub_commands


setup(cmdclass={"build": CustomBuild, "build_custom": CustomCommand})
