"""Tests for the host-agnostic emissions downloader (#498)."""

import os
import tempfile
import unittest

from jcm.data.emissions.downloader import fetch, stage


class DownloaderTest(unittest.TestCase):
    def test_local_path_returned_as_abspath(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "emis.nc")
            with open(p, "w") as fh:
                fh.write("x")
            # A local file resolves to itself (absolute), no download.
            self.assertEqual(fetch(p), os.path.abspath(p))

    def test_local_relative_path_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rel.nc")
            with open(p, "w") as fh:
                fh.write("x")
            cwd = os.getcwd()
            try:
                os.chdir(d)
                # Compared through realpath because what is under test is that
                # a relative path comes back absolute, not which of several
                # names for the same file it uses: on macOS the temporary
                # directory is reached as /var/folders/... while chdir'ing into
                # it leaves the cwd as the resolved /private/var/folders/...,
                # so abspath on either side alone gives two different strings
                # for the one file.
                self.assertEqual(os.path.realpath(fetch("rel.nc")),
                                 os.path.realpath(p))
            finally:
                os.chdir(cwd)

    def test_stage_copies_to_dest(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.nc")
            with open(src, "w") as fh:
                fh.write("payload")
            dest = os.path.join(d, "sub", "dest.nc")
            out = stage(src, dest)
            self.assertEqual(out, os.path.abspath(dest))
            with open(out) as fh:
                self.assertEqual(fh.read(), "payload")


if __name__ == "__main__":
    unittest.main()
