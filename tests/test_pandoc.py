from __future__ import print_function

import pytest
import subprocess

from sphinx.util.osutil import cd


def parse_with_pandoc(file, to='native'):
    return subprocess.check_output(
        ['pandoc', '-f', 'vjson', '-t', to, '--standalone', file])


@pytest.mark.sphinx('pandoc')
def test_all(app, status, warning):
    app.builder.build_all()
    ast = app.outdir / app.config.master_doc + '.json'
    with cd(app.outdir):
        print(parse_with_pandoc(ast))


@pytest.mark.sphinx('pandoc', testroot='numfig',
                    confoverrides={'numfig': True})
def test_numfig(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    ast = app.outdir / app.config.master_doc + '.json'
    with cd(app.outdir):
        print(parse_with_pandoc(ast))