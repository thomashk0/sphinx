from __future__ import print_function

import pytest
import subprocess
import json

from sphinx.util.osutil import cd
from sphinx.writers.pandoc import Str


def has_pandoc():
    try:
        subprocess.check_call(['pandoc', '--version'])
        return True
    except subprocess.CalledProcessError:
        return False


pytestmark = pytest.mark.skipif(
    not has_pandoc(), reason="pandoc is not installed")


def parse_with_pandoc(file, to='native'):
    return subprocess.check_output(
        ['pandoc', '-f', 'json', '-t', to, '--standalone', file])


def check_pandoc_parsing(app):
    ast = app.outdir / (app.config.master_doc + '.json')
    with cd(app.outdir):
        parse_with_pandoc(ast)


def find_pandoc_node(tree, node_type):
    if isinstance(tree, dict):
        if tree.get('t') == node_type:
            yield tree['c']
        else:
            for v in tree.values():
                for el in find_pandoc_node(v, node_type):
                    yield el
    elif isinstance(tree, list):
        for v in tree:
            for el in find_pandoc_node(v, node_type):
                yield el


@pytest.mark.sphinx('pandoc')
def test_all(app, status, warning):
    app.builder.build_all()
    ast = app.outdir / app.config.master_doc + '.json'
    with cd(app.outdir):
        parse_with_pandoc(ast)


@pytest.mark.sphinx('pandoc', testroot='latex-table')
def test_table(app, status, warning):
    app.builder.build_all()
    check_pandoc_parsing(app)
    warnings = warning.getvalue()

    assert "rowspan > 1" in warnings, "should warn about rowspan policy"
    assert "colspan > 1" in warnings, "should warn about colspan policy"

    ast = app.outdir / (app.config.master_doc + '.json')
    json_ast = json.loads(ast.text(encoding='utf-8'))
    tables = list(find_pandoc_node(json_ast, 'Table'))
    for t in tables:
        rows = t[4]
        assert [len(rows[0])] * len(rows) == list(map(len, rows)), \
            "all rows must have the same length"

    # quickcheck of complex.rst : grid table
    rows = tables[-2][4]
    assert rows[1][1] == [], "colspan empty cell filling failed"
    assert rows[2][0] == [], "rowspan empty filling failed"
    assert rows[2][1] != []
    assert rows[2][2] == []


@pytest.mark.sphinx('pandoc')
def test_rubric(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    ast = app.outdir / (app.config.master_doc + '.json')
    ast_json = json.loads(ast.text(encoding='utf-8'))
    strong = list(find_pandoc_node(ast_json, 'Strong'))
    assert [Str("Footnotes")] not in strong, \
        "rubric 'Footnotes' must have been removed"
    assert [Str("Citations")] in strong


@pytest.mark.sphinx(
    'pandoc', testroot='numfig', confoverrides={'numfig': True})
def test_numfig(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    ast = app.outdir / app.config.master_doc + '.json'
    with cd(app.outdir):
        parse_with_pandoc(ast)


@pytest.mark.sphinx('pandoc', testroot='pandoc')
def test_substitution(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    check_pandoc_parsing(app)
    assert "warning.png" in warnings, "must warn about image not found"
    assert "warning2.png" in warnings, "must warn about image not found"
