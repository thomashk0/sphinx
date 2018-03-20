from __future__ import print_function

import collections

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


def parse_json_ast(app):
    ast = app.outdir / (app.config.master_doc + '.json')
    return json.loads(ast.text(encoding='utf-8'))


def json_content(app):
    ast = app.outdir / (app.config.master_doc + '.json')
    return ast.text(encoding='utf-8')


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


def duplicates(iter):
    return ((item, count) for item, count in collections.Counter(iter).items()
            if count > 1)


def check_internal_refs(json_ast, ignore=set(), ignore_cat=set()):
    """Checks internal references consistency in a pandoc document
    """
    def _keep_node(x):
        if x in ignore:
            return False

        cat = x[1:].split(':')[0]
        return cat not in ignore_cat

    def _extract_ids(block, attr_idx):
        return (node[attr_idx][0] for node in find_pandoc_node(json_ast, block)
                if node[attr_idx][0])

    refs = set(node[2][0] for node in find_pandoc_node(json_ast, 'Link')
               if _keep_node(node[2][0]))

    ids = []
    ids.extend(_extract_ids('Span', 0))
    ids.extend(_extract_ids('Div', 0))
    ids.extend(_extract_ids('Header', 1))

    assert [] == list(duplicates(ids)), \
        "no id (div, span, header) should be duplicated accross the pandoc " \
        "document"

    ids = set(ids)

    for r in refs:
        if r.startswith('#'):
            assert r[1:] in ids, \
                "every reference must have an associated span id"


@pytest.mark.sphinx('pandoc')
def test_all(app, status, warning):
    app.builder.build_all()
    check_pandoc_parsing(app)


@pytest.mark.sphinx('pandoc', testroot='latex-table')
def test_table(app, status, warning):
    app.builder.build_all()
    check_pandoc_parsing(app)
    warnings = warning.getvalue()

    assert "rowspan > 1" in warnings, "should warn about rowspan policy"
    assert "colspan > 1" in warnings, "should warn about colspan policy"

    json_ast = parse_json_ast(app)
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
    check_pandoc_parsing(app)
    ast = app.outdir / (app.config.master_doc + '.json')
    json_ast = json.loads(ast.text(encoding='utf-8'))
    check_internal_refs(json_ast)


@pytest.mark.sphinx(
    'pandoc', testroot='numfig',
    confoverrides={'numfig': True,
                   'pandoc_use_short_refs': True})
def test_ref_renaming(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    check_pandoc_parsing(app)
    ast = app.outdir / (app.config.master_doc + '.json')
    json_ast = json.loads(ast.text(encoding='utf-8'))
    check_internal_refs(json_ast)


@pytest.mark.sphinx('pandoc', testroot='pandoc')
def test_substitution(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    check_pandoc_parsing(app)
    assert "warning.png" in warnings, "must warn about image not found"
    assert "warning2.png" in warnings, "must warn about image not found"
    json_str = json_content(app)
    assert "42%" in json_str


@pytest.mark.sphinx('pandoc', testroot='pandoc',
                    confoverrides={'pandoc_force_absolute_size': True})
def test_image_resize_warning(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    assert "requires textwidth" in warnings, \
        "warns about missing configuration"
    assert "reference id longer than 40 character" in warnings, \
        "warns about too long reference"
    json_ast = json_content(app)
    assert "42%" in json_ast, "no size conversion was done"


@pytest.mark.sphinx('pandoc', testroot='pandoc',
                    confoverrides={'pandoc_force_absolute_size': True,
                                   'pandoc_options': {'textwidth': (40, 'cm')}
                                   })
def test_image_resize(app, status, warning):
    app.builder.build_all()
    warnings = warning.getvalue()
    assert "requires textwidth" not in warnings, \
        "warns about missing configuration"
    json_ast = json_content(app)
    assert "42%" not in json_ast
    assert "16.8cm" in json_ast
