# -*- coding: utf-8 -*-
"""
    sphinx.writers.pandoc
    ~~~~~~~~~~~~~~~~~~~~~

    Pandoc AST writer.

    :copyright: Copyright 2007-2018 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import itertools
import json
import re
import sys
from collections import namedtuple
from os import path
from typing import TYPE_CHECKING

from docutils import nodes
from docutils.writers import Writer

from sphinx import addnodes
from sphinx.util import logging

if TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple, Set, Union  # NOQA

logger = logging.getLogger(__name__)
RE_TOKENS = re.compile(r'(?P<white>\s+)|(?P<nonwhite>\S+)')


class collected_footnote(nodes.footnote):
    pass


def elt(eltType, numargs):
    def fun(*args):
        assert len(args) == numargs
        if numargs == 0:
            return {'t': eltType}
        elif len(args) == 1:
            xs = args[0]
        else:
            xs = list(args)
        return {'t': eltType, 'c': xs}

    return fun


# Below are defined a set of constructors for most important pandoc AST types.
# They must match the pandoc API reference (see
# http://hackage.haskell.org/package/pandoc-types-1.17.3.1/docs/Text-Pandoc-Definition.html)

# Enums
AlignLeft = elt('AlignLeft', 0)
AlignRight = elt('AlignRight', 0)
AlignCenter = elt('AlignCenter', 0)
AlignDefault = elt('AlignDefault', 0)

# Block elements
BlockQuote = elt('BlockQuote', 1)
BulletList = elt('BulletList', 1)
CodeBlock = elt('CodeBlock', 2)
DefinitionList = elt('DefinitionList', 1)
Div = elt('Div', 2)
Header = elt('Header', 3)
HorizontalRule = elt('HorizontalRule', 0)
LineBlock = elt('LineBlock', 1)
MetaInlines = elt('MetaInlines', 1)
Null = elt('Null', 0)
OrderedList = elt('OrderedList', 2)
Para = elt('Para', 1)
Plain = elt('Plain', 1)
RawBlock = elt('RawBlock', 2)
Table = elt('Table', 5)

# Inline elements
Cite = elt('Cite', 2)
NormalCitation = elt('NormalCitation', 0)
Code = elt('Code', 2)
DisplayMath = elt('DisplayMath', 0)
Emph = elt('Emph', 1)
Image = elt('Image', 3)
InlineMath = elt('InlineMath', 0)
LineBreak = elt('LineBreak', 0)
Link = elt('Link', 3)
Math = elt('Math', 2)
Note = elt('Note', 1)
Quoted = elt('Quoted', 2)
RawInline = elt('RawInline', 2)
SmallCaps = elt('SmallCaps', 1)
SoftBreak = elt('SoftBreak', 0)
Space = elt('Space', 0)
Span = elt('Span', 2)
Str = elt('Str', 1)
Strong = elt('Strong', 1)
Subscript = elt('Subscript', 1)
Superscript = elt('Superscript', 1)

LineBlockLine = namedtuple('LineBlockLine', 'contents')


def intercalate(glue, l):
    """Join elements of list `l` using list `glue`.

    >>> intercalate(['a', 'b'], [])
    []
    >>> intercalate(['a', 'b'], [12])
    [12]
    >>> intercalate(['a', 'b'], [12, 13])
    [12, 'a', 'b', 13]
    >>> intercalate(['a', 'b'], [12, 13, 14])
    [12, 'a', 'b', 13, 'a', 'b', 14]
    """
    return l[:1] + [part for el in l[1:] for part in glue + [el]]


class DefListItemBuilder:
    def __init__(self):
        self.terms = []  # type: List[Any]
        self.defs = []  # type: List[Any]


class TableCell:
    def __init__(self, content, colspan, rowspan):
        self.content = content
        self.colspan = colspan
        self.rowspan = rowspan

    @property
    def width(self):
        return 1 + self.colspan


class TableBuilder:
    """A builder for pandoc tables

    .. important::

        Unfortunately, pandoc tables are not expressive enough to support
        all ReST tables (see https://github.com/jgm/pandoc/issues/1024).
        In cases where input table cannot be converted, we try a "best looking"
        conversion (yeah, that's quite subjective) and report a warning.
    """

    def __init__(self, node, width):
        # type: (nodes.table, Union[int, float]) -> None
        self.headers = []  # type: List[unicode]
        self.rows = []  # type: List[List[Any]]
        self.node = node
        self.colwidths = []  # type: List[int]
        self.caption = None
        self.width = width

        self.in_header = False
        self.currow = []  # type: List[Any]

    @staticmethod
    def _row_width(row):
        return sum(el.width for el in row)

    @staticmethod
    def _gen_table(rows):
        """Convert a matrix of cells into a grid (i.e., expand colspan, rowspan)
        """
        if not rows:
            return []

        width = max(map(TableBuilder._row_width, rows))
        height = len(rows) + max(x.rowspan for x in rows[-1])
        res = [[None for _ in range(width)] for _ in range(height)]
        for i, row in enumerate(rows):
            j = 0
            for el in row:
                while res[i][j] is not None:
                    j += 1
                indices = itertools.product(
                    range(i, i + el.rowspan + 1), range(j, j + el.colspan + 1))
                for kr, kc in indices:
                    res[kr][kc] = []  # type: ignore
                res[i][j] = el.content

        return res

    def start_thead(self):
        self.in_header = True

    def leave_thead(self):
        assert self.in_header
        self.in_header = False
        self.headers = self._gen_table(self.rows)
        self.rows = []

    def add_colspec(self, width):
        self.colwidths.append(width)

    def next_row(self):
        self.rows.append(self.currow)
        self.currow = []

    def add_cell(self, content, morecols, morerows=0):
        if morecols > 0:
            logger.warning(
                "pandoc doesn't support colspan > 1, filling with empty cells "
                "instead.",
                location=self.node)
        if morerows > 0:
            logger.warning(
                "pandoc doesn't support rowspan > 1, filling with empty cells "
                "instead.",
                location=self.node)
        self.currow.append(TableCell(content, morecols, morerows))

    def as_pandoc_ast(self):
        if len(self.headers) > 1:
            logger.warning(
                "pandoc doesn't support more than one row in a table header, "
                "keeping only the first row",
                location=self.node)
        header = self.headers[0] if len(self.headers) > 0 else []
        caption = self.caption or []
        column_align = [AlignDefault() for _ in range(len(self.colwidths))]
        total_width = sum(self.colwidths)
        relative_widths = [self.width * float(x) / total_width
                           for x in self.colwidths]
        rows = TableBuilder._gen_table(self.rows)
        return Table(caption, column_align, relative_widths, header, rows)


class PandocWriter(Writer):
    def __init__(self, builder):
        Writer.__init__(self)
        self.builder = builder

    def translate(self):
        visitor = self.builder.create_translator(self.document, self.builder)
        self.document.walkabout(visitor)
        meta = {}
        if visitor.title is not None:
            meta['title'] = visitor.title
        output = {
            'blocks': visitor.body,
            'pandoc-api-version': [1, 17, 3],
            'meta': meta
        }
        self.output = json.dumps(output, indent=2)


def _pop_with(el):
    def func(self, node):
        contents = self.pop()
        self.body.append(el(contents))

    return func


def _div(div_classes, contents, style=None):
    """Create a div with an optional 'custom-style' attribute
    """
    attrs = [["custom-style", style]] if style else []
    return Div(["", div_classes, attrs], contents)


def _admonition_contents(name, title, contents, style=None):
    title_style = style + "Title" if style else None
    return Div(["", [name], []], [
        _div(["admonition-title"], [Para([Str(title)])], style=title_style),
        _div(["adminition-title"], contents, style=style)
    ])


def _admonition(name, title, style=None):
    def func(self, node):
        contents = self.pop()
        self.body.append(
            _admonition_contents(name, title, contents, style=style))

    return func


def _div_wrap(*classes):
    def func(self, node):
        contents = self.pop()
        # TODO: id
        self.body.append(Div(["", classes, []], contents))

    return func


class PandocTranslator(nodes.NodeVisitor):
    ignore_missing_images = False

    def __init__(self, document, builder):
        nodes.NodeVisitor.__init__(self, document)
        self.builder = builder
        self._convert_size = self._gen_convert_size(builder)

        self.top_in_section = 1
        self.in_section = 0
        self.in_line_block = 0
        self.in_list = 0
        self.in_title = 0
        self.in_caption = 0
        self.in_footnote = 0
        self.in_term = 0
        self.in_parsed_literal = 0
        self.in_toc = False

        self.body_stack = []  # type: List[Any]
        self.body = []  # type: List[Any]
        self.title = None
        self.caption = None
        self.legend = None
        self.table = None
        self.def_list = None

        self.curfilestack = []  # type: List[unicode]
        self.footnotestack = [
        ]  # type: List[Dict[unicode, Tuple[collected_footnote, bool]]]
        self.hlsettingstack = \
            2 * [[builder.config.highlight_language, sys.maxsize]]
        self.next_section_ids = set()  # type: Set[unicode]
        self.next_figure_ids = set()  # type: Set[unicode]
        self.next_table_ids = set()  # type: Set[unicode]
        self.next_listing_ids = set()  # type: Set[unicode]
        self.handled_abbrs = set()  # type: Set[unicode]

    def _skip(self, node):
        raise nodes.SkipNode

    def _pass(self, node):
        pass

    def _push(self, node):
        self.push()

    def _pop_flat(self, node):
        contents = self.pop()
        self.body.append(contents)

    @staticmethod
    def _gen_convert_size(builder):
        """Create a size conversion function

        Not all pandoc backends provide support for relative size (e.g., the
        docx backend does not). By default, no conversion is applied, the user
        needs to set the option pandoc_force_absolute_size = True to enable
        conversion. In such case, the maximum size (i.e., the 100% value) must
        also be provided ('textwidth' attribute in pandoc_options config
        values).

        Defaults to the identity function (no conversion applied)
        """
        def identity(x):
            return x

        if not builder.config.pandoc_force_absolute_size:
            return identity

        textwidth = builder.config.pandoc_options.get('textwidth', None)
        if not textwidth:
            logger.warning(
                "using options 'pandoc_force_absolute_size' requires textwidth"
                "to be set in pandoc_options. For instance, define\n"
                "pandoc_options = {'textwidth': (16, 'cm')} in your conf.py")
            return identity

        def convert(measure):
            if not measure.endswith('%'):
                return measure

            width, unit = textwidth
            f = float(measure[:-1])
            return str(f * width / 100) + unit

        return convert

    @staticmethod
    def _pop_ids(id_set):
        """Given a set of ids, pop an arbitrary element and clear the set

        If no id is found, returns an empty string ("")
        """
        id = ""
        try:
            id = list(id_set)[0]
        except IndexError:
            pass
        id_set.clear()
        return id

    @staticmethod
    def _is_listing(node):
        return (isinstance(node, nodes.container) and
                'literal-block-wrapper' in node.get('classes', []))

    def push(self, head=None):
        head = head or []
        self.body_stack.append(self.body)
        self.body_stack.append(head)
        self.body = head

    def pop(self):
        out = self.body_stack.pop()
        self.body = self.body_stack.pop()
        return out

    def dispatch_visit(self, node):
        if isinstance(node.parent, nodes.sidebar):
            raise nodes.SkipNode
        return super(PandocTranslator, self).dispatch_visit(node)

    def unknown_visit(self, node):
        logger.warning("not implemented: '%s'", node.tagname)
        raise nodes.SkipNode

    def hypertarget(self, id, withdoc=True):
        if withdoc:
            id = self.curfilestack[-1] + ':' + id
        return id

    @staticmethod
    def is_inline(node):
        return isinstance(node.parent, nodes.TextElement)

    @staticmethod
    def get_text(text):
        tokens = []

        def matcher(m):
            if m.group('white'):
                token = Space()
            else:
                token = Str(m.group('nonwhite'))
            tokens.append(token)

        RE_TOKENS.sub(matcher, text)
        return tokens

    @staticmethod
    def hyperlink(id, contents):
        return Link(["", [], []], contents, [id, ""])

    @staticmethod
    def collect_footnotes(node):
        def footnotes_under(n):
            if isinstance(n, nodes.footnote):
                yield n
            else:
                for c in n.children:
                    if isinstance(c, addnodes.start_of_file):
                        continue
                    for k in footnotes_under(c):
                        yield k

        fnotes = {}
        for fn in footnotes_under(node):
            num = fn.children[0].astext().strip()
            newnode = collected_footnote(*fn.children, number=num)
            fnotes[num] = [newnode, False]
        return fnotes

    def visit_start_of_file(self, node):
        # collect new footnotes
        self.footnotestack.append(self.collect_footnotes(node))
        # document target
        self.curfilestack.append(node['docname'])
        # use default highlight settings for new file
        self.hlsettingstack.append(self.hlsettingstack[0])

    def depart_start_of_file(self, node):
        self.footnotestack.pop()
        self.curfilestack.pop()
        self.hlsettingstack.pop()

    def visit_document(self, node):
        self.footnotestack.append(self.collect_footnotes(node))
        self.curfilestack.append(node.get('docname', ''))
        self.in_section = self.top_in_section - 1

    depart_document = _pass

    visit_comment = _skip

    visit_substitution_definition = _skip

    def visit_Text(self, node):
        self.body.extend(self.get_text(node.astext()))
        # glue space and footnote reference
        if (self.body and self.body[-1]["t"] == "Space" and isinstance(
                node.next_node(siblings=True), nodes.footnote_reference)):
            self.body.pop()
        raise nodes.SkipNode

    def visit_section(self, node):
        self.in_section += 1

    def depart_section(self, node):
        self.in_section = max(self.in_section - 1, self.top_in_section - 1)

    visit_rubric = _push

    def depart_rubric(self, node):
        contents = self.pop()
        if len(contents) > 0 and contents[0].get('c') == 'Footnotes':
            # As within the latex backend we pattern match the 'Footnotes'
            # section and exclude it from the output
            return
        # NOTE: a rubic is currently implemented as a named paragraph
        self.body.append(Para([Strong(contents)]))

    def visit_title(self, node):
        if isinstance(node.parent, nodes.Admonition):
            # Admonition title is handled in depart_admonition
            raise nodes.SkipNode
        self.push()

    def depart_title(self, node):
        contents = self.pop()
        if isinstance(node.parent, nodes.table):
            id = self._pop_ids(self.next_table_ids)
            prefix = self._get_numref_prefix(node.parent)
            self.table.caption = [Span([id, [], []], prefix + contents)]
            return

        if isinstance(node.parent, nodes.topic):
            self.body.append(Para([Span(["", ["topic-title"], []], contents)]))
            return

        if isinstance(node.parent, nodes.section):
            id = self._pop_ids(self.next_section_ids)
            if self.title is None:
                if id:
                    contents = [Span([id, [], []], contents)]
                self.title = MetaInlines(contents)
                self.in_section = 0
                return
            assert self.in_section > 0
            self.body.append(Header(self.in_section, [id, [], []], contents))
        else:
            # TODO
            pass

    visit_compound = depart_compound = _pass

    visit_paragraph = _push

    def depart_paragraph(self, node):
        contents = self.pop()

        if len(node.get('ids', [])) > 0 \
                and node['ids'][0].startswith("bibtex-bibliography"):
            # NOTE: A citation entry (generated by sphinxcontrib-bibtex) is a
            # paragraph (with a special id). So, we need to catch this special
            # case there.
            self.body.append(Div(["citations", [], []], contents))
            return

        cls = Para
        if self.in_list > 0:
            cls = Plain
        elem = cls(contents)
        self.body.append(elem)

    def visit_literal_block(self, node):
        if node.rawsource != node.astext():
            # most probably a parsed-literal block -- don't highlight
            raise nodes.SkipNode

        classes = []
        opts = []
        code = node.astext()
        lang = self.hlsettingstack[-1][0]
        if node.hasattr('language'):
            lang = node['language']
        if lang and lang != 'default':
            classes.append("sourceCode")
            classes.append(lang)
        if node.get('linenos') is True:
            classes.append("numberLines")
            try:
                start = node['highlight_args']['linenostart']
                if start != 1:
                    opts.append(["startFrom", str(start)])
            except KeyError:
                pass

        self.body.append(CodeBlock(["", classes, opts], code))
        raise nodes.SkipNode

    def depart_literal_block(self, node):
        self.in_parsed_literal -= 1

    visit_doctest_block = visit_literal_block
    depart_doctest_block = depart_literal_block

    def visit_bullet_list(self, node):
        # TODO
        # if self.within_toc:
        #     return
        self.in_list += 1
        self.push()

    def depart_bullet_list(self, node):
        # TODO
        # if self.within_toc:
        #     return
        self.in_list -= 1
        contents = self.pop()
        self.body.append(BulletList(contents))

    visit_definition_list = _push
    depart_definition_list = _pop_with(DefinitionList)

    visit_list_item = _push
    depart_list_item = _pop_flat

    def visit_definition_list_item(self, node):
        self.def_list = DefListItemBuilder()

    def depart_definition_list_item(self, node):
        # Pandoc supports multiple definitions, but not multiple terms.
        # So, we separate them by comma + space.
        terms = intercalate([Str(","), Space()], self.def_list.terms)
        self.body.append([terms, self.def_list.defs])
        self.def_list = None

    def visit_term(self, node):
        for n in node[:]:
            if isinstance(n, (addnodes.index, nodes.target)):
                n.walkabout(self)
                node.remove(n)
        self.push()

    def depart_term(self, node):
        contents = self.pop()
        id = ""
        if node.get('ids'):
            # glossary term, wrap with a Span with the right id
            id = self.hypertarget(node['ids'][0])
        contents = Span([id, [], []], contents)
        self.def_list.terms.append(contents)

    visit_definition = _push

    def depart_definition(self, node):
        contents = self.pop()
        self.def_list.defs.append(contents)

    # TODO?
    visit_index = _skip

    # TODO: what is this?
    visit_classifier = depart_classifier = _pass

    visit_container = _push

    def depart_container(self, node):
        contents = self.pop()
        id = ""
        opts = []
        if self._is_listing(node):
            id = self._pop_ids(self.next_listing_ids)
            opts.append(["custom-style", "Table Caption"])
        if self.caption:
            caption = Span([id, ["caption"], []], self.caption)
            self.body.append(Div(["", ["caption"], opts], [Plain([caption])]))
        self.body.append(Div(["", ["container"], []], contents))
        self.caption = None

    visit_inline = _push

    def depart_inline(self, node):
        contents = self.pop()
        self.body.append(Span(["", node.get('classes', []), []], contents))

    visit_strong = _push

    depart_strong = _pop_with(Strong)

    def visit_literal_strong(self, node):
        self.body.append(Strong([Code(["", [], []], node.astext())]))
        raise nodes.SkipNode

    visit_emphasis = _push

    depart_emphasis = _pop_with(Emph)

    def visit_literal_emphasis(self, node):
        self.body.append(Emph([Code(["", [], []], node.astext())]))
        raise nodes.SkipNode

    def visit_highlightlang(self, node):
        self.hlsettingstack[-1] = [node['lang'], node['linenothreshold']]
        raise nodes.SkipNode

    visit_reference = _push

    def depart_reference(self, node):
        contents = self.pop()

        uri = node.get('refuri', '')
        if not uri and node.get('refid'):
            uri = '%' + self.curfilestack[-1] + '#' + node['refid']

        if uri.startswith('#'):
            # references to labels in the same document
            id = '#' + self.hypertarget(uri[1:])
            self.body.append(self.hyperlink(id, contents))
        elif uri.startswith('%'):
            # references to documents or labels inside documents
            hashindex = uri.find('#')
            if hashindex == -1:
                # reference to the document itself (??)
                id = uri[1:hashindex]
            else:
                # reference to a label
                id = '#{}:{}'.format(uri[1:hashindex], uri[hashindex + 1:])
            self.body.append(self.hyperlink(id, contents))
        else:
            if len(node) == 1 and uri == node[0]:
                self.body.append(self.hyperlink(uri, contents))
            else:
                self.body.append(self.hyperlink(uri, contents))

    def visit_target(self, node):
        # postpone the labels until after the sectioning command
        parindex = node.parent.index(node)
        try:
            try:
                next = node.parent[parindex + 1]
            except IndexError:
                # last node in parent, look at next after parent
                # (for section of equal level)
                next = node.parent.parent[node.parent.parent.index(
                    node.parent)]

            ids = set()
            if node.get('refid'):
                ids.add(self.hypertarget(node['refid']))
            ids.update(self.hypertarget(id) for id in node['ids'])

            if isinstance(next, nodes.section):
                self.next_section_ids.update(ids)
                return

            if isinstance(next, nodes.figure):
                self.next_figure_ids.update(ids)
                return

            if isinstance(next, nodes.table):
                self.next_table_ids.update(ids)
                return

            if self._is_listing(next):
                self.next_listing_ids.update(ids)
                return
        except (IndexError, AttributeError):
            pass

        # TODO: other cases?

    depart_target = _pass

    def visit_enumerated_list(self, node):
        self.in_list += 1
        self.push()

    def depart_enumerated_list(self, node):
        # type: (nodes.Node) -> None
        self.in_list -= 1
        contents = self.pop()
        style = {
            "upperalpha": "UpperAlpha",
            "loweralpha": "LowerAlpha",
            "upperroman": "UpperRoman",
            "lowerroman": "LowerRoman",
            "arabic": "Decimal",
        }.get(node.get('enumtype'), "DefaultStyle")
        delim = {
            ".": "Period",
        }.get(node.get('suffix'), "DefaultDelim")
        start = 1
        if node.hasattr('start'):
            start = node['start']
        self.body.append(
            OrderedList([start, {
                "t": style
            }, {
                "t": delim
            }], contents))

    def visit_footnote_reference(self, node):
        num = node.astext().strip()
        try:
            footnode, used = self.footnotestack[-1][num]
        except (KeyError, IndexError):
            raise nodes.SkipNode
        # footnotes are repeated for each reference
        footnode.walkabout(self)
        raise nodes.SkipChildren

    depart_footnote_reference = _pass

    def visit_footnote(self, node):
        # already done though collect_footnotes()
        raise nodes.SkipNode

    def visit_collected_footnote(self, node):
        self.in_footnote += 1
        self.push()

    def depart_collected_footnote(self, node):
        self.in_footnote -= 1
        contents = self.pop()
        self.body.append(Note(contents))

    def visit_line_block(self, node):
        self.in_line_block += 1
        self.push()

    def depart_line_block(self, node):
        self.in_line_block -= 1

        if self.in_line_block > 0:
            contents = self.pop()
            self.body.append(contents)
        else:
            # RST parser nests line_blocks for each line, while pandoc expects
            # a single LineBlock with child lines
            def unfold(data, depth):
                if isinstance(data, LineBlockLine):
                    if depth:
                        yield [Str('\xa0' * depth)] + data.contents
                    else:
                        yield data.contents
                    return
                for el in data:
                    # py2 compatible 'yield from unfold(el, depth + 1)'
                    for e in unfold(el, depth + 1):
                        yield e

            contents = self.pop()
            self.body.append(LineBlock(list(unfold(contents, -1))))

    visit_line = _push

    depart_line = _pop_with(LineBlockLine)

    def visit_transition(self, node):
        self.body.append(HorizontalRule())
        raise nodes.SkipNode

    visit_superscript = _push

    depart_superscript = _pop_with(Superscript)

    visit_subscript = _push

    depart_subscript = _pop_with(Subscript)

    def visit_math_block(self, node):
        math = Math(DisplayMath(), node.astext())
        self.body.append(Para([math]))  # go figure
        raise nodes.SkipNode

    def visit_math(self, node):
        self.body.append(Math(InlineMath(), node.astext()))
        raise nodes.SkipNode

    def visit_label(self, node):
        self.body.append(Str("[" + node[0] + "]"))
        raise nodes.SkipNode

    visit_citation = _push

    def depart_citation(self, node):
        contents = self.pop()
        try:
            anchor = self.hypertarget(node['ids'][0])
        except IndexError:
            logger.warning(
                "Failed to create reference target: no ids found",
                location=node)
            return
        # Expected contents shape:
        #   contents[0] -> Str (label)
        #   contents[1] -> Para (label description)
        cite_para = [
            Span([anchor, ["citation-label"], []], [contents[0]]),
            Space()
        ] + contents[1]['c']
        self.body.append(Div(["citation", [], []], [Para(cite_para)]))

    def visit_literal(self, node):
        self.body.append(Code(["", [], []], node.astext()))
        raise nodes.SkipNode

    visit_block_quote = _push

    depart_block_quote = _pop_with(BlockQuote)

    def visit_image(self, node):
        if node['uri'] in self.builder.images:
            uri = self.builder.images[node['uri']]
        else:
            # missing image!
            if self.ignore_missing_images:
                raise nodes.SkipNode
            uri = node['uri']
        if uri.find('://') != -1:
            # ignore remote images
            raise nodes.SkipNode
        attrs = []

        alt = Str(node.attributes.get('alt', ''))
        for attr in ('width', 'height'):
            if node.hasattr(attr):
                attrs.append([attr, self._convert_size(node[attr])])

        if self.builder.config.pandoc_convert_svg_to_png and uri.endswith(
                '.svg'):
            uri = path.splitext(uri)[0] + '.png'

        img = Image(["", [], attrs], [alt], [uri, ""])
        content = Para([img])
        if node.parent.tagname == 'paragraph':
            # NOTE: a substitution with an image will already generate a
            # paragraph. So, we need to return just the inline image in that
            # case.
            content = img

        self.body.append(content)
        raise nodes.SkipNode

    def visit_acks(self, node):
        text = self.get_text(', '.join(
            n.astext() for n in node.children[0].children) + '.')
        self.body.append(Para(text))
        raise nodes.SkipNode

    visit_note = _push
    visit_important = _push
    visit_warning = _push
    visit_tip = _push
    visit_seealso = _push
    # NOTE: for some reasons, one style is created for each todo...
    # support is left as an improvement
    visit_todo_node = _push

    depart_note = _admonition("note", "Note", style="Note")
    depart_important = _admonition("important", "Important", style="Warning")
    depart_warning = _admonition("warning", "Warning", style="Warning")
    depart_tip = _admonition("tip", "Tip", style="Tip")
    depart_seealso = _admonition("seealso", "See Also", style="Tip")
    depart_todo_node = _admonition("todo", "Todo", style="Warning")

    visit_admonition = _push

    def depart_admonition(self, node):
        contents = self.pop()
        title = node.children[0].astext()
        name = node['classes'][0]
        self.body.append(_admonition_contents(name, title, contents))

    def visit_table(self, node):
        width = self.builder.config.pandoc_options.get('max_tab_width', 1.0)
        self.table = TableBuilder(node, width)

    def depart_table(self, node):
        self.body.append(self.table.as_pandoc_ast())
        self.table = None

    visit_tgroup = _pass
    depart_tgroup = _pass

    def visit_colspec(self, node):
        self.table.add_colspec(node['colwidth'])
        raise nodes.SkipNode

    def visit_thead(self, node):
        self.table.start_thead()

    def depart_thead(self, node):
        self.table.leave_thead()

    visit_tbody = _pass
    depart_tbody = _pass
    visit_row = _pass

    def depart_row(self, node):
        self.table.next_row()

    visit_entry = _push

    def depart_entry(self, node):
        contents = self.pop()
        self.table.add_cell(contents, node.get('morecols', 0),
                            node.get('morerows', 0))

    visit_figure = _push

    def _get_numref(self, node):
        """ Extract the numref caption prefix associated with the given node

        :return:
            None if no numbering was found.
            When a configuration option is missing it will return None and
            reports a warning.
            In case of success, returns the string prefix (e.g., "Fig 42. ")
        """
        fig_type = self.builder.env.domains['std'].get_figtype(node)
        if not fig_type:
            logger.warning(
                "figure type {} is unknown".format(fig_type), location=node)
            return None

        if len(node['ids']) == 0:
            logger.warning(
                "No ids assigned for {}".format(node.tagname), location=node)
            return None
        fig_id = node['ids'][0]

        key = "{}/{}".format(self.curfilestack[-1], fig_type)
        fig_numbers = self.builder.fignumbers.get(key, {}).get(fig_id)
        if not fig_numbers:
            # Normal execution path when compiling with numfig = False (default)
            return None

        prefix = self.builder.config.numfig_format.get(fig_type)
        if not prefix:
            logger.warning(
                "numfig_format has to entry for type {}".format(fig_type))
            return None

        # NOTE: an extra space is appended by default.
        #       However, it could be part of the user defined prefix
        return prefix % '.'.join(map(str, fig_numbers)) + " "

    def _get_numref_prefix(self, node):
        """Convenience wrapper around _get_numref

        :return:
            [] when no numref is associated with the node.
            In case of success, returns a list of pandoc Inline for the numref.
        """
        numref = self._get_numref(node)
        return [Str(numref)] if numref else []

    def depart_figure(self, node):
        image = self.pop()[0]["c"][0]
        _, classes, opts = image["c"][0]
        if node.hasattr('align'):
            classes.append("align-" + node['align'])
        for attr in ('width', 'height'):
            if node.hasattr(attr):
                opts.append([attr, self._convert_size(node[attr])])
        image["c"][1] = self.caption or []
        image["c"][2][1] = "fig:"
        id = self._pop_ids(self.next_figure_ids)
        self.body.append(
            Div([id, ["figure"], []], [Para([image])] + (self.legend or [])))
        self.caption = None
        self.legend = None

    visit_legend = _push

    def depart_legend(self, node):
        self.legend = self.pop()

    visit_caption = _push

    def depart_caption(self, node):
        prefix = self._get_numref_prefix(node.parent)
        self.caption = prefix + self.pop()

    visit_glossary = _push

    def depart_glossary(self, node):
        contents = self.pop()
        self.body.append(contents[0])

    visit_topic = _push

    depart_topic = _div_wrap("topic")

    visit_centered = _push

    def depart_centered(self, node):
        contents = self.pop()
        self.body.append(_div(["centered"], [Plain(contents)]))

    visit_number_reference = _push

    def depart_number_reference(self, node):
        contents = self.pop()
        # FIXME (TH): why not reusing reference handling from depart_reference ?
        if node.get('refid'):
            id = self.hypertarget(node['refid'])
        else:
            id = node.get('refuri', '')[1:].replace('#', ':')
        self.body.append(self.hyperlink("#" + id, contents))

    visit_download_reference = _push

    def depart_download_reference(self, node):
        contents = self.pop()
        if node.hasattr('reftarget'):
            uri = 'file://' + node['reftarget']
            contents = self.hyperlink(uri, contents)
        self.body.append(contents)

    visit_abbreviation = _push

    def depart_abbreviation(self, node):
        contents = self.pop()
        abbr = node.astext()
        # spell out the explanation once
        if node.hasattr('explanation') and abbr not in self.handled_abbrs:
            self.handled_abbrs.add(abbr)
            # append explanation
            contents.extend(self.get_text(' ({})'.format(node['explanation'])))
        self.body.append(Span(["", ["abbr"], []], contents))
