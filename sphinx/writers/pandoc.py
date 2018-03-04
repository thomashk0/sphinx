import json
import re
import sys
from collections import namedtuple
from os import path

from docutils import nodes
from docutils.writers import Writer

from sphinx import addnodes
from sphinx.util import logging

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

# Below are defined a set of contructors for most important pandoc AST types.
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
Div = elt('Div', 2)  # TODO
Header = elt('Header', 3)
HorizontalRule = elt('HorizontalRule', 0)
LineBlock = elt('LineBlock', 1)
MetaInlines = elt('MetaInlines', 1)
Null = elt('Null', 0)  # TODO
OrderedList = elt('OrderedList', 2)
Para = elt('Para', 1)
Plain = elt('Plain', 1)
RawBlock = elt('RawBlock', 2)  # TODO
Table = elt('Table', 5)  # TODO

# Inline elements
# Strikeout = elt('Strikeout', 1)  # unsupported
Cite = elt('Cite', 2)
NormalCitation = elt('NormalCitation', 0)
Code = elt('Code', 2)
DisplayMath = elt('DisplayMath', 0)
Emph = elt('Emph', 1)
Image = elt('Image', 3)
InlineMath = elt('InlineMath', 0)
LineBreak = elt('LineBreak', 0)  # TODO
Link = elt('Link', 3)  # TODO (partial)
Math = elt('Math', 2)
Note = elt('Note', 1)
Quoted = elt('Quoted', 2)  # TODO
RawInline = elt('RawInline', 2)  # TODO
SmallCaps = elt('SmallCaps', 1)  # TODO
SoftBreak = elt('SoftBreak', 0)  # TODO
Space = elt('Space', 0)
Span = elt('Span', 2)  # TODO (partial)
Str = elt('Str', 1)
Strong = elt('Strong', 1)
Subscript = elt('Subscript', 1)
Superscript = elt('Superscript', 1)

LineBlockLine = namedtuple('LineBlockLine', 'contents')


class TableBuilder:
    """A builder for pandoc tables

    .. important::

        Unfortunately, pandoc tables are not expressive enough to support
        all ReST tables (see https://github.com/jgm/pandoc/issues/1024).
        In cases where input table cannot be converted, we try a "best looking"
        conversion (yeah, that's quite subjective) and report a warning.
    """
    def __init__(self, node):
        # type: (nodes.table) -> None
        self.headers = []                        # type: List[unicode]
        self.rows = []
        self.node = node
        self.colcount = 0
        self.colwidths = []                     # type: List[int]

        self.in_header = False
        self.currow = []
        self.row = 0
        self.col = 0

    def start_thead(self):
        self.in_header = True

    def leave_thead(self):
        assert self.in_header == True
        self.in_header = False
        self.headers = self.rows
        self.rows = []
        self.row = 0

    def add_colspec(self, width):
        self.colwidths.append(width)
        self.colcount += 1

    def add_header(self, content):
        self.header.append(content)

    def next_row(self):
        self.row += 1
        self.rows.append(self.currow)
        self.currow = []
        self.col = 0

    def add_cell(self, content, morecols):
        if morecols > 0:
            logger.warning(
                "pandoc doesn't support multicolumn cell, filling with empty "
                "cell instead.",
                location=self.node)
        self.currow.extend([content] + [[] for _ in range(morecols)])
        self.col += 1 + morecols

    def as_pandoc_ast(self):
        if len(self.headers) > 1:
            logger.warning("pandoc doesn't support more than one row in a table"
                           " header", location=self.node)
        headers = self.headers[0] if len(self.headers) > 0 else []
        caption = []
        column_align = [AlignDefault() for _ in range(self.colcount)]
        total_width = sum(self.colwidths)
        relative_widths = [float(x) / total_width for x in self.colwidths]
        return Table(caption, column_align, relative_widths,
                     headers,
                     self.rows)


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
        output = {'blocks': visitor.body,
                  'pandoc-api-version': [1, 17, 3],
                  'meta': meta}
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
    return Div(["", [name], []],
               [_div(["admonition-title"], [Para([Str(title)])], style=title_style),
                _div(["adminition-title"], contents, style=style)])


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

        self.body_stack = []
        self.body = []
        self.title = None
        self.caption = None
        self.legend = None
        self.table = None

        self.curfilestack = []
        self.footnotestack = []
        self.pending_footnotes = []
        self.hlsettingstack = \
            2 * [[builder.config.highlight_language, sys.maxsize]]
        self.next_section_ids = set()
        self.next_figure_ids = set()
        self.handled_abbrs = set()

    def _skip(self, node):
        raise nodes.SkipNode

    def _pass(self, node):
        pass

    def _push(self, node):
        self.push()

    def _pop_flat(self, node):
        contents = self.pop()
        self.body.append(contents)

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
        logger.debug("VISIT %s", node.tagname)
        return super().dispatch_visit(node)

    def dispatch_departure(self, node):
        logger.debug("DEPART %s", node.tagname)
        return super().dispatch_departure(node)

    def unknown_visit(self, node):
        logger.warning("not implemented: '%s'", node.tagname)
        raise nodes.SkipNode

    def hypertarget(self, id, withdoc=True):
        if withdoc:
            id = self.curfilestack[-1] + ':' + id
        return id

    def is_inline(self, node):
        return isinstance(node.parent, nodes.TextElement)

    def get_text(self, text):
        tokens = []

        def matcher(m):
            if m.group('white'):
                token = Space()
            else:
                token = Str(m.group('nonwhite'))
            tokens.append(token)

        RE_TOKENS.sub(matcher, text)
        return tokens

    def hyperlink(self, id, contents):
        return Link(["", [], []], contents, [id, ""])

    def collect_footnotes(self, node):
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

    # already handled
    visit_substitution_definition = _skip

    def visit_Text(self, node):
        self.body.extend(self.get_text(node.astext()))
        # glue space and footnote reference
        if (self.body
                and self.body[-1]["t"] == "Space"
                and isinstance(node.next_node(siblings=True),
                               nodes.footnote_reference)):
            self.body.pop()
        raise nodes.SkipNode

    def visit_section(self, node):
        self.in_section += 1

    def depart_section(self, node):
        self.in_section = max(self.in_section - 1, self.top_in_section - 1)

    def visit_title(self, node):
        if isinstance(node.parent, nodes.Admonition):
            # Admonition title is handled in depart_admonition
            raise nodes.SkipNode
        self.push()

    def depart_title(self, node):
        contents = self.pop()
        if isinstance(node.parent, nodes.table):
            return
        if isinstance(node.parent, nodes.topic):
            self.body.append(Para([Span(["", ["topic-title"], []], contents)]))
            return
        if isinstance(node.parent, nodes.section):
            if self.title is None:
                self.title = MetaInlines(contents)
                self.in_section = 0
                return
            assert self.in_section > 0
            id = ""
            try:
                id = list(self.next_section_ids)[0]
            except IndexError:
                pass
            self.next_section_ids.clear()
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

    visit_definition_list_item = _push
    depart_definition_list_item = _pop_flat

    def visit_term(self, node):
        for n in node[:]:
            if isinstance(n, (addnodes.index, nodes.target)):
                n.walkabout(self)
                node.remove(n)
        self.push()

    def depart_term(self, node):
        contents = self.pop()
        if node.get('ids'):
            # glossary term, wrap with a Span with the right id
            id = self.hypertarget(node['ids'][0])
            contents = [Span([id, [], []], contents)]
        self.body.append(contents)

    # TODO?
    visit_index = _skip

    # TODO: what is this?
    visit_classifier = depart_classifier = _pass

    visit_container = _push

    def depart_container(self, node):
        contents = self.pop()
        if self.caption:
            contents.insert(0, Para([Span(["", ["caption"], []], self.caption)]))
        self.body.append(Div(["", ["container"], []], contents))
        self.caption = None

    visit_definition = _push

    def depart_definition(self, node):
        contents = self.pop()
        self.body.append([contents])  # singleton list mandatory

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
                next = node.parent.parent[
                    node.parent.parent.index(node.parent)]

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
            OrderedList([start, {"t": style}, {"t": delim}], contents))

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
                    yield from unfold(el, depth + 1)

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
            logger.warning("Failed to create reference target: no ids found",
                           location=node)
            return
        # Expected contents shape:
        #   contents[0] -> Str (label)
        #   contents[1] -> Para (label description)
        cite_para = [Span([anchor, ["citation-label"], []], [contents[0]]),
                     Space()] + contents[1]['c']
        self.body.append(Div(["citation", [], []], [Para(cite_para)]))

    def visit_literal(self, node):
        self.body.append(Code(["", [], []], node.astext()))
        raise nodes.SkipNode

    visit_block_quote = _push

    depart_block_quote = _pop_with(BlockQuote)

    def _convert_size(self, measure):
        """Convert a relative distance (e.g., 12%) to an absolute one

        Not all pandoc backends provide support for relative size (e.g., the
        docx backend does not). By default, no conversion is applied, the user
        needs to set the option pandoc_force_absolute_size = True to enable
        conversion. In such case, the maximum size (i.e., the 100% value) must
        also be provided ('textwidth' attribute in pandoc_options config
        values).
        """
        if not self.builder.config.pandoc_force_absolute_size:
            return measure

        if not measure.endswith('%'):
            return measure

        textwidth = self.builder.config.pandoc_options.get('textwidth', None)
        if not textwidth:
            logger.warning(
                "using options 'pandoc_force_absolute_size' requires textwidth"
                "to be set in pandoc_options. For instance, define\n"
                "pandoc_options = {'textwidth': (16, 'cm')} in your conf.py")
            return measure
        width, unit = textwidth
        f = float(measure[:-1])
        return str(f * width / 100) + unit

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

        if uri.endswith('.svg'):
            uri = path.splitext(uri)[0] + '.png'
        self.body.append(Para([Image(["", [], attrs], [alt], [uri, ""])]))
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
    # visit_todo_node = _push

    depart_note = _admonition("note", "Note", style="Note")
    depart_important = _admonition("important", "Important", style="Warning")
    depart_warning = _admonition("warning", "Warning", style="Warning")
    depart_tip = _admonition("tip", "Tip", style="Tiplo")
    depart_seealso = _admonition("seealso", "See Also", style="Tip")
    # depart_todo_node = _admonition("todo", "Todo", style="Warning")

    visit_admonition = _push

    def depart_admonition(self, node):
        contents = self.pop()
        title = node.children[0].astext()
        name = node['classes'][0]
        self.body.append(_admonition_contents(name, title, contents))

    def visit_table(self, node):
        self.table = TableBuilder(node)

    def depart_table(self, node):
        def plain_str(x):
            return Plain([Str(x)])
        self.body.append(self.table.as_pandoc_ast())
        # self.body.append(
        #     Table(
        #         [],
        #         [AlignCenter(), AlignLeft()],
        #         [0.15, 0.16],
        #         [[plain_str("header0")], [plain_str("header1")]],
        #         [[[plain_str("col0")], [plain_str("col0")]],
        #          [[plain_str("col merged")], []]]
        #         ))
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
        self.table.add_cell(contents, node.get('morecols', 0))

    visit_figure = _push

    def _get_figure_number(self, node):
        """ Extract caption numbering

        :return:
            None if no numbering was found.
            When a configuration option is missing it also reports a warning.
            In case of success, returns the string prefix (e.g., "Fig 42. ")
        """
        fig_type = self.builder.env.domains['std'].get_figtype(node)
        if not fig_type:
            logger.warning("figure type {} is unknown".format(fig_type),
                           location=node)
            return None

        if len(node['ids']) == 0:
            logger.warning("No ids assigned for {}".format(node.tagname),
                           location=node)
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
        id = ""
        if self.next_figure_ids:
            id = list(self.next_figure_ids)[0]
            self.next_figure_ids.clear()
        self.body.append(Div(
            [id, ["figure"], []],
            [Para([image])] + (self.legend or [])))
        self.caption = None
        self.legend = None

    visit_legend = _push

    def depart_legend(self, node):
        self.legend = self.pop()

    visit_caption = _push

    def depart_caption(self, node):
        fig_number = self._get_figure_number(node.parent)
        prefix = [Str(fig_number)] if fig_number else []
        self.caption = prefix + self.pop()

    visit_glossary = _push

    def depart_glossary(self, node):
        contents = self.pop()
        self.body.append(contents[0])

    visit_topic = _push

    depart_topic = _div_wrap("topic")

    visit_centered = _push

    depart_centered = _div_wrap("centered")

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
