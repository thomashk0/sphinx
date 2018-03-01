import subprocess
from docutils.frontend import OptionParser
from docutils.io import FileOutput
from os import path

from sphinx import addnodes
from sphinx.builders import Builder
from sphinx.environment import NoUri
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.util import logging, ensuredir, status_iterator, copy_asset_file
from sphinx.util.nodes import inline_all_toctrees
from sphinx.util.console import darkgreen
from sphinx.util.osutil import SEP
from sphinx.writers.pandoc import PandocTranslator, PandocWriter

logger = logging.getLogger(__name__)

# TODO: cleanup, test, test multiple document merge


class PandocBuilder(Builder):
    name = 'pandoc'
    format = 'pandoc'
    supported_image_types = ['image/png', 'image/jpeg']
    supported_remote_images = True
    default_translator_class = PandocTranslator

    def init(self):
        self.docnames = []
        self.document_data = []

    def get_outdated_docs(self):
        return 'all documents'  # for now

    def get_target_uri(self, docname, typ=None):
        if docname not in self.docnames:
            raise NoUri
        else:
            return '%' + docname

    def get_relative_uri(self, from_, to, typ=None):
        # ignore source path
        return self.get_target_uri(to, typ)

    def init_document_data(self):
        preliminary_document_data = [list(x) for x in self.config.pandoc_documents]
        if not preliminary_document_data:
            logger.warning(
                'no "pandoc_documents" config value found; no documents '
                'will be written')
            return
        self.titles = []
        for entry in preliminary_document_data:
            docname = entry[0]
            if docname not in self.env.all_docs:
                continue
            self.document_data.append(entry)
            if docname.endswith(SEP + 'index'):
                docname = docname[:-5]
            self.titles.append((docname, entry[1]))

    def write(self, *ignored):
        docwriter = PandocWriter(self)
        docsettings = OptionParser(
            defaults=self.env.settings,
            components=(docwriter,),
            read_config_files=True).get_default_values()

        self.init_document_data()

        for docname, title, author in self.document_data:
            logger.info("processing %s...", docname)
            doctree = self.assemble_doctree(docname)
            self.post_process_images(doctree)
            doctree.settings = docsettings
            doctree.settings.author = author
            doctree.settings.title = title
            doctree.settings.docname = docname

            logger.info("writing...")
            filename = path.join(self.outdir, docname + '.json')
            ensuredir(path.dirname(filename))
            destination = FileOutput(destination_path=filename, encoding='utf-8')
            docwriter.write(doctree, destination)
            logger.info("done")

    def assemble_doctree(self, indexfile):
        self.docnames = {indexfile}
        logger.info(darkgreen(indexfile) + " ", nonl=1)
        tree = self.env.get_doctree(indexfile)
        tree['docname'] = indexfile
        largetree = inline_all_toctrees(self, self.docnames, indexfile, tree,
                                        darkgreen, [indexfile])
        largetree['docname'] = indexfile
        logger.info("")
        logger.info("resolving references...")
        self.env.resolve_references(largetree, indexfile, self)
        # TODO: pending nodes?
        for pendingnode in largetree.traverse(addnodes.pending_xref):
            pendingnode.replace_self(pendingnode.children)

        return largetree

    def finish(self):
        # type: () -> None
        self.copy_image_files()

    def copy_image_files(self):
        # type: () -> None
        if self.images:
            stringify_func = ImageAdapter(self.app.env).get_original_image_uri
            for src in status_iterator(self.images, 'copying images... ',
                                       "brown",
                                       len(self.images),
                                       self.app.verbosity,
                                       stringify_func=stringify_func):
                dest = self.images[src]

                # SVG to PNG
                if src.endswith('.svg'):
                    dest = path.splitext(dest)[0] + '.png'
                    logger.info("converting %s to %s", src, dest)
                    subprocess.check_call(
                        ['inkscape', '-w', '2048',
                         '-e', path.join(self.outdir, dest),
                         path.join(self.srcdir, src)],
                        stdout=subprocess.DEVNULL)
                    continue
                try:
                    copy_asset_file(path.join(self.srcdir, src),
                                    path.join(self.outdir, dest))
                except Exception as err:
                    logger.warning('cannot copy image file %r: %s',
                                   path.join(self.srcdir, src), err)


def setup(app):
    app.add_builder(PandocBuilder)
    app.add_config_value('pandoc_documents',
                         lambda self: [(self.master_doc, self.project, 'AUTHOR')],
                         None)
    return {
        'version': 'builtin',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
