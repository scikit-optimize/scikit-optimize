from __future__ import absolute_import, division, print_function
import argparse

import codecs
import datetime
import imp
import os
import os.path as path
import subprocess
import sys
import tempfile
import glob

import pdoc

# `xrange` is `range` with Python3.
try:
    xrange = xrange
except NameError:
    xrange = range

version_suffix = '%d.%d' % (sys.version_info[0], sys.version_info[1])
default_http_dir = path.join(tempfile.gettempdir(), 'pdoc-%s' % version_suffix)

parser = argparse.ArgumentParser(
    description='Automatically generate API docs for Python modules.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
aa = parser.add_argument
aa('module_name', type=str, nargs='?',
   help='The Python module name. This may be an import path resolvable in '
        'the current environment, or a file path to a Python module or '
        'package.')
aa('ident_name', type=str, nargs='?',
   help='When specified, only identifiers containing the name given '
        'will be shown in the output. Search is case sensitive. '
        'Has no effect when --http is set.')
aa('--version', action='store_true',
   help='Print the version of pdoc and exit.')
aa('--html', action='store_true',
   help='When set, the output will be HTML formatted.')
aa('--html-dir', type=str, default='.',
   help='The directory to output HTML files to. This option is ignored when '
        'outputting documentation as plain text.')
aa('--html-no-source', action='store_true',
   help='When set, source code will not be viewable in the generated HTML. '
        'This can speed up the time required to document large modules.')
aa('--overwrite', action='store_true',
   help='Overwrites any existing HTML files instead of producing an error.')
aa('--all-submodules', action='store_true',
   help='When set, every submodule will be included, regardless of whether '
        '__all__ is set and contains the submodule.')
aa('--external-links', action='store_true',
   help='When set, identifiers to external modules are turned into links. '
        'This is automatically set when using --http.')
aa('--template-dir', type=str, default=None,
   help='Specify a directory containing Mako templates. '
        'Alternatively, put your templates in $XDG_CONFIG_HOME/pdoc and '
        'pdoc will automatically find them.')
aa('--notebook-dir', type=str, default=None,
   help='Specify a directory containing Notebooks. ')
aa('--link-prefix', type=str, default='',
   help='A prefix to use for every link in the generated documentation. '
        'No link prefix results in all links being relative. '
        'Has no effect when combined with --http.')
aa('--only-pypath', action='store_true',
   help='When set, only modules in your PYTHONPATH will be documented.')
aa('--http', action='store_true',
   help='When set, pdoc will run as an HTTP server providing documentation '
        'of all installed modules. Only modules found in PYTHONPATH will be '
        'listed.')
aa('--http-dir', type=str, default=default_http_dir,
   help='The directory to cache HTML documentation when running as an HTTP '
        'server.')
aa('--http-host', type=str, default='localhost',
   help='The host on which to run the HTTP server.')
aa('--http-port', type=int, default=8080,
   help='The port on which to run the HTTP server.')
aa('--http-html', action='store_true',
   help='Internal use only. Do not set.')
args = parser.parse_args()


def quick_desc(imp, name, ispkg):
    if not hasattr(imp, 'path'):
        # See issue #7.
        return ''

    if ispkg:
        fp = path.join(imp.path, name, '__init__.py')
    else:
        fp = path.join(imp.path, '%s.py' % name)
    if os.path.isfile(fp):
        with codecs.open(fp, 'r', 'utf-8') as f:
            quotes = None
            doco = []
            for i, line in enumerate(f):
                if i == 0:
                    if len(line) >= 3 and line[0:3] in ("'''", '"""'):
                        quotes = line[0:3]
                        line = line[3:]
                    else:
                        break
                line = line.rstrip()
                if line.endswith(quotes):
                    doco.append(line[0:-3])
                    break
                else:
                    doco.append(line)
            desc = '\n'.join(doco)
            if len(desc) > 200:
                desc = desc[0:200] + '...'
            return desc
    return ''


def _eprint(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)


def last_modified(fp):
    try:
        return datetime.datetime.fromtimestamp(os.stat(fp).st_mtime)
    except:
        return datetime.datetime.min


def module_file(m):
    mbase = path.join(args.html_dir, *m.name.split('.'))
    if m.is_package():
        return path.join(mbase, pdoc.html_package_name)
    else:
        return '%s%s' % (mbase, pdoc.html_module_suffix)


def quit_if_exists(m):
    def check_file(f):
        if os.access(f, os.R_OK):
            _eprint('%s already exists. Delete it or run with --overwrite' % f)
            sys.exit(1)

    if args.overwrite:
        return
    f = module_file(m)
    check_file(f)

    # If this is a package, make sure the package directory doesn't exist
    # either.
    if m.is_package():
        check_file(path.dirname(f))


def html_out(m, html=True, all_notebooks=[]):
    f = module_file(m)
    if not html:
        f = module_file(m).replace(".html", ".md")
    dirpath = path.dirname(f)
    if not os.access(dirpath, os.R_OK):
        os.makedirs(dirpath)
    try:
        with codecs.open(f, 'w+', 'utf-8') as w:
            if not html:
                out = m.text()
            else:
                out = m.html(external_links=args.external_links,
                             link_prefix=args.link_prefix,
                             http_server=args.http_html,
                             source=not args.html_no_source,
                             notebook=None,
                             all_notebooks=all_notebooks)
            print(out, file=w)
    except Exception:
        try:
            os.unlink(f)
        except:
            pass
        raise
    for submodule in m.submodules():
        html_out(submodule, html, all_notebooks=all_notebooks)


def html_out_notebook(m, notebook, all_notebooks=[]):
    f = module_file(m)
    f = f.rsplit(sep="/", maxsplit=1)[0] + "/notebooks/" + notebook.rsplit(sep="/", maxsplit=1)[-1][:-3] + ".html"

    dirpath = path.dirname(f)
    if not os.access(dirpath, os.R_OK):
        os.makedirs(dirpath)
    try:
        with codecs.open(f, 'w+', 'utf-8') as w:
            out = m.html(external_links=args.external_links,
                         link_prefix=args.link_prefix,
                         http_server=args.http_html,
                         source=not args.html_no_source,
                         notebook=notebook,
                         all_notebooks=all_notebooks)
            print(out, file=w)
    except Exception:
        try:
            os.unlink(f)
        except:
            pass
        raise



def process_html_out(impath):
    # This unfortunate kludge is the only reasonable way I could think of
    # to support reloading of modules. It's just too difficult to get
    # modules to reload in the same process.

    cmd = [sys.executable,
           path.realpath(__file__),
           '--html',
           '--html-dir', args.html_dir,
           '--http-html',
           '--overwrite',
           '--link-prefix', args.link_prefix]
    if args.external_links:
        cmd.append('--external-links')
    if args.all_submodules:
        cmd.append('--all-submodules')
    if args.only_pypath:
        cmd.append('--only-pypath')
    if args.html_no_source:
        cmd.append('--html-no-source')
    if args.template_dir:
        cmd.append('--template-dir')
        cmd.append(args.template_dir)
    cmd.append(impath)

    # Can we make a good faith attempt to support 2.6?
    # YES WE CAN!
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0].strip().decode('utf-8')
    if p.returncode > 0:
        err = subprocess.CalledProcessError(p.returncode, cmd)
        err.output = out
        raise err
    if len(out) > 0:
        print(out)


if __name__ == '__main__':
    if args.version:
        print(pdoc.__version__)
        sys.exit(0)

    # We close stdin because some modules, upon import, are not very polite
    # and block on stdin.
    try:
        sys.stdin.close()
    except:
        pass

    if not args.http and args.module_name is None:
        _eprint('No module name specified.')
        sys.exit(1)
    if args.template_dir is not None:
        pdoc.tpl_lookup.directories.insert(0, args.template_dir)

    # If PYTHONPATH is set, let it override everything if we want it to.
    pypath = os.getenv('PYTHONPATH')
    if args.only_pypath and pypath is not None and len(pypath) > 0:
        pdoc.import_path = pypath.split(path.pathsep)

    docfilter = None
    if args.ident_name and len(args.ident_name.strip()) > 0:
        search = args.ident_name.strip()

        def docfilter(o):
            rname = o.refname
            if rname.find(search) > -1 or search.find(o.name) > -1:
                return True
            if isinstance(o, pdoc.Class):
                return search in o.doc or search in o.doc_init
            return False

    # Try to do a real import first. I think it's better to prefer
    # import paths over files. If a file is really necessary, then
    # specify the absolute path, which is guaranteed not to be a
    # Python import path.
    try:
        module = pdoc.import_module(args.module_name)
    except Exception as e:
        module = None

    # Get the module that we're documenting. Accommodate for import paths,
    # files and directories.
    if module is None:
        print(module)
        isdir = path.isdir(args.module_name)
        isfile = path.isfile(args.module_name)
        if isdir or isfile:
            fp = path.realpath(args.module_name)
            module_name = path.basename(fp)
            if isdir:
                fp = path.join(fp, '__init__.py')
            else:
                module_name, _ = path.splitext(module_name)

            # Use a special module name to avoid import conflicts.
            # It is hidden from view via the `Module` class.
            with open(fp) as f:
                module = imp.load_source('__pdoc_file_module__', fp, f)
                if isdir:
                    module.__path__ = [path.realpath(args.module_name)]
                module.__pdoc_module_name = module_name
        else:
            module = pdoc.import_module(args.module_name)
    module = pdoc.Module(module, docfilter=docfilter,
                         allsubmodules=args.all_submodules)

    # Plain text?
    if not args.html and not args.all_submodules:
        output = module.text()
        try:
            print(output)
        except IOError as e:
            # This seems to happen for long documentation.
            # This is obviously a hack. What's the real cause? Dunno.
            if e.errno == 32:
                pass
            else:
                raise e
        sys.exit(0)

    # Hook notebook generation
    all_notebooks = []

    if args.notebook_dir:
        all_notebooks = [f for f in sorted(glob.glob("%s/*.md" % args.notebook_dir))]

        for notebook in all_notebooks:
            html_out_notebook(module, notebook, all_notebooks=all_notebooks)

    # HTML output depends on whether the module being documented is a package
    # or not. If not, then output is written to {MODULE_NAME}.html in
    # `html-dir`. If it is a package, then a directory called {MODULE_NAME}
    # is created, and output is written to {MODULE_NAME}/index.html.
    # Submodules are written to {MODULE_NAME}/{MODULE_NAME}.m.html and
    # subpackages are written to {MODULE_NAME}/{MODULE_NAME}/index.html. And
    # so on... The same rules apply for `http_dir` when `pdoc` is run as an
    # HTTP server.
    if not args.http:
        quit_if_exists(module)
        html_out(module, args.html, all_notebooks=all_notebooks)
        sys.exit(0)
