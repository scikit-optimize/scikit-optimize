<%
  import re
  import sys
  import inspect

  import markdown
  try:
    import pygments
    import pygments.formatters
    import pygments.lexers
    use_pygments = True
  except ImportError:
    use_pygments = False

  import pdoc

  root_url = "http://scikit-optimize.github.io/"

  # From language reference, but adds '.' to allow fully qualified names.
  pyident = re.compile('^[a-zA-Z_][a-zA-Z0-9_.]+$')
  indent = re.compile('^\s*')

  # Whether we're showing the module list or a single module.
  module_list = 'modules' in context.keys()

  def decode(s):
    if sys.version_info[0] < 3 and isinstance(s, str):
      return s.decode('utf-8', 'ignore')
    return s

  def ident(s):
    return '<span class="ident">%s</span>' % s

  def sourceid(dobj):
    return 'source-%s' % dobj.refname

  def clean_source_lines(lines):
    """
    Cleans the source code so that pygments can render it well.

    Returns one string with all of the source code.
    """
    base_indent = len(indent.match(lines[0]).group(0))
    base_indent = 0
    for line in lines:
      if len(line.strip()) > 0:
        base_indent = len(indent.match(lines[0]).group(0))
        break
    lines = [line[base_indent:] for line in lines]
    if not use_pygments:  # :-(
      return '<pre><code>%s</code></pre>' % (''.join(lines))

    pylex = pygments.lexers.PythonLexer()
    htmlform = pygments.formatters.HtmlFormatter(cssclass='codehilite')
    return pygments.highlight(''.join(lines), pylex, htmlform)

  def linkify(match):
    matched = match.group(0)
    ident = matched[1:-1]
    name, url = lookup(ident)
    if name is None:
      return matched
    return '[`%s`](%s)' % (name, url)

  def mark(s, linky=True):
    if linky:
      s, _ = re.subn('\b\n\b', ' ', s)
    if not module_list:
      s, _ = re.subn('`[^`]+`', linkify, s)


    extensions = []
    if use_pygments:
      extensions = ['markdown.extensions.codehilite(linenums=False)',
                    'markdown.extensions.fenced_code']
    s = markdown.markdown(s.strip(), extensions=extensions)
    return s

  def glimpse(s, length=100):
    if len(s) < length:
      return s
    return s[0:length] + '...'

  def module_url(m):
    """
    Returns a URL for `m`, which must be an instance of `Module`.
    Also, `m` must be a submodule of the module being documented.

    Namely, '.' import separators are replaced with '/' URL
    separators. Also, packages are translated as directories
    containing `index.html` corresponding to the `__init__` module,
    while modules are translated as regular HTML files with an
    `.m.html` suffix. (Given default values of
    `pdoc.html_module_suffix` and `pdoc.html_package_name`.)
    """
    if module.name == m.name:
      return ''

    if len(link_prefix) > 0:
      base = m.name
    else:
      base = m.name[len(module.name)+1:]
    url = base.replace('.', '/')
    if m.is_package():
      url += '/%s' % pdoc.html_package_name
    else:
      url += pdoc.html_module_suffix
    return link_prefix + url

  def external_url(refname):
    """
    Attempts to guess an absolute URL for the external identifier
    given.

    Note that this just returns the refname with an ".ext" suffix.
    It will be up to whatever is interpreting the URLs to map it
    to an appropriate documentation page.
    """
    return '/%s.ext' % refname

  def is_external_linkable(name):
    return external_links and pyident.match(name) and '.' in name

  def lookup(refname):
    """
    Given a fully qualified identifier name, return its refname
    with respect to the current module and a value for a `href`
    attribute. If `refname` is not in the public interface of
    this module or its submodules, then `None` is returned for
    both return values. (Unless this module has enabled external
    linking.)

    In particular, this takes into account sub-modules and external
    identifiers. If `refname` is in the public API of the current
    module, then a local anchor link is given. If `refname` is in the
    public API of a sub-module, then a link to a different page with
    the appropriate anchor is given. Otherwise, `refname` is
    considered external and no link is used.
    """
    d = module.find_ident(refname)
    if isinstance(d, pdoc.External):
      if is_external_linkable(refname):
        return d.refname, external_url(d.refname)
      else:
        return None, None
    if isinstance(d, pdoc.Module):
      return d.refname, module_url(d)
    if module.is_public(d.refname):
      return d.name, '#%s' % d.refname
    return d.refname, '%s#%s' % (module_url(d.module), d.refname)

  def link(refname):
    """
    A convenience wrapper around `href` to produce the full
    `a` tag if `refname` is found. Otherwise, plain text of
    `refname` is returned.
    """
    name, url = lookup(refname)
    if name is None:
      return refname

    if notebook:
        url = "../" + url

    return '<a href="%s">%s</a>' % (url, name)
%>
<%def name="show_source(d)">
  % if show_source_code and d.source is not None and len(d.source) > 0:
  <p class="source_link"><a href="javascript:void(0);" onclick="toggle('${sourceid(d)}', this);">Show source &equiv;</a></p>
  <div id="${sourceid(d)}" class="source">
    ${decode(clean_source_lines(d.source))}
  </div>
  % endif
</%def>

<%def name="show_desc(d, limit=None)">
  <%
  inherits = (hasattr(d, 'inherits')
           and (len(d.docstring) == 0
            or d.docstring == d.inherits.docstring))
  docstring = (d.inherits.docstring if inherits else d.docstring).strip()
  if limit is not None:
    docstring = glimpse(docstring, limit)
  %>
  % if len(docstring) > 0:
  % if inherits:
    <div class="desc inherited">${docstring | mark}</div>
  % else:
    <div class="desc">${docstring | mark}</div>
  % endif
  % endif
  % if not isinstance(d, pdoc.Module):
  <div class="source_cont">${show_source(d)}</div>
  % endif
</%def>

<%def name="show_inheritance(d)">
  % if hasattr(d, 'inherits'):
    <p class="inheritance">
     <strong>Inheritance:</strong>
     % if hasattr(d.inherits, 'cls'):
       <code>${link(d.inherits.cls.refname)}</code>.<code>${link(d.inherits.refname)}</code>
     % else:
       <code>${link(d.inherits.refname)}</code>
     % endif
    </p>
  % endif
</%def>

<%def name="show_module_list(modules)">
<h1>Python module list</h1>

% if len(modules) == 0:
  <p>No modules found.</p>
% else:
  <table id="module-list">
  % for name, desc in modules:
    <tr>
      <td><a href="${link_prefix}${name}">${name}</a></td>
      <td>
      % if len(desc.strip()) > 0:
        <div class="desc">${desc | mark}</div>
      % endif
      </td>
    </tr>
  % endfor
  </table>
% endif
</%def>

<%def name="show_column_list(items, numcols=3)">
  <ul>
  % for item in items:
    <li class="mono">${item}</li>
  % endfor
  </ul>
</%def>

<%def name="show_module(module)">
  <%
  variables = module.variables()
  classes = module.classes()
  functions = module.functions()
  submodules = module.submodules()
  %>

  <%def name="show_func(f)">
  <div class="item">
    <div class="name def" id="${f.refname}">
    <p>def ${ident(f.name)}(</p><p>${f.spec() | h})</p>
    </div>
    ${show_inheritance(f)}
    ${show_desc(f)}
  </div>
  </%def>

  % if 'http_server' in context.keys() and http_server:
    <p id="nav">
      <a href="/">All packages</a>
      <% parts = module.name.split('.')[:-1] %>
      % for i, m in enumerate(parts):
        <% parent = '.'.join(parts[:i+1]) %>
        :: <a href="/${parent.replace('.', '/')}">${parent}</a>
      % endfor
    </p>
  % endif

  <header id="section-intro">
  <h1 class="title"><span class="name">${module.name}</span> module</h1>
  ${module.docstring | mark}
  ${show_source(module)}
  </header>



  <section id="section-items">
    % if len(variables) > 0:
    <h2 class="section-title" id="header-variables">Module variables</h2>
    % for v in variables:
      <div class="item">
      <p id="${v.refname}" class="name">var ${ident(v.name)}</p>
      ${show_desc(v)}
      </div>
    % endfor
    % endif

    % if len(functions) > 0:
    <h2 class="section-title" id="header-functions">Functions</h2>
    % for f in functions:
      ${show_func(f)}
    % endfor
    % endif

    % if len(classes) > 0:
    <h2 class="section-title" id="header-classes">Classes</h2>
    % for c in classes:
      <%
      class_vars = c.class_variables()
      smethods = c.functions()
      smethods = [f for f in smethods if inspect.getmodule(f.func).__name__.startswith("skopt")]
      inst_vars = c.instance_variables()
      methods = c.methods()
      mro = c.module.mro(c)
      %>
      <div class="item">
      <p id="${c.refname}" class="name">class ${ident(c.name)}</p>
      ${show_desc(c)}

      <div class="class">
        % if len(mro) > 0:
          <h3>Ancestors (in MRO)</h3>
          <ul class="class_list">
          % for cls in mro:
          <li>${link(cls.refname)}</li>
          % endfor
          </ul>
        % endif
        % if len(class_vars) > 0:
          <h3>Class variables</h3>
          % for v in class_vars:
            <div class="item">
            <p id="${v.refname}" class="name">var ${ident(v.name)}</p>
            ${show_inheritance(v)}
            ${show_desc(v)}
            </div>
          % endfor
        % endif
        % if len(smethods) > 0:
          <h3>Static methods</h3>
          % for f in smethods:
            ${show_func(f)}
          % endfor
        % endif
        % if len(inst_vars) > 0:
          <h3>Instance variables</h3>
          % for v in inst_vars:
            <div class="item">
            <p id="${v.refname}" class="name">var ${ident(v.name)}</p>
            ${show_inheritance(v)}
            ${show_desc(v)}
            </div>
          % endfor
        % endif
        % if len(methods) > 0:
          <h3>Methods</h3>
          % for f in methods:
            ${show_func(f)}
          % endfor
        % endif
      </div>
      </div>
    % endfor
    % endif
  </section>
</%def>

<%def name="module_index(module)">
  <%
  variables = module.variables()
  classes = module.classes()
  functions = module.functions()
  submodules = module.submodules()
  %>
  <div id="sidebar">
    <ul id="index">
    <li class="set"><h3><a href="${ root_url }">Index</a></h3></li>

    % if len(variables) > 0:
    <li class="set"><h3><a href="#header-variables">Module variables</a></h3>
      ${show_column_list(map(lambda v: link(v.refname), variables))}
    </li>
    % endif

    % if len(functions) > 0:
    <li class="set"><h3><a href="#header-functions">Functions</a></h3>
      ${show_column_list(map(lambda f: link(f.refname), functions))}
    </li>
    % endif

    % if len(classes) > 0:
    <li class="set"><h3><a href="#header-classes">Classes</a></h3>
      <ul>
      % for c in classes:
        <li class="mono">
        <span class="class_name">${link(c.refname)}</span>
        </li>
      % endfor
      </ul>
    </li>
    % endif

    % if len(submodules) > 0:
    <li class="set"><h3><a href="#header-submodules">Sub-modules</a></h3>
      <ul>
      % for m in submodules:
        <li class="mono">${link(m.refname)}</li>
      % endfor
      </ul>
    </li>
    % endif

    % if len(submodules) == 0:
    <li class="set"><h3><a href="${ root_url }"></a></h3>
    </li>
    % endif

    % if len(all_notebooks) > 0:
    <li class="set"><h3><a href="#">Notebooks</a></h3>
      <ul>
      % for notebook in all_notebooks:
        <%
        filename = notebook.rsplit(sep="/", maxsplit=1)[-1][:-3]
        nbname = filename.replace("-", " ").capitalize()
        %>
        <li><a href="${ root_url }notebooks/${ filename }.html">${ nbname }</a></li>
      % endfor
      </ul>
    </li>
    % endif
    </ul>
  </div>
</%def>

<!doctype html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />

  % if module_list:
    <title>Python module list</title>
    <meta name="description" content="A list of Python modules in sys.path" />
  % else:
    <title>${module.name} API documentation</title>
    <meta name="description" content="${module.docstring | glimpse, trim}" />
  % endif

  <link href='http://fonts.googleapis.com/css?family=Source+Sans+Pro:400,300' rel='stylesheet' type='text/css'>
  <%namespace name="css" file="css.mako" />
  <style type="text/css">
  ${css.pre()}
  </style>

  <style type="text/css">
  ${css.pdoc()}
  </style>

  % if use_pygments:
  <style type="text/css">
  ${pygments.formatters.HtmlFormatter().get_style_defs('.codehilite')}
  </style>
  % endif

  <style type="text/css">
  ${css.post()}
  </style>

  <script type="text/javascript">
  function toggle(id, $link) {
    $node = document.getElementById(id);
    if (!$node)
    return;
    if (!$node.style.display || $node.style.display == 'none') {
    $node.style.display = 'block';
    $link.innerHTML = 'Hide source &nequiv;';
    } else {
    $node.style.display = 'none';
    $link.innerHTML = 'Show source &equiv;';
    }
  }
  </script>
  <script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
  </script>
  <script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
  });
  </script>
</head>
<body>
<a href="https://github.com/scikit-optimize/scikit-optimize"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/652c5b9acfaddf3a9c326fa6bde407b87f7be0f4/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f6f72616e67655f6666373630302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png"></a>
<a href="#" id="top">Top</a>

<div id="container">
  % if module_list:
    <article id="content">
      ${show_module_list(modules)}
    </article>
  % else:
    ${module_index(module)}
    <article id="content">
      % if not notebook:
          ${show_module(module)}
      % else:
          <%
          content = open(notebook, "r").read()
          %>
          ${content | mark}
      % endif
    </article>
  % endif
  <div class="clear"> </div>
  <footer id="footer">
    <p>
      Documentation generated by
      <a href="https://github.com/BurntSushi/pdoc">pdoc ${pdoc.__version__}</a>
    </p>

    <p>Design by <a href="http://nadh.in">Kailash Nadh</a></p>
  </footer>
</div>
</body>
</html>
