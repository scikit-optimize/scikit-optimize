## Define mini-templates for each portion of the doco.

<%!
  import re

  def indent(s, spaces=4):
      """
      Inserts `spaces` after each string of new lines in `s`
      and before the start of the string.
      """
      new = re.sub('(\n+)', '\\1%s' % (' ' * spaces), s)
      return (' ' * spaces) + new.strip()

  def docstring(d):
      if len(d.docstring) == 0 and hasattr(d, 'inherits'):
          return d.inherits.docstring
      else:
          return d.docstring
%>

<%def name="function(func)" filter="trim">
${func.name}(${func.spec()})
${docstring(func) | indent}
</%def>

<%def name="variable(var)" filter="trim">
${var.name}
${docstring(var) | indent}
</%def>

<%def name="class_(cls)" filter="trim">
${cls.name} \
% if len(cls.docstring) > 0:

${cls.docstring | indent}
% endif
<%
  class_vars = cls.class_variables()
  static_methods = cls.functions()
  inst_vars = cls.instance_variables()
  methods = cls.methods()
  mro = cls.module.mro(cls)
  descendents = cls.module.descendents(cls)
%>
% if len(mro) > 0:
    Ancestors (in MRO)
    ------------------
    % for c in mro:
    ${c.refname}
    % endfor

% endif
% if len(descendents) > 0:
    Descendents
    -----------
    % for c in descendents:
    ${c.refname}
    % endfor

% endif
% if len(class_vars) > 0:
    Class variables
    ---------------
    % for v in class_vars:
${capture(variable, v) | indent}

    % endfor
% endif
% if len(static_methods) > 0:
    Static methods
    --------------
    % for f in static_methods:
${capture(function, f) | indent}

    % endfor
% endif
% if len(inst_vars) > 0:
    Instance variables
    ------------------
    % for v in inst_vars:
${capture(variable, v) | indent}

    % endfor
% endif
% if len(methods) > 0:
    Methods
    -------
    % for m in methods:
${capture(function, m) | indent}

    % endfor
% endif
</%def>

## Start the output logic for an entire module.

<%
  variables = module.variables()
  classes = module.classes()
  functions = module.functions()
  submodules = module.submodules()
%>

Module ${module.name}
-------${'-' * len(module.name)}
% if not module._filtering:
${module.docstring}
% endif


% if len(variables) > 0:
Variables
---------
    % for v in variables:
${variable(v)}

    % endfor
% endif


% if len(functions) > 0:
Functions
---------
    % for f in functions:
${function(f)}

    % endfor
% endif


% if len(classes) > 0:
Classes
-------
    % for c in classes:
${class_(c)}

    % endfor
% endif


% if len(submodules) > 0:
Sub-modules
-----------
    % for m in submodules:
    ${m.name}
    % endfor
% endif
