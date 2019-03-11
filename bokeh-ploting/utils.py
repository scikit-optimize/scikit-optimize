import numpy as np
import pickle
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Button
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from ProcessOptimizer.benchmarks import branin as branin
from ProcessOptimizer.benchmarks import hart6 as hart6_
from ProcessOptimizer.plots import plot_objective
from ProcessOptimizer import gp_minimize, forest_minimize, dummy_minimize
from ProcessOptimizer import plots
from ProcessOptimizer.plots import dependence
import matplotlib.pyplot as plt
#from bokeh.models import Toggle
from bokeh.models.widgets import Toggle, CheckboxButtonGroup, Div, PreText, Slider
from bokeh.models.glyphs import Text

# Create a button group with n buttons = max_pars
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Button
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from ProcessOptimizer.benchmarks import branin as branin
from ProcessOptimizer.benchmarks import hart6 as hart6_
from ProcessOptimizer.plots import plot_objective
from ProcessOptimizer import gp_minimize, forest_minimize, dummy_minimize
from ProcessOptimizer import plots
from ProcessOptimizer.plots import dependence
import matplotlib.pyplot as plt
#from bokeh.models import Toggle
from bokeh.models.widgets import Toggle, CheckboxButtonGroup, Div, PreText, Slider

def get_index_of_child(id,layout):
    ''' Returns the index of the child with the corresponding id. Returns None if no id was found
    Arguments:
        str id
    Outputs
        int ind
    '''
    assert isinstance(id, str), 'input argument ´id´ must be a string'
    ind = [i for i in range(len(layout)) if layout[i].id == id]
    assert not len(ind) == 0, 'Could not find index ' + id
    assert not len(ind) > 1, 'Expected to only find one id. Found ' + str(len(ind)) + ' for id ' + id
    if len(ind) == 1:
        return ind[0]
    elif len(ind) == 0:
        raise ValueError('Could not find index ' + id)
    elif len(ind)> 1:
        raise ValueError('Could not find index ' + id)

def get_plot_layout(layout,result,active_list,n_points):
    print('yo')
    # return a column of rows of plots
    plots=[]
    # if no parameters have been selected
    if not active_list:
        div = Div(text="""<font size="6">No parameters selected</font>""",
width=500, height=100)
        #t = PreText(text="""No parameters selected""", width=500, height=100,text_font_size = 30)
        return div
    space = result.space
    model = result.models[-1]
    for i_list in range(len(active_list)): #only evaluate the paramets that have
        #been selected
        plots.append([])
        for j_list in range(len(active_list)): #we only evaluate the lower left half of the grid
            #if i in active_pars or j in active_pars:
                #break
            i =active_list[i_list]
            j = active_list[j_list]
            if j>i:
                break
            elif i==j: #diagonal
                xi,yi = dependence(space, model, i, j=None, sample_points=None,
                       n_samples=50, n_points=40, x_eval = None)
                #x_range = [-1,1]
                #y_range = [0,min(1,np.max(y))]
                source = ColumnDataSource(data=dict(x=xi, y=yi))
                plot = figure(plot_height=200, plot_width=200, title=str(i)+str(j),
                tools = '',
                x_range=[-1,1],y_range=[0, 10])
                plot.toolbar.logo = None #remove the bokeh logo fom figures
                plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
            else: #contour plot
                xi,yi,zi = dependence(space, model, i, j=j, sample_points=None,
                       n_samples=50, n_points=40, x_eval = None)
                #y = np.random.rand(100,100)
                #print(y)
                plot = figure(plot_height=200, plot_width=200,x_range=(-1, 1), y_range=(-1, 1),
                tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])

                # must give a vector of image data for image parameter
                plot.image(image=[zi], x=-1, y=-1, dw=2, dh=2, palette="Spectral11")

                
                #plot.circle(x='x', y='y', source=source)
                #plot.circle(x='x',y='y',source=source_red, size=10, color="red", alpha=0.5)
            plots[i_list].append(plot)
    # Now plots is a list of rows of plots
    # We now convert these to a layout
    # Here we create a list of row objects, where each row object contains a list of plot objects
    rows = []
    for i in range(len(plots)):
        rows.append(row(*plots[i]))
    # Here we create a column object with all the rows
    plot_layout = column(*rows)
    return plot_layout
