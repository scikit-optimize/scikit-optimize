import numpy as np
import pickle
from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot, Spacer
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
from ProcessOptimizer.space import Integer, Categorical
# Create a button group with n buttons = max_pars
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, column, Spacer
from bokeh.models import ColumnDataSource, Button, Range1d, Span
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
from bokeh.models.widgets import Toggle, CheckboxButtonGroup, Div, PreText, Slider, Select
from ProcessOptimizer.plots import _map_categories

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

def get_plot_list(layout,result,active_list,n_points,source):
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
    bounds = result.space.bounds
    samples, minimum, iscat = _map_categories(space, result.x_iters, result.x)
    # While iterating throuhg each plot we keep track of the max and min values of y and z in case of 2d plots
    #  so we can adjust all axis later
    y_min = float("inf")
    y_max = -float("inf")
    z_min = float("inf")
    z_max = -float("inf")
    print('kan')
    source['red'] = []
    print('kan')
    for i_list in range(len(active_list)): #only evaluate the paramets that have
        #been selected
        print('bank')
        source['red'].append([])
        source['samples'].append([])
        print('bank')
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
                       n_samples=50, n_points=n_points, x_eval = None)
                if isinstance(space.dimensions[i],Categorical): #check if values are categorical
                    source_red = minimum[i]+0.5 # The Span class does not support categorical values. Using
                        # numerical values and adding 0.5 is a workaround
                    x_range = space.dimensions[i].categories
                    #convert integers to catogorical strings
                    xi = [bounds[i][ind] for ind in xi]
                else:
                    source_red = minimum[i]
                    x_range = [bounds[i][0],bounds[i][1]]
                print('diag 1')
                #x_range = [-1,1]
                #y_range = [0,min(1,np.max(y))]
                print('fad')
                source_temp = ColumnDataSource(data=dict(x=xi, y=yi))
                source_samples = [] # We dont plot samples on diagonal
                print('diag 2')
                plot = figure(plot_height=200, plot_width=200, title=str(i)+str(j),
                tools = '',
                x_range=x_range,y_range=[0,20])
                print('diag 3')
                plot.toolbar.logo = None #remove the bokeh logo fom figures
                plot.line('x', 'y', source=source_temp, line_width=3, line_alpha=0.6)
                # Add red line
                plot.add_layout(Span(location=source_red, dimension='height', line_color='red', line_width=3))
                print('diag 4')
                if np.min(yi) < y_min:
                    y_min = np.min(yi)
                if np.max(yi) > y_max:
                    y_max = np.max(yi)
                
                print('diag 5')
            else: #contour plot
                xi,yi,zi = dependence(space, model, i, j=j, sample_points=None,
                       n_samples=50, n_points=n_points, x_eval = None)
                if isinstance(space.dimensions[j],Categorical): #check if values are categorical
                    x_range = space.dimensions[j].categories
                    #convert integers to catogorical strings
                    xi = [bounds[j][ind] for ind in xi]
                    x_anchor = 0
                    x_span = len(result.space.dimensions[j].categories)
                else:
                    x_anchor = bounds[j][0]
                    x_span = bounds[j][1]-bounds[j][0]
                    x_range =[np.min(xi),np.max(xi)]
                if isinstance(space.dimensions[i],Categorical): #check if values are categorical
                    yi = [bounds[i][ind] for ind in yi]
                    y_range = space.dimensions[i].categories
                    y_anchor = 0
                    y_span = len(result.space.dimensions[i].categories)
                else:
                    y_anchor = bounds[i][0]
                    y_span = bounds[i][1]-bounds[i][0]
                    y_range =[np.min(yi),np.max(yi)]

                if np.min(zi) < z_min:
                    z_min = np.min(zi)
                if np.max(zi) > z_max:
                    z_max = np.max(zi)
                if i == 4 and j == 2:
                    print(x_range)
                #y = np.random.rand(100,100)
                plot = figure(plot_height=200, plot_width=200,x_range=x_range,y_range=y_range, tools = '')
                plot.toolbar.logo = None

                # must give a vector of image data for image parameter
                plot.image(image=[zi], x=x_anchor, y=y_anchor, dw=x_span, dh=y_span, palette="Spectral11")
                #red_source = result.x
                #source.red[i_list].append(red_source)
                x_samples = [val[j] for val in result.x_iters]
                print('fas')
                y_samples = [val[i] for val in result.x_iters]
                print(y_samples)
                source_samples = ColumnDataSource(data=dict(x = x_samples, y = y_samples))
                source_red = ColumnDataSource(data=dict(x = [result.x[j]], y = [result.x[i]]))
                print('fas')
                plot.circle(x='x',y='y',source=source_samples, size=10, color="black", alpha=0.5)
                plot.circle(x='x',y='y',source=source_red, size=10, color="red", alpha=1)
            if isinstance(space.dimensions[j],Categorical):
                plot.xaxis.major_label_orientation = 0.3
            if isinstance(space.dimensions[i],Categorical) and i != j:
                plot.yaxis.major_label_orientation = 1
                
                #plot.circle(x='x', y='y', source=source)
                #plot.circle(x='x',y='y',source=source_red, size=10, color="red", alpha=0.5)
            plots[i_list].append(plot)
            source['samples'][i_list].append(source_samples)
            source['red'][i_list].append(source_red)
    # Now plots is a list of rows of plots
    # here we set the range for all the diagonal plots
    print('hej')
    for i in range(len(plots)):
        plots[i][i].y_range = Range1d(y_min,y_max)
        
    return plots

def get_plots_layout(layout,result,active_list,n_points,source):
    plots = get_plot_list(layout,result,active_list,n_points,source)
    print('1')
    value_adjusters = get_value_adjusters_list(result,active_list)
    print('2')
    # We now convert the plots to a layout
    # Here we create a list of row objects, where each row object contains a list of plot objects
    if True:
        rows = []
        rows.append(row(value_adjusters[0]))
        print('3')
        for i in range(len(plots)): #we create one extra row because of the value adjusters
            print('4')
            if i == len(value_adjusters)-1:
                print('5')
                rows.append(row(*plots[i]))
                print('5')
            else:
                print('6') 
                rows.append(row(*plots[i],value_adjusters[i+1]))
                print('6')
    # Here we create a column object with all the rows
        print('7')
        plot_layout = column(*rows)
    else:
        grid = []
        grid.append(value_adjusters[0])
        print('3')
        for i in range(len(plots)): #we create one extra row because of the value adjusters
            print('4')
            if i == len(value_adjusters)-1:
                print('5')
                grid.append(plots[i])
                print('5')
            else:
                print('6') 
                grid.append([*plots[i],value_adjusters[i+1]])
                print('6')
    # Here we create a column object with all the rows
        plot_layout = gridplot(plots,merge_tools=True, 
                toolbar_options=dict(logo=None),plot_width=200, plot_height=200)
        print('7')

    return plot_layout

def get_value_adjusters_list(result,active_list):
    #returns a list of sliders for non-categorical values and groups of buttons for categorical values
    bounds = result.space.bounds
    if len(active_list):
        value_adjusters=[]
        for i in active_list:

            if isinstance(result.space.dimensions[i],Categorical):
                #create a buton group with categorical values
                select_row = row(Div(text='',width = 50))
                cats = list(result.space.dimensions[i].categories)
                select = Select(title="Value:", value=cats[0], options=cats,width = 100,height = 15)
                select_row.children.append(select)
                select_row.children.append(Div(text='',width = 50))
                select_col = column(Div(text='',height = 50))
                select_col.children.append(select_row)
                select_col.children.append(Div(text='',height = 50))
                value_adjusters.append(select_col)
            else:
                slider_row=row(Div(text='',width = 50))
                start = bounds[i][0]
                end = bounds[i][1]
                span = end-start
                value = start+span/2 
            #we create space on each side of slider
                slider  = Slider(start=start, end=end, value=value, step=.1,title="",width=150, height=30)
                slider_row.children.append(slider)
                slider_row.children.append(Div(text='',width = 50))
                slider_col = column(Div(text='',height = 50))
                slider_col.children.append(slider_row)
                slider_col.children.append(Div(text='',height = 50))
                value_adjusters.append(slider_col)
            #sliders.append(row(column(space,width = 1000),column(slider,width = 1000),column(space,width = 1000)))
        return value_adjusters#row(*sliders,width = 1000)
    else:
        div = Div(text="""<font size="2">No Sliders</font>""",
width=500, height=100)
        return div

#def plot_red(plot,values):
