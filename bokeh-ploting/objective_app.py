
import pickle
from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot, Spacer
from bokeh.models import ColumnDataSource, Button, Range1d, Span, CustomJS
from bokeh.models.widgets import Slider, TextInput,Toggle, CheckboxButtonGroup, Div, PreText,Select
from bokeh.plotting import figure
from bokeh.models.glyphs import Text, ImageRGBA
from ProcessOptimizer.space import Integer, Categorical
from ProcessOptimizer.plots import _map_categories, dependence, expected_min_random_sampling, cat_to_int
from ProcessOptimizer import expected_minimum
import numpy as np
from bokeh.models.markers import Circle
import matplotlib.pyplot as plt
import copy

#from bokeh.models import Toggle
#this is the main
value_adjusters=[] # The valueadjusters are kept global so we can get their values when needed
old_active_list = []
source = {'reds' : [],'red_span' : [], 'samples' : []} #this is a global parameter where we keep the data for the plots
        # so we easily can update it
#first we load models
pickle_in = open("result_cat.pickle","rb")
result = pickle.load(pickle_in)
max_pars = len(result.space.dimensions)
slider_values = copy.copy(result.x) # We define the slider values globally so we can acces slider values for plots that have been destroyed
#Then we define layout
topBox = row([]) # This row contains update button as well as toggle buttons for parameters

buttonGenerate = Button(label="Generate", button_type="success")
buttonGenerate.on_click(lambda : handleButtonGenerate(layout,result))
buttonsParameters = CheckboxButtonGroup(
        labels=['x '+str(s) for s in range(max_pars)], active=[])
button_partial_dependence = Toggle(label="Use partial dependence", button_type="default")
eval_method_dropdown = Select(title="Evaluation method:", value='Result', options=['Result','Exp min','Exp min rand','Sliders'],width = 200,height = 40)
sliderNPoints = Slider(start=1, end=20, value=5, step=1,title="n-points",width=200, height=10)
rowSliders = row([], id = 'sliders',width = 300)
rowPlots = row([],id = 'plots')
rowTop = row(buttonGenerate,buttonsParameters)
colRightSide = column(button_partial_dependence,eval_method_dropdown, sliderNPoints, id = 'rightSide')
colLeftSide = column(rowTop,rowSliders,rowPlots,id = 'leftSide')
layout = row(colLeftSide,colRightSide)

                # must give a vector of image data for image parameter
               # plot.image(image=[zi], x=-1, y=-1, dw=2, dh=2, palette="Spectral11")
#layout.children[0].children[2] = plot
#a= layout.children[0].children[2]
                #plot.circle(x='x', y='y', source=source)
#a.circle(x='x',y='y',source=source, size=10, color="red", alpha=0.5)

#checkbox_button_group = CheckboxButtonGroup(
 #       labels=["Option 1", "Option 2", "Option 3"], active=[0,1,2])
#

#button_red.on_click(update)
#button.on_click(lambda : handleButtonGenerate(layout))


def handleButtonGenerate(layout,result):
    global old_active_list
    print(get_use_partial_dependence())
    # Callback for when generate button gets pressed
    active_list = get_active_list()
    n_points = get_n_points()
    # Updating plots
    if active_list:

        x_eval = get_x_eval(result,active_list)
        #eval_x_method = layout.children[1].children[1].children[1].value
        
        layout.children[0].children[2] = get_plots_layout(layout,result,active_list,n_points,x_eval)
        #
        
        #source['reds'][1][0].data['x']=[5]
    else:
        layout.children[0].children[2] = Div(text="""<font size="10"><br><br>Let's select som parameters to plot!</font>""",
            width=500, height=100)

    #Updating sliders
    #layout.children[0].children[1] = get_sliders_layout(result,active_list)
    old_active_list = active_list
def get_slider_values():
    global value_adjusters
    global old_active_list
    vals = slider_values
    #hasattr('abc', 'upper')
    
    if True:
        n=0
        for i in old_active_list: # Use slider value from a displayed slider instead
            #val = value_adjusters[n].value
            val = layout.children[0].children[2].children[n].children[n].children[0].value
            vals[i] = val
            
            n+=1
    vals = slider_values
    return vals
def handleSliders(layout):
    # Callback whenever there is a change in the parameters sliders
    layout.children.pop()
def get_active_list():
    # The active list is the list of parameters that have been
    #clicked in the button group
    return layout.children[0].children[0].children[1].children[0].active
def get_n_points():
    # The active list is the list of parameters that have been
    #clicked in the button group
    return layout.children[1].children[2].children[0].value
def get_use_partial_dependence():
    # The active list is the list of parameters that have been
    #clicked in the button group
    return button_partial_dependence.active #layout.children[1].children[1].children[0].value
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

def get_plot_list(layout,result,active_list,n_points,x_eval):
    if get_use_partial_dependence():
        dependence_eval = None
        
    else:
        dependence_eval = x_eval
    # return a column of rows of plots
    plots=[]
    # if no parameters have been selected
    space = result.space
    model = result.models[-1]
    bounds = result.space.bounds
    samples, red_val, iscat = _map_categories(space, result.x_iters, x_eval)
    # While iterating throuhg each plot we keep track of the max and min values of y and z in case of 2d plots
    #  so we can adjust all axis later
    y_min = float("inf")
    y_max = -float("inf")
    z_min = float("inf")
    z_max = -float("inf")
    
    source['reds'] = [] # a list of lists with source values for red dots and red lines (spans)
        # for diagonal elements i.e source['reds'][2][2] the type is span. For off-diagonal type is ColumnDataSource
    source['red_span'] = [] # a list of spans of length n_parameters
    source['samples'] =[] #
    
    for i_list in range(len(active_list)): #only evaluate the paramets that have
        #been selected
        
        source['reds'].append([])
        source['samples'].append([])
        
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
                       n_samples=50, n_points=n_points, x_eval = dependence_eval)
                if isinstance(space.dimensions[i],Categorical): #check if values are categorical
                    source_red = Span(location=red_val[i]+0.5, dimension='height', line_color='red', line_width=3) # The Span class does not support categorical values. Using
                        # numerical values and adding 0.5 is a workaround
                    x_range = space.dimensions[i].categories
                    #convert integers to catogorical strings
                    xi = [bounds[i][ind] for ind in xi]
                else:
                    source_red = Span(location=red_val[i], dimension='height', line_color='red', line_width=3)
                    x_range = [bounds[i][0],bounds[i][1]]
                
                
                #x_range = [-1,1]
                #y_range = [0,min(1,np.max(y))]
                
                source_temp = ColumnDataSource(data=dict(x=xi, y=yi))
                source_samples = [] # We dont plot samples on diagonal
                
                plot = figure(plot_height=200, plot_width=200,
                tools = '',
                x_range=x_range,y_range=[0,20])
                
                plot.toolbar.logo = None #remove the bokeh logo fom figures
                plot.line('x', 'y', source=source_temp, line_width=3, line_alpha=0.6)
                # Add red line
                plot.add_layout(source_red)
                
                if np.min(yi) < y_min:
                    y_min = np.min(yi)
                if np.max(yi) > y_max:
                    y_max = np.max(yi)
                
                
            else: #contour plot
                xi,yi,zi = dependence(space, model, i, j=j, sample_points=None,
                       n_samples=50, n_points=n_points, x_eval = dependence_eval)
                if isinstance(space.dimensions[j],Categorical): #check if values are categorical
                    x_range = space.dimensions[j].categories
                    #convert integers to catogorical strings
                    xi = [bounds[j][ind] for ind in xi]
                    x_anchor = 0
                    x_span = len(result.space.dimensions[j].categories)
                    x_red = red_val[j]+0.5
                else:
                    x_anchor = bounds[j][0]
                    x_span = bounds[j][1]-bounds[j][0]
                    x_range =[np.min(xi),np.max(xi)]
                    x_red = result.x[j]
                if isinstance(space.dimensions[i],Categorical): #check if values are categorical
                    yi = [bounds[i][ind] for ind in yi]
                    y_range = space.dimensions[i].categories
                    y_anchor = 0
                    y_span = len(result.space.dimensions[i].categories)
                    y_red = red_val[i]+0.5
                else:
                    y_anchor = bounds[i][0]
                    y_span = bounds[i][1]-bounds[i][0]
                    y_range =[np.min(yi),np.max(yi)]
                    y_red = result.x[i]
                if np.min(zi) < z_min:
                    z_min = np.min(zi)
                if np.max(zi) > z_max:
                    z_max = np.max(zi)
                    
                #y = np.random.rand(100,100)
                plot = figure(plot_height=200, plot_width=200,x_range=x_range,y_range=y_range, tools = '')
                plot.toolbar.logo = None
                
                # must give a vector of image data for image parameter
                #im = get_plt_contour_as_im(xi, yi, zi)
                im = get_plt_contour_as_rgba(xi, yi, zi)
                #ImageRGBA
                #plot.image(image=[im], x=x_anchor, y=y_anchor, dw=x_span, dh=y_span, palette="Spectral11")
                
                #glyph = ImageRGBA(image=[im], x=x_anchor, y=y_anchor, dw=x_span, dh=y_span)
                print('das')
                #plot.add_glyph(glyph)
                plot.image_rgba(image=[im], x=x_anchor, y=y_anchor, dw=x_span, dh=y_span)
                print('das')
                #red_source = result.x
                #source.red[i_list].append(red_source)
                x_samples = [val[j] for val in result.x_iters]
                
                y_samples = [val[i] for val in result.x_iters]
                
                source_samples = ColumnDataSource(data=dict(x = x_samples, y = y_samples))
                source_red = ColumnDataSource(data=dict(x = [x_red], y = [y_red]))
                
                plot.circle(x='x',y='y',source=source_samples, size=10, color="black", alpha=0.5)
                #glyph = Circle(x='x',y='y', size=10, line_color="red", fill_color="red")
                #plot.add_glyph(source_red, glyph)
                plot.circle(x='x',y='y',source=source_red, size=10, color="red", alpha=1)
            if isinstance(space.dimensions[j],Categorical):
                plot.xaxis.major_label_orientation = 0.3
            if isinstance(space.dimensions[i],Categorical) and i != j:
                plot.yaxis.major_label_orientation = 1
                
                #plot.circle(x='x', y='y', source=source)
                #plot.circle(x='x',y='y',source=source_red, size=10, color="red", alpha=0.5)
            plots[i_list].append(plot)
            source['reds'][i_list].append(source_red)
            source['samples'][i_list].append(source_samples)
            
    # Now plots is a list of rows of plots
    # here we set the range for all the diagonal plots
    
    for i in range(len(plots)):
        plots[i][i].y_range = Range1d(y_min,y_max)
        
    return plots

def get_plots_layout(layout,result,active_list,n_points,x_eval):
    global value_adjusters
    plots = get_plot_list(layout,result,active_list,n_points,x_eval)
    
    value_adjusters = get_value_adjusters_list(result,active_list,x_eval)
    #print([adjuster.value for adjuster in value_adjusters])
    # We now convert the plots to a layout
    # Here we create a list of row objects, where each row object contains a list of plot objects
    if True:
        rows = []
        rows.append(row(value_adjusters[0]))
        
        for i in range(len(plots)): #we create one extra row because of the value adjusters
            
            if i == len(value_adjusters)-1:
                
                rows.append(row(*plots[i]))
                
            else:
                
                rows.append(row(*plots[i],value_adjusters[i+1]))
                
    # Here we create a column object with all the rows
        
        plot_layout = column(*rows)
    else:
        grid = []
        grid.append(value_adjusters[0])
        
        for i in range(len(plots)): #we create one extra row because of the value adjusters
            
            if i == len(value_adjusters)-1:
                
                grid.append(plots[i])
                
            else:
                
                grid.append([*plots[i],value_adjusters[i+1]])
                
    # Here we create a column object with all the rows
        plot_layout = gridplot(plots,merge_tools=True, 
                toolbar_options=dict(logo=None),plot_width=200, plot_height=200)
        

    return plot_layout

def get_value_adjusters_list(result,active_list,x_eval):
    global slider_values
    #global SOURCE
    #returns a list of sliders for non-categorical values and groups of buttons for categorical values

    bounds = result.space.bounds
    value_adjusters=[]
    n=0
    for i in active_list:

        if isinstance(result.space.dimensions[i],Categorical):
            #create a buton group with categorical values
            select_row = row(Div(text='',width = 50))
            cats = list(result.space.dimensions[i].categories)
            select = Select(title='X'+str(i), value = x_eval[i], options = cats, width = 100,height = 15)
            select.js_on_change('value', CustomJS(args=dict(source=source, n=n,cats=cats), code="""
                
                var ind = cats.indexOf(cb_obj.value); // Convert categorical to index
                source['reds'][n][n]['location'] = ind + 0.5;
                for (i = n+1; i < source.reds.length; i++) { 
                    source.reds[i][n].data.x = [ind + 0.5] ;
                    source.reds[i][n].change.emit()
                }
                // Then all the horizontal plots
                    for (j = 0; j < n; j++) { 
                    source.reds[n][j].data.y = [ind + 0.5] ;
                    source.reds[n][j].change.emit();
                }
                    """)
                )
            
            #select.on_change(handle_eval_change)
            select_row.children.append(select)
            select_row.children.append(Div(text='',width = 50))
            select_col = column(Div(text='',height = 50))
            select_col.children.append(select_row)
            select_col.children.append(Div(text='',height = 50))
            value_adjusters.append(select)

            slider_values[i] = x_eval[i]
        else: 
            slider_row=row(Div(text='',width = 50))
            start = bounds[i][0]
            end = bounds[i][1]
            span = end-start
            value = start+span/2 
        #we create space on each side of slider
            slider  = Slider(start=start, end=end, value=x_eval[i], step=.1,title='X'+str(i),width=150, height=30)
            #lambda : handleButtonGenerate(layout,result,source)
            #slider.on_change('hej',handle_eval_change)
            #slider.js_on_change('value', f_callback,args = i)

            slider.js_on_change('value', CustomJS(args=dict(source=source, n=n), code="""
            // First we  change the diagonal element.
            source.reds[n][n].location = cb_obj.value;
            source.reds[n][n].change.emit()
            // Then we change all the vertical plots
            for (i = n+1; i < source.reds.length; i++) { 
            source.reds[i][n].data.x = [cb_obj.value] ;
            source.reds[i][n].change.emit();
            }
            // Then all the horizontal plots
            for (j = 0; j < n; j++) { 
            source.reds[n][j].data.y = [cb_obj.value] ;
            source.reds[n][j].change.emit();
            }
                    """)
                )
            slider_row.children.append(slider)
            slider_row.children.append(Div(text='',width = 50))
            slider_col = column(Div(text='',height = 50))
            slider_col.children.append(slider_row)
            slider_col.children.append(Div(text='',height = 50))
            value_adjusters.append(slider)
            slider_values[i] = x_eval[i]
        n+=1
    return value_adjusters#row(*sliders,width = 1000)
#def plot_red(plot,values):

def get_plt_contour_as_im(xi, yi, zi):
    fig = plt.figure()
    ax = fig.add_axes([0.,0.,1.,1.])
    ax = plt.gca()
    ax.contourf(xi, yi, zi, 10,
                                    locator=None, cmap='Greys')
    plt.axis('off')
    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    X=X[:,:,1]
    return X
def get_plt_contour_as_rgba(xi, yi, zi):
    fig = plt.figure()
    ax = fig.add_axes([0.,0.,1.,1.])
    ax = plt.gca()
    ax.contourf(xi, yi, zi, 10,
                                    locator=None, cmap='Greys')
    plt.axis('off')
    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    xdim= X.shape[1]
    ydim= X.shape[0]
# Create an array representation for the image `img`, and an 8-bit "4
# layer/RGBA" version of it `view`.
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
# Copy the RGBA image into view, flipping it so it comes right-side up
# with a lower-left origin
    view[:,:,:] = np.flipud(X)

    #X=X[:,:,1]
    return img
def get_x_eval(result,active_list):

    '''Returns values for parameters defined by the eval-method dropdown menu'''
    _,iscat = cat_to_int(result.space,[result.x])
    eval_method = eval_method_dropdown.value
    # Expected_minimum does not support categorical values
    if eval_method == 'Exp min' and any(iscat):
        eval_method = "Exp min rand"
    
    if eval_method == 'Result':
        x = result.x

    elif eval_method == 'Sliders':
        x = get_slider_values()
    elif eval_method == 'Exp min' and not any(iscat):
        x = expected_minimum(result, n_random_starts=1, random_state=None)
    elif eval_method == 'Exp min rand':
        x = expected_min_random_sampling(result.models[-1], result.space, np.min([10**len(result.x),10000]))
    else:
        ValueError('Could not find evalmethod from dropdown menu')
    return x
curdoc().add_root(layout)
handleButtonGenerate(layout,result)