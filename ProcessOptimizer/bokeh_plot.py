
import pickle
from bokeh.io import curdoc, showing
from bokeh.layouts import row, column, gridplot, Spacer
from bokeh.models import ColumnDataSource, Button, Range1d, Span, CustomJS
from bokeh.models.widgets import Slider, TextInput,Toggle, CheckboxButtonGroup, Div, PreText,Select
from bokeh.plotting import figure
from bokeh.models.glyphs import Text, ImageRGBA
from ProcessOptimizer.space import Integer, Categorical
from ProcessOptimizer.plots import _map_categories, dependence, expected_min_random_sampling
from ProcessOptimizer import expected_minimum
import numpy as np
from bokeh.models.markers import Circle
import matplotlib.pyplot as plt
import math
import copy
from bokeh.server.server import Server

# Quick introduction. 
# The GUI conists of several elements.
# Plots: the plots as they also appear in objective_plot
# Selectors: Sliders that appear on top of the plots with which the red markers can be adjusted
# Toggle_x button group: A group of buttons where we can toggle what parameters we want to plot.
# Generate button: Dependence is only calculated a new when this buton is pressed
# Other buttons: With buttons we can toggle partial dependence on and off, control the n_points variable (resolution when calculating dependence)
#   and control the evaluation method

# Here we define the global variables (global variables are generally not advices so maybe we should change this at some point)
result = None
x_eval_selectors=None # This list holds the sliders and selectors for parameter values
old_active_list = None # Used to determine what x_eval_selectors were present before last update
source = {'reds' : []}  # The 'reds' field is a (NxN) list. The diagonal holds a Span 
        # object with a "location" value, that detemines the location for the red line in the 1d plots.
        # The off diagonals holds coordinates for the red markers used in the contour plot.
        # 'samples"
x_eval_selectors_values = None
max_pars = None #Defines the maximum number of parameters that can be plotted
layout = None
button_partial_dependence = None
button_generate = None
buttons_toggle_x = None
dropdown_eval_method = None
slider_n_points = None

def set_globals(parsed_result):
    global button_generate, buttons_toggle_x, dropdown_eval_method, slider_n_points, max_pars, source, old_active_list
    global result, x_eval_selectors, layout, x_eval_selectors_values, button_partial_dependence
    #Here we se the values of the global variables and create the layout
    result = parsed_result
    x_eval_selectors_values = copy.copy(result.x)
    max_pars = len(result.space.dimensions)

    # Layout
    button_generate = Button(label="Generate", button_type="success")
    button_generate.on_click(lambda : handle_button_generate(layout,result))
    buttons_toggle_x = CheckboxButtonGroup(
            labels=['x '+str(s) for s in range(max_pars)], active=[])
    button_partial_dependence = Toggle(label="Use partial dependence", button_type="default")
    dropdown_eval_method = Select(title="Evaluation method:", value='Result', options=['Result','Exp min','Exp min rand','Sliders'],width = 200,height = 40)
    slider_n_points = Slider(start=1, end=20, value=5, step=1,title="n-points",width=200, height=10)
    row_x_eval_selectors = row([],width = 300)
    row_plots = row([])
    row_top = row(button_generate,buttons_toggle_x)
    col_right_side = column(button_partial_dependence,dropdown_eval_method, slider_n_points)
    col_left_side = column(row_top,row_x_eval_selectors,row_plots)
    layout = row(col_left_side,col_right_side)
def handle_button_generate(layout,result):
    global old_active_list
    # Callback for when generate button gets pressed
    active_list = get_active_list() # Get the current active list
    n_points = get_n_points()
    # Updating plots
    if active_list: # Only plot if there is at leas one selection
        x_eval = get_x_eval(result,active_list) # x_eval is used both for red markers and as the values for
            #calculating dependence  
        # We update the part of the layout that contains the plots
        layout.children[0].children[2] = get_plots_layout(layout,result,active_list,n_points,x_eval)
    else: # If no selection we encourage the user to select some parameters
        layout.children[0].children[2] = Div(text="""<font size="10"><br><br>Let's select som parameters to plot!</font>""",
            width=500, height=100)
    # Update the old_active_list for next update
    old_active_list = active_list

def get_x_eval_selectors_values():
    # Returns the values for the x_eval_selectors. Uses the global x_eval_selectors_values
    # and for each selector that is present in the GUI we replace the corresponding value.
    # Even though a selector isnot present in the GUI the last value that it had i still retrived.
    global x_eval_selectors
    global old_active_list
    values = x_eval_selectors_values  
    if True:
        n=0 # The index of the selctors in GUI
        for i in old_active_list: # Use value from GUI selector instead
            val = x_eval_selectors[n].value
            values[i] = val
            n+=1
    values = x_eval_selectors_values
    return values
def get_active_list():
    # returns the a list of the indexes of parameters that have been toggled in 
    # the buttons_toggle_x button-group in the GUI
    return buttons_toggle_x.active
def get_n_points():
    # Returns the value of the slider_n_points slider in the GUI
    return slider_n_points.value
def get_use_partial_dependence():
    # Returns True or false depending on wether or not the partial dependence button
    # is toggled in the GUI
    return button_partial_dependence.active
def get_plot_list(layout,result,active_list,n_points,x_eval):
    # Returns a NxN list of plots where N is the number of parameters to be plotted.
    # The diagonal is 1d plots and the off diagonal is contour plots
    global source
    if get_use_partial_dependence():
        # Not passing any eval values to dependency function makes it calculate
        # partial dependence
        dependence_eval = None
    else:
        dependence_eval = x_eval

    plots=[]
    space = result.space
    model = result.models[-1]
    bounds = result.space.bounds
    # the iscat variable is a list of bools. True if parameter is categorical. False otherwise.
    # red_vals are the coordinates for the x_eval that is mapped to integers in case they are 
    # categorical.
    _, red_vals, iscat = _map_categories(space, result.x_iters, x_eval)
    # We add 0.5 to the value of all categorical integers. This is due to the way bokeh
    # handles plotting of categorical values
    red_vals = [val+0.5 if iscat[i] else val for i, val in enumerate(red_vals)]
    # While iterating through each plot we keep track of the max and min values of y and z in case of 2d plots.
    # We use these values to set the axis of all plots, so they share the same axis.
    y_min = float("inf")
    y_max = -float("inf")
    
    source['reds'] = [] # reset the sources for red markers
    
    for i_active in range(len(active_list)): # Only plot the selected parameters
        source['reds'].append([])
        plots.append([])
        for j_active in range(len(active_list)):
            i = active_list[i_active]
            j = active_list[j_active]
            if j>i: # We only plot the lower left half of the grid, to avoid duplicates.
                break
            elif i==j: # Diagonal
                # Passing j = None to dependence makes it calculate a diagonal plot
                xi,yi = dependence(space, model, i, j=None, sample_points=None,
                    n_samples=250, n_points=n_points, x_eval = dependence_eval)
                if iscat[i]: # Categorical
                    x_range = space.dimensions[i].categories
                    # Convert integers to catogorical strings
                    xi = [bounds[i][ind] for ind in xi]
                else: # Numerical
                    x_range = [bounds[i][0],bounds[i][1]]
                
                # Source red is what we end up appending to the global source variable
                # So the location of the red line can be changed interactively
                source_red = Span(location=red_vals[i], dimension='height', line_color='red', line_width=3) 
                plot = figure(plot_height=200, plot_width=200, tools = '', x_range=x_range,y_range=[0,20])
                source_line = ColumnDataSource(data=dict(x=xi, y=yi))
                plot.line('x', 'y', source=source_line, line_width=3, line_alpha=0.6)
                # Add span i.e red line to plot
                plot.add_layout(source_red)
                #update max and minimum y _values
                if np.min(yi) < y_min:
                    y_min = np.min(yi)
                if np.max(yi) > y_max:
                    y_max = np.max(yi)
                
                
            else: # Contour plot
                xi,yi,zi = dependence(space, model, i, j=j, sample_points = None,
                    n_samples=50, n_points=n_points, x_eval = dependence_eval)

                if iscat[j]: #check if values are categorical
                    # Convert integers to catogorical strings
                    xi = [bounds[j][ind] for ind in xi]
                    # At what coordinate in plot to anchor the contour image.
                    # In case of categorical this should be set to 0.
                    x_anchor = 0
                    # The size the image should take up in the plot
                    x_span = len(result.space.dimensions[j].categories)
                    # Range for axis
                    x_range = space.dimensions[j].categories
                else:
                    x_anchor = bounds[j][0]
                    x_span = bounds[j][1]-bounds[j][0]
                    x_range =[np.min(xi),np.max(xi)]
                if iscat[i]: #check if values are categorical
                    yi = [bounds[i][ind] for ind in yi]
                    y_range = space.dimensions[i].categories
                    y_anchor = 0
                    y_span = len(result.space.dimensions[i].categories)
                else:
                    y_anchor = bounds[i][0]
                    y_span = bounds[i][1]-bounds[i][0]
                    y_range =[np.min(yi),np.max(yi)]

                plot = figure(plot_height=200, plot_width=200,x_range=x_range,y_range=y_range, tools = '')
                
                # Get an rgba contour image from matplotlib as bokeh does not support contour plots
                im = get_plt_contour_as_rgba(xi, yi, zi)
                plot.image_rgba(image=[im], x=x_anchor, y=y_anchor, dw=x_span, dh=y_span)
                # x and y samples are the coordinates of the parameter values that have been
                # sampled during the creation of the model
                x_samples = [val[j] for val in result.x_iters]
                y_samples = [val[i] for val in result.x_iters]
                source_samples = ColumnDataSource(data=dict(x = x_samples, y = y_samples))
                source_red = ColumnDataSource(data=dict(x = [red_vals[j]], y = [red_vals[i]]))
                # We plot samples as black circles and the evaluation marker as a  red circle
                plot.circle(x='x',y='y',source=source_samples, size=10, color="black", alpha=0.5)
                plot.circle(x='x',y='y',source=source_red, size=10, color="red", alpha=1)
            # We rotate the categorical labels slighty so they take up less space
            if iscat[j]:
                plot.xaxis.major_label_orientation = 0.3
            if iscat[i] and i != j: # In case of diagonal the y-labels are numbers and
                    # therefore should not be rotated
                plot.yaxis.major_label_orientation = 1.2

            plot.toolbar.logo = None # Remove the bokeh logo fom figures
            plots[i_active].append(plot) 
            source['reds'][i_active].append(source_red)

    # Setting the same y-range for all diagonal plots for easier comparison
    for i in range(len(plots)):
        plots[i][i].y_range = Range1d(y_min,y_max)
    return plots

def get_plots_layout(layout,result,active_list,n_points,x_eval):
    global x_eval_selectors
    plots = get_plot_list(layout,result,active_list,n_points,x_eval)
    x_eval_selectors = get_x_eval_selectors_list(result,active_list,x_eval)
    # Create the layout using the lists of plots and selectors.
    # The layout consists of rows that we append plots and selectors to.
    # The selectors should be on top of each diagonal plot, and therefore
    # the layout can be regarded as a (N+1,N) grid to make room for the top selector. 
    # The top row therefore consists of only on selector and the bottom row has no
    # selector as the selector for the last diagonal was added in the second last row.


    rows = []
    rows.append(row(x_eval_selectors[0])) # The selector in the top left corner
    for i in range(len(plots)):
        if i == len(x_eval_selectors)-1: # Last row
            rows.append(row(*plots[i]))
        else:
            rows.append(row(*plots[i],x_eval_selectors[i+1]))  
    # Create a column with all the rows
    plot_layout = column(*rows)
    return plot_layout

def get_x_eval_selectors_list(result,active_list,x_eval):
    # Returns a list of selectors. The selectors are sliders for numerical values and dropdown menus
    # ("Select" object) for categorical values. The selectors are interactive with callbacks everytime
    # a changed by the user
    global x_eval_selectors_values

    bounds = result.space.bounds # Used for defining what values can be selected
    x_eval_selectors=[]
    n=0 # Index of the plots. Example: If only parameter 3 and 5 is being plotted
    # the selectors for these parameters still have index n = 0 and n= 1.
    for i in active_list: # Only get selecters that is going to be shown in GUI
        if isinstance(result.space.dimensions[i],Categorical): # Categorical
            cats = list(result.space.dimensions[i].categories) # Categories
            # Create a "Select" object which is a type of dropdown menu
            # This object gets a title equal to the parameter number, and the value is set to
            # x_eval
            select = Select(title='X'+str(i), value = x_eval[i], options = cats, width = 100,height = 15)
            # Here we define a callback that updates the appropiate red markers by changing
            # with the current value of the selector by changing the global "source" variable
            # The callback function is written in javascript
            select.js_on_change('value', CustomJS(args=dict(source=source, n=n,cats=cats), code="""
                // Convert categorical to index
                var ind = cats.indexOf(cb_obj.value); 
                // Change red line in diagonal plots
                source['reds'][n][n]['location'] = ind + 0.5;
                // Change red markers in all contour plots
                // First we change the plots in a vertical direction
                for (i = n+1; i < source.reds.length; i++) { 
                    source.reds[i][n].data.x = [ind + 0.5] ;
                    source.reds[i][n].change.emit()
                }
                // Then in a horizontal direction
                for (j = 0; j < n; j++) { 
                    source.reds[n][j].data.y = [ind + 0.5] ;
                    source.reds[n][j].change.emit();
                }
                """)
            )
            x_eval_selectors.append(select)
            # We update the global selector values
            x_eval_selectors_values[i] = x_eval[i]
        else: # Numerical
            # For numerical values we create a slider
            # Minimum and maximum values for slider
            start = bounds[i][0]
            end = bounds[i][1]
            step = get_step_size(start,end) # We change the stepsize according to the range of the slider
            slider  = Slider(start=start, end=end, value=x_eval[i], step=step,title='X'+str(i),width=150, height=30)
            # javascript callback function that gets called everytime a user changes the slider value
            slider.js_on_change('value', CustomJS(args=dict(source=source, n=n), code="""
                source.reds[n][n].location = cb_obj.value;
                source.reds[n][n].change.emit()
                for (i = n+1; i < source.reds.length; i++) { 
                    source.reds[i][n].data.x = [cb_obj.value] ;
                    source.reds[i][n].change.emit();
                }
                for (j = 0; j < n; j++) { 
                    source.reds[n][j].data.y = [cb_obj.value] ;
                    source.reds[n][j].change.emit();
                }
                """)
            )
            x_eval_selectors.append(slider)
            x_eval_selectors_values[i] = x_eval[i]
        n+=1
    return x_eval_selectors

def get_plt_contour_as_rgba(xi, yi, zi):
    # Returns a matplotlib contour plot as an rgba image
    # We create a matplotlib figure and draws it so we can capture the figure as an image.
    fig = plt.figure()
    ax = fig.add_axes([0.,0.,1.,1.])
    ax = plt.gca()
    ax.contourf(xi, yi, zi, 10, locator=None, cmap='viridis_r')
    plt.axis('off')
    fig.canvas.draw()
    # Grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    xdim= X.shape[1]
    ydim= X.shape[0]
    # Converting image so that bokeh's image_rgba can read it. (code stolen of the internet)
    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    view[:,:,:] = np.flipud(X)
    plt.close()
    return img

def get_x_eval(result,active_list):
    # Returns the evaluation values that is defined by the evaluationmethod dropdown menu
    _, _, iscat = _map_categories(result.space, [result.x], result.x)
    eval_method = dropdown_eval_method.value # Get evaluation method from GUI
    
    if eval_method == 'Exp min' and any(iscat):
        # Expected_minimum does not support categorical values
        eval_method = "Exp min rand"
    if eval_method == 'Result':
        x = result.x
    elif eval_method == 'Sliders':
        x = get_x_eval_selectors_values()
    elif eval_method == 'Exp min' and not any(iscat):
        x = expected_minimum(result, n_random_starts=10, random_state=None)
    elif eval_method == 'Exp min rand':
        x = expected_min_random_sampling(result.models[-1], result.space, np.min([10**len(result.x),10000]))
    else:
        ValueError('Could not find evalmethod from dropdown menu')
    return x
def get_step_size(start,end):
    # Returns the stepsize to be used for sliders the stepsize will always be of the form 10**x
    range_log = round((math.log(end-start,10)))
    # The bigger the range the bigger the stepsize
    step = 10**(range_log-3)
    return step
def modify_doc(doc):
# Add layout to document
    doc.add_root(layout)
# Update once to initialize message
    handle_button_generate(layout,result)
def start(parsed_result):
    # Set the global variables using the parsed "result"
    set_globals(parsed_result)
    # Start server
    server = Server({'/': modify_doc}, num_procs=1)
    server.start()
    print('Opening Bokeh application on http://localhost:5006/')
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()