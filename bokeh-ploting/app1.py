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
from bokeh.models.glyphs import Text
max_pars = 5

# Create a button group with n buttons = max_pars
source_red = ColumnDataSource(data=dict(x=[0.5], y=[0.5]))
checkbox_button_group = CheckboxButtonGroup(
        labels=['x '+str(s) for s in range(max_pars)], active=[])
def funny_func(x):
    s = 0
    for i in range(len(x)):
        s += x[i]**2*(i+1)
    return s
def dep(x,i,N,j=None):
    if j == None: # 1d plot
        y=np.zeros(N)
        xi = np.linspace(-1,1,N)
        for ii in range(N):
            x_new = np.array(x,copy=True)
            x_new[i] = xi[ii]
            y[ii] = funny_func(x_new)
        return y,xi
    else: #2d plot
        y=np.zeros([N,N]) #preallocate 2d matrix
        xi = np.linspace(-1,1,N)
        xj = np.linspace(-1,1,N)
        for ii in range(N):
            for jj in range(N):
                x_new = np.array(x,copy=True)
                x_new[i] = xi[ii]
                x_new[j] = xj[jj]
                y[ii,jj] = funny_func(x_new)
        return y,xi,xj
def update_plots():
    activelist = checkbox_button_group.active
    #layout.children.pop()
    #layout.children.append(column(get_plot_layout(100,activelist)))
    layout.children[2] = column(get_plot_layout(100,activelist))
    layout.children[1] = column(get_slider_layout(activelist))
    #get_slider_layout
    print(layout.children)
    for i in layout.children:
        print(i)
#funciton for only updating red markers
def update_reds():
    source_red.data = dict(x=[0.5], y=[0.5])

#Make a row of sliders
def get_slider_layout(activelist):
    sliders=[]
    if len(activelist):
        for i in range(len(activelist)):
            sliders.append(Slider(start=0.1, end=10, value=1, step=.1,title="Amplitude",width=200, height=10))
        return row(*sliders)
    else:
        div = Div(text="""<font size="2">No Sliders</font>""",
width=500, height=100)
        return div
#Here we make a list with lists of plots'
def get_plot_layout(N,activelist):
    
    x=[float(0)]*max_pars
    plots=[]
    # if no parameters have been selected
    if not activelist:
        div = Div(text="""<font size="6">No parameters selected</font>""",
width=500, height=100)
        #t = PreText(text="""No parameters selected""", width=500, height=100,text_font_size = 30)
        return div
    for i_list in range(len(activelist)):
        plots.append([])
        for j_list in range(len(activelist)): #we only evaluate the lower left half of the grid
            #if i in active_pars or j in active_pars:
                #break
            i =activelist[i_list]
            j = activelist[j_list]
            if j>i:
                break
            elif i==j: #diagonal
                y,xi = dep(x,i,N,j=None)
                #x_range = [-1,1]
                #y_range = [0,min(1,np.max(y))]
                source = ColumnDataSource(data=dict(x=xi, y=y))
                plot = figure(plot_height=200, plot_width=200, title=str(i)+str(j),
                tools = '',
                x_range=[-1,1],y_range=[0, 10])
                plot.toolbar.logo = None #remove the bokeh logo fom figures
                plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
            else: #contour plot
                y,xi,xj = dep(x,i,N,j=j)
                #y = np.random.rand(100,100)
                #print(y)
                plot = figure(plot_height=200, plot_width=200,x_range=(-1, 1), y_range=(-1, 1),
                tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])

                # must give a vector of image data for image parameter
                plot.image(image=[y], x=-1, y=-1, dw=2, dh=2, palette="Spectral11")

                
                #plot.circle(x='x', y='y', source=source)
                plot.circle(x='x',y='y',source=source_red, size=10, color="red", alpha=0.5)
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
button = Button(label="Update", button_type="success")
button.on_click(update_plots)
button_red = Button(label="Update reds", button_type="success")
button_red.on_click(update_reds)
#checkbox_button_group = CheckboxButtonGroup(
 #       labels=["Option 1", "Option 2", "Option 3"], active=[0,1,2])
#print(checkbox_button_group.active)
button_row=row(button,button_red,checkbox_button_group)
layout = column(button_row,get_slider_layout([1,2,3]),get_plot_layout(3,checkbox_button_group.active))
curdoc().add_root(layout)

#plt.plot([1, 2, 3, 4])
##plt.ylabel('some numbers')
#plt.show()