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
from utils import get_plot_layout
import matplotlib.pyplot as plt
#from bokeh.models import Toggle
from bokeh.models.widgets import Toggle, CheckboxButtonGroup, Div, PreText, Slider

#this is the main
#first we load models
pickle_in = open("data.pickle","rb")
result = pickle.load(pickle_in)
max_pars = len(result.x)
#Then we define layout
topBox = row([]) # This row contains update button as well as toggle buttons for parameters

buttonGenerate = Button(label="Generate", button_type="success")
buttonGenerate.on_click(lambda : handleButtonGenerate(layout,result))
buttonsParameters = CheckboxButtonGroup(
        labels=['x '+str(s) for s in range(max_pars)], active=[])
button_partial_dependence = Toggle(label="Use partial dependence", button_type="success")
sliderNPoints = Slider(start=1, end=20, value=5, step=1,title="n-points",width=200, height=10)
rowSliders = row([], id = 'sliders')
rowPlots = row([],id = 'plots')
rowTop = row(buttonGenerate,buttonsParameters)
colRightSide = column(button_partial_dependence, sliderNPoints, id = 'rightSide')
colLeftSide = column(rowTop,rowSliders,rowPlots,id = 'leftSide')
layout = row(colLeftSide,colRightSide)


source = ColumnDataSource(data=dict(x=[0,1,2,3], y=[0,1,2,3]))
plot = figure(plot_height=200, plot_width=200, title='hej',
                tools = '',
                x_range=[0,4],y_range=[0, 4])
plot.toolbar.logo = None #remove the bokeh logo fom figures
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
                # must give a vector of image data for image parameter
               # plot.image(image=[zi], x=-1, y=-1, dw=2, dh=2, palette="Spectral11")
#layout.children[0].children[2] = plot
#a= layout.children[0].children[2]
                #plot.circle(x='x', y='y', source=source)
#a.circle(x='x',y='y',source=source, size=10, color="red", alpha=0.5)

#checkbox_button_group = CheckboxButtonGroup(
 #       labels=["Option 1", "Option 2", "Option 3"], active=[0,1,2])
#print(checkbox_button_group.active)

#button_red.on_click(update)
#button.on_click(lambda : handleButtonGenerate(layout))
curdoc().add_root(layout)

def handleButtonGenerate(layout,result):
    # Callback for when generate button gets pressed
    active_list = get_active_list()
    n_points = get_n_points()
    layout.children[0].children[2] = get_plot_layout(layout,result,active_list,n_points)
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
    return layout.children[1].children[1].children[0].value
def get_use_partial_dependence():
    # The active list is the list of parameters that have been
    #clicked in the button group
    return layout.children[1].children[1].children[0].value

