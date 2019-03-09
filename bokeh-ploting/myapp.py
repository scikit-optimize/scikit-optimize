''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure
from ProcessOptimizer.benchmarks import branin as branin
from ProcessOptimizer.benchmarks import hart6 as hart6_
from ProcessOptimizer.plots import plot_objective
from ProcessOptimizer import gp_minimize, forest_minimize, dummy_minimize
from ProcessOptimizer import plots
from ProcessOptimizer.plots import dependence
# For reproducibility
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
plt.set_cmap("viridis")

# Here we define a function that we evaluate.
def funny_func(x):
    s = 0
    for i in range(len(x)):
        s += (x[i])**2
    return s

# We run forest_minimize on the function
bounds = [(-1, 1.),] * 2
n_calls = 10

result = forest_minimize(funny_func, bounds, n_calls=n_calls, base_estimator="ET",
                             random_state=4)
space = result.space
model = result.models[-1]
x1,y1 = dependence(space, model, 0, j=None, sample_points=None, n_samples=250, n_points=20, x_eval = [0,0])
x2,y2 = dependence(space, model, 1, j=None, sample_points=None, n_samples=250, n_points=20, x_eval = [0,0])
#fig = plot_objective(result,usepartialdependence = True, n_points = 10)
# Set up data
N = 200
#x = np.linspace(0, 4*np.pi, N)
#y = np.sin(x)
source1 = ColumnDataSource(data=dict(x=x1, y=y1))
source2 = ColumnDataSource(data=dict(x=x2, y=y2))

# Set up plot
plot1 = figure(plot_height=400, plot_width=400, title="d1",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[np.min(x1), np.max(x1)], y_range=[np.min(y1), np.max(y1)])

plot1.line('x', 'y', source=source1, line_width=3, line_alpha=0.6)

plot2 = figure(plot_height=400, plot_width=400, title="d2",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[np.min(x2), np.max(x2)], y_range=[np.min(y2), np.max(y2)])
plot2.line('x', 'y', source=source2, line_width=3, line_alpha=0.6)


# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-1, end=1, step=0.1)
amplitude = Slider(title="amplitude", value=0.0, start=-1, end=1, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)


# Set up callbacks
def update_title(attrname, old, new):
    plot1.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value

    # Generate the new curve
    x1,y1 = dependence(space, model, 1, j=None, sample_points=None, n_samples=250, n_points=10, x_eval = [a,b])
    x2,y2 = dependence(space, model, 1, j=None, sample_points=None, n_samples=250, n_points=10, x_eval = [a,b])
    source1.data = dict(x=x1, y=y1)
    source2.data = dict(x=x2, y=y2)
    plot2.x_range.start = np.min(x2)
    plot2.x_range.end = np.max(x2)
    plot2.x_range.start = np.min(y2)
    plot2.x_range.end = np.max(y2)
for w in [offset, amplitude, phase, freq]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(text, offset, amplitude, phase, freq)

curdoc().add_root(row(inputs, plot1, width=800))
curdoc().add_root(row(inputs, plot2, width=800))
curdoc().title = "Sliders"