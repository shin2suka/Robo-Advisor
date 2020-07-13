import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

df_composition = pd.read_csv("visualization/data/df_composition.csv")
labels = df_composition['Class']

fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['CAD', 'USD'])
fig.add_trace(go.Pie(labels=labels, values=df_composition['CAD'], scalegroup='one',
                     name="CAD Account"), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=df_composition['USD'], scalegroup='one',
                     name="USD Account"), 1, 2)

fig.show()
