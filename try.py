import networkx as nx
import holoviews as hv
from holoviews import opts

from holoviews.element.graphs import layout_nodes
from bokeh.sampledata.airport_routes import routes, airports

hv.extension('bokeh')

# Create dataset indexed by AirportID and with additional value dimension
airports = hv.Dataset(airports, ['AirportID'], ['Name', 'IATA', 'City'])

label = 'Alaska Airline Routes'

# Select just Alaska Airline routes
as_graph = hv.Graph((routes[routes.Airline=='AS'], airports), ['SourceID', "DestinationID"], 'Airline', label=label)

as_graph = layout_nodes(as_graph, layout=nx.layout.fruchterman_reingold_layout)
labels = hv.Labels(as_graph.nodes, ['x', 'y'], ['IATA', 'City'], label=label)

graph_with_labels = (as_graph * labels).opts(
    opts.Graph(directed=True, node_size=8, bgcolor='gray', xaxis=None, yaxis=None,
               edge_line_color='white', edge_line_width=1, width=800, height=800, arrowhead_length=0.01,
               node_fill_color='white', node_nonselection_fill_color='black'),
    opts.Labels(xoffset=-0.04, yoffset=0.03, text_font_size='10pt'))

hv.save(graph_with_labels, 'graph_with_labels.html', fmt='html')