from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL, ctx
import uuid
import os
if __name__ == "__main__":
    from lib import load, annotate_image_plotly, batch, LoG
else:
    from .lib import load, annotate_image_plotly, batch, LoG
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import dash_daq as daq

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    # r"C:\Users\andre\Downloads\Leadmark Free Website Template - Free-CSS.com.zip\leadmark\public_html\assets\css\leadmark.css",
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets = external_stylesheets)
app.layout = html.Div()

app.layout = html.Div([
    html.Button("Add element", id={"type": "add", "index": -1}, n_clicks=0),
    html.Div(id='dropdown-container', children=[]),
    html.Div(id='dropdown-container-output')
])

def plotter(
    path,
    frames,
    
    flavor,
    
    diameter,
    separation,
    noise_size,
    smoothing_size,
    filter_close,

    min_r,
    max_r,
    n_radii,
    log_spacing,
    overlap_threshold,

    minmass,
    max_iterations,
    characterize,
    # search_range,
    # memory,
    doughnut_correction,
    bg_radius,
    gap_radius,
    snr,
    minmass_snr,
    truncate_preprocessed,
    illumination_sigma,
    illumination_correction_per_frame,
    adaptive_background,

    **kwargs
):
    try:
        vid = load(path, keys = range(frames)).astype("float32")
    except FileNotFoundError:
        raise FileNotFoundError("The file was not found or the file format is not supported")
    except IndexError:
        raise FileNotFoundError("The file was not found or the file format is not supported")
    
    vid = vid.reshape(-1, *vid.shape[-2:])

    shared_args = dict(
        minmass = minmass,
        max_iterations = max_iterations,
        characterize = characterize,
        # search_range = search_range,
        # memory = memory,
        doughnut_correction = doughnut_correction,
        bg_radius = bg_radius,
        gap_radius = gap_radius,
        snr = snr,
        minmass_snr = minmass_snr,
        truncate_preprocessed = truncate_preprocessed,
        illumination_sigma = illumination_sigma,
        illumination_correction_per_frame = illumination_correction_per_frame,
        adaptive_background = adaptive_background,
    )
    command = "gpu_tracking."
    if flavor == "Trackpy":
        command += "batch(\n"
        command += f"    r'{path}',\n"
        command += f"    {diameter},\n"
        args = dict(
            separation = separation,
            noise_size = noise_size,
            smoothing_size = smoothing_size,
            filter_close = filter_close,
        )
        df = batch(
            vid,
            diameter,
            **args,
            **shared_args
        )
        
        r = diameter / 2
    elif flavor == "LoG":
        command += "LoG(\n"
        command += f"    r'{path}',\n"
        command += f"    {min_r},\n"
        command += f"    {max_r},\n"
        args = dict(
            n_radii = n_radii,
            log_spacing = log_spacing,
            overlap_threshold = overlap_threshold,
        )
        df = LoG(
            vid,
            min_r,
            max_r,
            **args,
            **shared_args,
        )
        r = None
    for key, val in args.items():
        if val is not None:
            command += f"    {key} = {val},\n"
    for key, val in shared_args.items():
        if val is not None:
            command += f"    {key} = {val},\n"
    command += ")"
    
    fig = annotate_image_plotly(vid[-1], df, frame = frames-1, r = r, **kwargs)
    return fig, command, df, vid.shape[1:]

def create_input(name, idx, *args, which = dcc.Input, **kwargs):
    defaults = {"type": "number"}
    defaults.update(kwargs)
    kwargs = defaults
    return [f"{name} ", which(*args, id = {"type": name, "index": idx}, **kwargs)]

def create_element(idx):
    new_ele = html.Div([
        html.Div([
            html.Button("Submit", id = {"type": "modify", "index": idx}, style = {"display": "inline-block"}),
            html.Button("Previous frame", id = {"type": "prev", "index": idx}, style = {"display": "inline-block"}),
            html.Button("Next frame", id = {"type": "next", "index": idx}, style = {"display": "inline-block"}),
            "Copy python command to clipboard: ",
            dcc.Clipboard(id = {"type": "clip", "index": idx}, style = {"display": "inline-block"}),
        ], style = {"margin-top": "15px"}),
        
        html.Div([
            # dcc.Input(id = {"type": "input-path", "index": idx}, value = r"C:\Users\andre\Documents\tracking_optimizations\gpu-tracking\testing\easy_test_data.tif"),
            dcc.Input(id = {"type": "input-path", "index": idx}, value = os.getcwd()),
            *create_input("Up to frame", idx, value = 1, type = "number"),
            # dcc.Upload(),
            # dcc.Graph(figure = px.imshow(np.zeros((100, 100)), color_continuous_scale = "viridis"), id = {"type": "graph", "index": idx}),
        ]),
        html.Div([
            dcc.RadioItems(["Trackpy", "LoG"], "Trackpy", id = {"type": "flavor", "index": idx}, style = {"display": "inline-block"}),
            dcc.Checklist(["Extended options"], id = {"type": "extended", "index": idx}, style = {"display": "inline-block"}),
            dcc.Checklist(["Color Options"], id = {"type": "color_options_toggle", "index": idx}, style = {"display": "inline-block"}),
            dcc.Dropdown([], id = {"type": "var 1", "index": idx}, style = {"horizontalAlign": "right"}),
            dcc.Dropdown([], id = {"type": "var 2", "index": idx}, style = {"horizontalAlign": "right"}),
            html.Div([
            daq.ColorPicker(
                id = {"type": "colorpicker", "index": idx},
                value = dict(hex = "#FFFFFF")
            ),
            dcc.Dropdown(px.colors.named_colorscales(), value = "viridis", id = {"type": "colorscale", "index": idx}, style = {"horizontalAlign": "right"}),
            ], id = {"type": "color_options", "index": idx}, style = {"display": "none"}),
        ]),
        html.Div([
            *create_input("diameter", idx, value = 7),
            html.Div([
                *create_input("separation", idx),
                dcc.Checklist(["filter_close"], value = ["filter_close"], id = {"type": "filter_close", "index": idx}, style = {"display": "inline-block"}),
                # *create_input("filter_close", idx),
            ], id = {"type": "Trackpy-extended", "index": idx}),
        ], id = {"type": "Trackpy-options", "index": idx}, style = {"display": "none"}),
        html.Div([
            *create_input("min_r", idx, value = 2.2),
            *create_input("max_r", idx, value = 3.5),
            html.Div([
                *create_input("n_radii", idx, value = 10),
                dcc.Checklist(["log_spacing"], id = {"type": "log_spacing", "index": idx}),
                *create_input("overlap_threshold", idx, 0),
            ], id = {"type": "LoG-extended", "index": idx}),
        ], id = {"type": "LoG-options", "index": idx}, style = {"display": "none"}),
        html.Div([
            *create_input("Peak SNR", idx, value = 1.5),
            *create_input("Area SNR", idx, value = 0.3),
            dcc.Checklist(["characterize", "doughnut_correction"], ["characterize", "doughnut_correction"], id = {"type": "char_list", "index": idx}),
            
        ]),
        html.Div([
            *create_input("smoothing_size", idx),
            *create_input("minmass", idx),
            *create_input("max_iterations", idx),
            *create_input("bg_radius", idx),
            *create_input("gap_radius", idx),
            # *create_input("truncate_preprocessed", idx),
            dcc.Checklist(["truncate_preprocessed", "illumination_correction_per_frame"], value = ["truncate_preprocessed"], id = {"type": "shared_toggles", "index": idx}, style = {"display": "inline-block"}),
            *create_input("illumination_sigma", idx),
            *create_input("adaptive_background", idx),
            *create_input("noise_size", idx),
            
        ], id = {"type": "shared_extended", "index": idx}, style = {"display": "none"}),
        html.Div([
            dcc.Graph(figure = px.imshow(np.zeros((100, 100)), color_continuous_scale = "viridis"), id = {"type": "graph", "index": idx}, style = {"display": "inline-block"}),
            dcc.Graph(figure = px.scatter(), id = {"type": "prop_graph", "index": idx}, style = {"display": "inline-block"})
        ]),
        html.Button("Add", id = {"type": "add", "index": idx}),
        html.Button("Remove", id = {"type": "remove", "index": idx}),
    ], id = {"type": "ui", "index": idx})
    return new_ele

import re
pat = re.compile(r"'index': '[A-Za-z0-9]+?'")

class IdOrder(dict):
    def insert_after_key(self, key, newkey):
        to_update_after = self[key]
        for k, val in self.items():
            if val > to_update_after:
                self[k] = val + 1
        self[newkey] = to_update_after + 1
        return to_update_after
    
    def delete_key(self, key):
        to_update_after = self[key]
        for k, val in self.items():
            if val > to_update_after:
                self[k] = val - 1
        del self[key]
        return to_update_after
                

order = IdOrder()
dfs = {}

def adder(children, triggering_idx):
    ident = uuid.uuid4().hex
    if triggering_idx == -1:
        clicked_pos = len(children)
        order[ident] = clicked_pos
        new_ele = create_element(ident)
        
    else:
        clicked_pos = order.insert_after_key(triggering_idx, ident)
        new_ele = children[clicked_pos]
        new_ele = eval(pat.sub(f"'index': '{ident}'", str(new_ele)))
    # new_ele = create_element(ident)
    children.insert(clicked_pos + 1, new_ele)
    return children

def remover(children, triggering_idx):
    idx = order.delete_key(triggering_idx)
    children.pop(idx)
    return children


@app.callback(
    Output({"type": "modify", "index": MATCH}, "n_clicks"),
    Output({"type": "Up to frame", "index": MATCH}, "value"),
    Input({"type": "prev", "index": MATCH}, "n_clicks"),
    Input({"type": "next", "index": MATCH}, "n_clicks"),
    State({"type": "Up to frame", "index": MATCH}, "value"),
    # State({"type": "Up to frame", "index": MATCH}, "n_clicks"),
)
def prev_next(_prev_clicks, _next_clicks, current_frame):
    who_triggered = list(ctx.triggered_prop_ids.values())
    if len(who_triggered) != 1:
        return 0, current_frame 
    if who_triggered[0]["type"] == "prev":
        return 0, current_frame - 1
    if who_triggered[0]["type"] == "next":
        return 0, current_frame + 1

cases = {"add": adder, "remove": remover}

@app.callback(
    Output({"type": "Trackpy-options", "index": MATCH}, "style"),
    Output({"type": "LoG-options", "index": MATCH}, "style"),
    Output({"type": "Trackpy-extended", "index": MATCH}, "style"),
    Output({"type": "LoG-extended", "index": MATCH}, "style"),
    Output({"type": "shared_extended", "index": MATCH}, "style"),
    Input({"type": "flavor", "index": MATCH}, "value"),
    Input({"type": "extended", "index": MATCH}, "value"),
)
def set_visibility(flavor_value, extended_value):
    
    if flavor_value == "Trackpy":
        out = ({"display": "block"}, {"display": "none"})
    if flavor_value == "LoG":
        out = ({"display": "none"}, {"display": "block"})
    
    if extended_value:
        out = (*out, *out, {"display": "block"})
    else:
        out = (*out, {"display": "none"}, {"display": "none"}, {"display": "none"})
    return out

@app.callback(
    Output({"type": "color_options", "index": MATCH}, "style"),
    Input({"type": "color_options_toggle", "index": MATCH}, "value"),
)
def set_color_opt_visibility(color_options_toggle_value):
    # print(color_options_toggle_value)
    if color_options_toggle_value is not None and "Color Options" in color_options_toggle_value:
        return {"display": "block"}
    else:
        return {"display": "none"}
    

@app.callback(
    Output({"type": "graph", "index": MATCH}, "figure"),
    Output({"type": "clip", "index": MATCH}, "content"),
    Output({"type": "var 1", "index": MATCH}, "options"),
    Output({"type": "var 2", "index": MATCH}, "options"),
    Input({"type": "modify", "index": MATCH}, "n_clicks"),
    
    State({"type": "input-path", "index": MATCH}, "value"),
    State({"type": "Up to frame", "index": MATCH}, "value"),
    State({"type": "colorpicker", "index": MATCH}, "value"),
    State({"type": "colorscale", "index": MATCH}, "value"),
    
    State({"type": "flavor", "index": MATCH}, "value"),
    
    State({"type": "diameter", "index": MATCH}, "value"),
    State({"type": "separation", "index": MATCH}, "value"),
    State({"type": "noise_size", "index": MATCH}, "value"),
    State({"type": "smoothing_size", "index": MATCH}, "value"),
    State({"type": "filter_close", "index": MATCH}, "value"),
    
    State({"type": "min_r", "index": MATCH}, "value"),
    State({"type": "max_r", "index": MATCH}, "value"),
    State({"type": "n_radii", "index": MATCH}, "value"),
    State({"type": "log_spacing", "index": MATCH}, "value"),
    State({"type": "overlap_threshold", "index": MATCH}, "value"),
    
    State({"type": "minmass", "index": MATCH}, "value"),
    State({"type": "max_iterations", "index": MATCH}, "value"),
    State({"type": "char_list", "index": MATCH}, "value"),
    State({"type": "bg_radius", "index": MATCH}, "value"),
    State({"type": "gap_radius", "index": MATCH}, "value"),
    State({"type": "Peak SNR", "index": MATCH}, "value"),
    State({"type": "Area SNR", "index": MATCH}, "value"),
    State({"type": "shared_toggles", "index": MATCH}, "value"),
    State({"type": "illumination_sigma", "index": MATCH}, "value"),
    State({"type": "adaptive_background", "index": MATCH}, "value"),
)
def modify(
    _n_clicks,
    
    path,
    frames,
    circle_color,
    color_scale,

    flavor,

    diameter,
    separation,
    noise_size,
    smoothing_size,
    filter_close,

    min_r,
    max_r,
    n_radii,
    log_spacing,
    overlap_threshold,

    minmass,
    max_iterations,
    # characterize,
    # doughnut_correction,
    char_list,
    bg_radius,
    gap_radius,
    snr,
    minmass_snr,
    shared_toggles,
    illumination_sigma,
    adaptive_background,


):
    # print("\n"*5)
    # print(ctx.triggered_prop_ids)
    if char_list is None:
        characterize, doughnut_correction = (None, None)
    else:
        characterize = "characterize" in char_list
        doughnut_correction = "doughnut_correction" in char_list

    if filter_close is None:
        filter_close = False
    else:
        filter_close = "filter_close" in filter_close

    if shared_toggles is None:
        truncate_preprocessed = False
        illumination_correction_per_frame = False
    else:
        truncate_preprocessed = "truncate_preprocessed" in shared_toggles
        illumination_correction_per_frame = "illumination_correction_per_frame" in shared_toggles
    if log_spacing is None:
        log_spacing = False
    else:
        log_spacing = "log_spacing" in log_spacing

    fig, command, df, im_shape = plotter(
        path,
        frames,

        flavor,

        diameter,
        separation,
        noise_size,
        smoothing_size,
        filter_close,

        min_r,
        max_r,
        n_radii,
        log_spacing,
        overlap_threshold,

        minmass,
        max_iterations,
        characterize,
        doughnut_correction,
        bg_radius,
        gap_radius,
        snr,
        minmass_snr,
        truncate_preprocessed,
        illumination_sigma,
        illumination_correction_per_frame,
        adaptive_background,

        circle_color = circle_color["hex"],
        color_scale = color_scale,
    )
    # fig.update_layout(height = 750, width = 750, coloraxis_showscale=False)
    fig.update_layout(height = 750, width = 750, margin=dict(l=0, r=0, b=0, t=0), coloraxis_showscale=False)
    scatter_df = df[df["frame"] == frames - 1].drop(columns = ["frame"])
    iter_df = scatter_df.copy()
    # print("before: ", iter_df)
    # scatter_df = scatter_df.drop(columns = ["x", "y"])
    try:
        # print(scatter_df["Rg"])
        iter_df["10*Rg"] = iter_df["Rg"] * 10
        iter_df["100*ecc"] = iter_df["ecc"] * 100
        iter_df = iter_df.drop(columns = ["Rg", "ecc"])
    except KeyError:
        pass
    # print("after: ", iter_df)
    it = iter(iter_df.drop(columns = ["x", "y"]))
    col = next(it)
    hovertext = f"{col} = " + np.round(iter_df[col], 1).astype(str) + "<br>"
    for col in it:
        # print(col)
        # print(np.round(scatter_df[col], 1).astype(str))
        # print(col + scatter_df[col].astype(str))
        hovertext += f"{col} = " + np.round(iter_df[col], 1).astype(str) + "<br>"
    
    # fig.update_xaxes(autorange = False)
    # fig.update_yaxes(autorange = False)
    # full_fig = fig.full_figure_for_development()
    # print(full_fig.layout.xaxis.range)
    fig.add_trace(go.Scatter(x = scatter_df["x"], y = scatter_df["y"], text = hovertext, mode = "markers", marker_color = 'rgba(0,0,0,0)'))
    fig.update_xaxes(range = [0, im_shape[1]])
    fig.update_yaxes(range = [0, im_shape[0]])
    options1 = list(scatter_df.columns.drop(["x", "y"]))
    options2 = ["None"] + options1

    who_triggered = list(ctx.triggered_prop_ids.values())
    triggering = who_triggered[0]
    triggering_idx = triggering["index"]
    dfs[triggering_idx] = df

    return fig, command, options1, options2

@app.callback(
    Output({"type": "prop_graph", "index": MATCH}, "figure"),
    
    Input({"type": "var 1", "index": MATCH}, "value"),
    Input({"type": "var 2", "index": MATCH}, "value"),
    # State({"type": "prop_graph", "index": MATCH}, "figure"),
)
def update_prop_graph(
    var1, var2,
    # old_graph,
):
    who_triggered = list(ctx.triggered_prop_ids.values())
    if len(who_triggered) == 0:
        return px.scatter()
    triggering = who_triggered[0]
    triggering_idx = triggering["index"]
    df = dfs[triggering_idx]
    # print(df)
    if var1:
        if var2 is not None and var2 != "None":
            prop_graph = px.scatter(x = df[var1], y = df[var2])
        else:
            prop_graph = px.histogram(x = df[var1])
            mean = df[var1].mean()
            prop_graph.add_vline(x = mean)
            median = df[var1].median()
            if mean != 0 and abs(1 - median/mean) > 0.2:
                prop_graph.add_vline(x = median, line_dash = "dash")
        
        prop_graph.update_layout(height = 600, width = 900, margin=dict(l=0, r=0, b=0, t=0))
    else:
        return px.scatter()
    return prop_graph

@app.callback(
    Output('dropdown-container', 'children'),
    Input({"type": "add", "index": ALL}, 'n_clicks'),
    Input({"type": "remove", "index": ALL}, 'n_clicks'),
    State('dropdown-container', 'children'),
)
def add_remove(_add_n_clicks, _remove_n_clicks, children):
    who_triggered = list(ctx.triggered_prop_ids.values())
    if len(who_triggered) != 1:
        return children
    # print("\n"*5)
    triggering = who_triggered[0]
    triggering_type = triggering["type"]
    triggering_idx = triggering["index"]
    cases[triggering_type](children, triggering_idx)
    return children


if __name__ == '__main__':
    app.run_server(debug=True)
