import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import os
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# Utility functions from the first script
def get_files_in_case_study(case_study):
    path = os.path.join(case_studies_path, case_study)
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Define the path to the case studies folder
case_studies_path = 'project/case_studies'

# Get a list of all case studies
case_studies_files = [d for d in os.listdir(case_studies_path) if os.path.isdir(os.path.join(case_studies_path, d))]


# Load data from CSV files
def load_data(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            case_study = filename.replace('.csv', '')
            df = pd.read_csv(os.path.join(directory, filename))
            for col in df.columns:
                if '_optimized' in col:
                    df[col] = df[col].astype(bool)
            data[case_study] = df
    return data

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
data = load_data('./dataCFD/Python_files')
case_studies = list(data.keys())

cpp_dir = 'project/source_code_cpp'
python_dir = 'project/source_code_python'
cpp_files = list_files(cpp_dir)
python_files = list_files(python_dir)

def create_bar_plot(data):
    first_key = next(iter(data))
    categories = data[first_key].columns.tolist()
    values = data[first_key].iloc[0].tolist()
    max_y_value = max(values)
    y_range = [0, max_y_value * 1.1]
    bar_plot = px.bar(x=categories, y=values, title="Performance Comparison of Different Cases",
                      labels={'x': 'Input Decks', 'y': 'Values'},
                      color=values, color_continuous_scale='Reds',
                      text=values, range_y=y_range)
    bar_plot.update_traces(textposition='outside')
    bar_plot.update_layout(
        xaxis_title="Input Decks",
        yaxis_title="Values",
        barmode='group',
        template='plotly_white',
        coloraxis_colorbar=dict(
            title='Values',
            tickvals=[],
            ticktext=[]
        )
    )
    return bar_plot


# Function to load data from Python_files
def load_python_files_data(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            data[filename.replace('.csv', '')] = df.iloc[0].tolist()  # first row values
    return data

# Load data
cpp_data2 = load_data('./dataCFD/Cpp_files')
python_data2 = load_python_files_data('./dataCFD/Python_files')

def create_bar_plot2(cpp_data, python_data):
    first_key = next(iter(cpp_data))
    categories = cpp_data[first_key].columns.tolist()
    cpp_values = cpp_data[first_key].iloc[0].tolist()
    python_values = [python_data.get(cat, [0])[-2] for cat in categories]

    # Calculate ratios
    ratios = [py / cpp if cpp != 0 else 0 for cpp, py in zip(cpp_values, python_values)]

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Categories': categories,
        'C++': cpp_values,
        'Python': python_values,
        'Ratios': ratios
    })

    # Create the bar plot
    fig = px.bar(df, x='Categories', y=['C++', 'Python'], barmode='group',
                 title="Performance Comparison of Different Cases",
                 labels={'value': 'Values', 'variable': 'Language'},
                 color_discrete_sequence=['#ff7f0e', '#1f77b4'])
    
    fig.update_traces(textposition='outside')

    # Add ratio text above the bars
    for i, cat in enumerate(categories):

        # Add ratio text above the bars
        fig.add_annotation(
            x=cat,
            y=cpp_values[i] + python_values[i],
            text=f"{ratios[i]:.2f}x",
            showarrow=False,
            font=dict(size=16, color='black', family="Arial Bold"),  # Increase font size and set to bold
            xanchor='center',
            yanchor='bottom'
        )

    fig.update_layout(
        xaxis_title="Case Studies",
        yaxis_title="Seconds",
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_cpp_rows_comparison_plot(cpp_data):
    first_key = next(iter(cpp_data))
    categories = cpp_data[first_key].columns.tolist()
    
    # Extract values from the first and second rows of cpp_data
    cpp_values_first_row = cpp_data[first_key].iloc[0].tolist()
    cpp_values_second_row = cpp_data[first_key].iloc[1].tolist()

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Categories': categories * 2,  # Repeat categories for each row
        'Values': cpp_values_first_row + cpp_values_second_row,
        'Language': ['C++'] * len(categories) + ['Python'] * len(categories)
    })

    # Create the bar plot
    fig = px.bar(df, x='Categories', y='Values', color='Language', barmode='group',
                 title="Performance Comparison between C++ and fully optimized Python code",
                 labels={'Values': 'Seconds', 'variable': 'Language'},
                 color_discrete_sequence=['#ff7f0e','#1f77b4'])

    fig.update_layout(
        xaxis_title="Case Studies",
        yaxis_title="Seconds",
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Assuming cpp_data2 is already loaded
cpp_rows_comparison_plot = create_cpp_rows_comparison_plot(cpp_data2)



bar_plot = create_bar_plot2(cpp_data2, python_data2)


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Performance Profiling & Optimization Dashboard", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("0. Project Overview"),
                        dcc.Markdown("""
                        ### Purpose
                        This project involved translating a C++ CFD solver, **fluidchen**, into Python and optimizing the Python code for performance. The solver focuses on the incompressible Navier-Stokes equations (NSE) for single-phase flow in 2D, with applications ranging from weather prediction to hemodynamics. The primary goal was to measure and improve the execution times of key functions across various case studies.

                        ### Process
                        1. **Translation**: The original C++ code was translated into Python, ensuring functionality remained consistent.
                        2. **Measurement**: Execution times were recorded for the original Python code to establish a baseline performance.
                        3. **Optimization**: Several optimization techniques were applied to the Python code, including vectorization and the use of Numba for just-in-time compilation.
                        4. **Analysis**: The impact of each optimization was analyzed by comparing the execution times before and after optimization.
                        5. **Final Result**: The plot below visualizes execution times, allowing you to compare the performance of the optimized Python code against the C++ implementation.
                        6. **Reproducibility**: This section provides instructions and tools to reproduce the simulations and analysis.

                        ### Goals
                        - Improve the performance of the Python code to match or exceed the efficiency of the original C++ program.
                        - Provide a comprehensive dashboard to visualize the optimization process and results, enabling easy comparison between the optimized Python code and the C++ implementation.
                        """)
                    ])
                ], width=12)
            ], className='mt-3'),
        ], width=12)
    ], className='mt-3'),
    dbc.Row([
        dbc.Col([
            html.H2("1. Translation: Code Comparison Tool"),
            html.P("""
                This tool allows you to compare the original C++ code with its translated Python counterpart. By selecting corresponding files from the dropdown menus, you can view and analyze the code side by side. This comparison helps in verifying the consistency of functionality between the two implementations and understanding the translation outcomes.
            """),
            html.Div([
                html.Div([
                    html.Label("Select C++ File"),
                    dcc.Dropdown(id='cpp-file', options=[{'label': f, 'value': f} for f in cpp_files])
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Select Python File"),
                    dcc.Dropdown(id='python-file', options=[{'label': f, 'value': f} for f in python_files])
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'}),
            ]),
            html.Div([
                html.Div(id='cpp-content', style={
                    'width': '48%', 'display': 'inline-block', 'vertical-align': 'top',
                    'font-family': 'monospace', 'padding': '10px', 'border': '1px solid #ccc',
                    'border-radius': '5px', 'overflow': 'auto', 'max-height': '500px'
                }),
                html.Div(id='python-content', style={
                    'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '4%',
                    'font-family': 'monospace', 'padding': '10px', 'border': '1px solid #ccc',
                    'border-radius': '5px', 'overflow': 'auto', 'max-height': '500px'
                })
            ], style={'marginTop': '20px'})
        ], width=12)
    ], className='mt-3'),
    dbc.Row([
        dbc.Col([
            html.H2("2. Measurement: Performance Baseline"),
            html.P("""
                To establish a performance baseline, execution times were recorded for the original Python code across various input decks. This baseline is crucial for understanding the impact of subsequent optimizations. The bar plot below visualizes these execution times, allowing you to compare the performance of the original Python code against the C++ implementation and track the effectiveness of our optimization efforts.
            """),
            dcc.Graph(figure=bar_plot, id='new-bar-plot', config={'displayModeBar': False})
        ], width=12)
    ], className='mt-3'),
    dbc.Row([
        dbc.Col([
            html.H2("3. Optimization: Performance Enhancement"),
            html.P("""
                This section provides detailed insights into the optimizations applied to various functions in the Python code. After a short explanation of the applied optimization techniques you can explore the specific optimization techniques used for an selected function, such as vectorization, Just-In-Time (JIT) compilation with Numba, and algorithmic improvements by selecting. The explanations below will help you understand how these optimizations enhance performance and efficiency.
            """),
            html.Div([
                html.H3("Optimization Techniques Applied"),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Vectorization"),
                            dcc.Markdown("""
                            - Applies operations to entire arrays.
                            - Leverages modern CPUs for parallel computation.
                            - NumPy optimizes this in Python.

                            **Example:**
                            ```python
                            import numpy as np
                            
                            arr = np.array([1, 2, 3])
                            result = arr * 2  # Vectorized multiplication
                            ```
                            """)
                        ], width=6),
                        dbc.Col([
                            html.H4("Just-In-Time (JIT) Compilation with Numba"),
                            dcc.Markdown("""
                            - Translates Python functions to optimized machine code at runtime.
                            - Uses LLVM, achieving C-like performance without switching languages.

                            **Example:**
                            ```python
                            from numba import jit
                            
                            @jit(nopython=True)
                            def sum_optimized(arr):
                                return sum(arr)
                            ```
                            """)
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Profiling and Code Optimization"),
                            dcc.Markdown("""
                            - Identifies bottlenecks.
                            - Enables optimization through algorithmic improvements and efficient data structures.
                            """)
                        ], width=12)
                    ])
                ], className='mb-3'),
            ], className='mb-3'),

            dcc.Dropdown(
                id='explanation-dropdown',
                options=[
                    {'label': 'calc_Fluxes', 'value': 'calc_Fluxes'},
                    {'label': 'sor_Solve', 'value': 'sor_Solve'},
                    {'label': 'calc_temp', 'value': 'calc_temp'},
                    {'label': 'calc_Vel', 'value': 'calc_Vel'},
                    {'label': 'calc_Rs', 'value': 'calc_Rs'},
                    {'label': 'applyPressure', 'value': 'applyPressure'},
                    {'label': 'solve_naive', 'value': 'solve_naive'},
                    {'label': 'solve_numba', 'value': 'solve_numba'}
                ],
                value=None,
                placeholder="Select function to explain",
                className='mb-3'
            ),
            dbc.Collapse(
                dbc.Card([
                    dbc.CardHeader("Function Explanation"),
                    dbc.CardBody(id='explanation-content')
                ]),
                id="explanation-collapse",
                is_open=False
            )
        ], width=12)
    ], className='mt-3'),
    dbc.Row([
        dbc.Col([
            html.H2("4. Analysis: Optimization Impact"),
            html.P("""
                This section allows you to analyze the impact of each optimization by comparing the execution times before and after optimization. Select a case study and functions to be used in their optimized state, and observe the changes in the plots below. Additionally, you can view the full performance report outputted by the Python tool pyinstrument.
            """),
            dcc.Dropdown(
                id='case-study-dropdown',
                options=[{'label': case, 'value': case} for case in case_studies],
                value=case_studies[0],
                placeholder="Select an input deck",
                className='mb-3'
            ),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='non-optimized-plot', config={'displayModeBar': False})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='optimized-plot', config={'displayModeBar': False})
                ], width=6)
            ]),
            dcc.Dropdown(
                id='function-dropdown',
                options=[{'label': col, 'value': col} for col in data[case_studies[0]].columns if '_optimized' not in col and col != 'simulation_time' and col != "html_file"],
                value=[],
                multi=True,
                placeholder="Select which optimized functions to use",
                className='mb-3'
            ),
            html.Div([
                dcc.Checklist(
                    id='show-html-report',
                    options=[{'label': 'Show HTML Report', 'value': 'show'}],
                    value=[],
                    className='mb-3'
                )
            ]),
            html.Div(id='html-output'),
            dbc.Row([
                dbc.Col([
                    html.H3("5. Final Result", className="mb-4"),
                    html.P("""
                        The plot below visualizes execution times, allowing you to compare the performance of the optimized Python code against the C++ implementation.
                    """),
                    dcc.Graph(figure=cpp_rows_comparison_plot, id='cpp-rows-comparison-plot', config={'displayModeBar': False})
                ], width=12)
            ], className='mt-3')
        ], width=12)
    ], className='mt-3'),
    dbc.Row([
        dbc.Col([
            html.H2("6. Reproducibility"),
            html.P("""
    This section provides tools and instructions to ensure the reproducibility of the simulations and analysis. The "Input Deck Explorer" allows you to investigate the input decks used for the simulations, enabling you to understand the configuration and parameters of each case study.
"""),
            html.Div([
                html.H3("Input Deck Explorer", id="case-study-explorer"),
                dcc.Dropdown(
                    id='case-study-selector',
                    options=[{'label': case_study, 'value': case_study} for case_study in case_studies_files],
                    value=case_studies_files[0] if case_studies_files else None,
                    placeholder="Select an input deck"
                ),
                dcc.Dropdown(
                    id='file-selector',
                    placeholder="Select a file"
                ),
                html.Div(id='file-display')
            ]),
            html.Br(),
            html.Div([
                html.H3("Instructions for Running the Simulation"),
            ]),
            html.P("""
                   Here you can find how to execute the simulation using the provided Python script and the C++ program. These instructions will guide you through running the simulation, checking the output, and viewing the results."""),
            dbc.Row([
                dbc.Col([
                    html.H4("Python Script Execution"),
                    dcc.Markdown("""
                    To run the simulation using the provided Python script, follow these steps:

                    1. **Prepare the Input File**: Ensure you have an input file ready. This file should be in a format that the `Case` class can understand (e.g., `input_data.dat`).

                    2. **Run the Script**: Use the command line to run the script with the required arguments. The script expects three arguments:
                       - The path to the input file.
                       - The base name for the output HTML file.
                       - A code or identifier to append to the output HTML file name.

                       Here is the general format of the command:
                       ```sh
                       mpiexec -n <number_of_processes> python <path_to_script> <path_to_input_file> <output_base_name> <code>
                       ```
                    For example, if you have the script saved as fluidchen.py and the input file as input_data.dat in the current directory, and you want to use 4 MPI processes, the command would look like this:

                    ```sh
                    mpiexec -n 4 python fluidchen.py input_data.dat output_file my_code
                    ```

                    3. **Check the Output**: After running the command, the script will:
                       - Run the simulation using the provided input file.
                       - Generate an HTML profiling report using pyinstrument.
                       - Save the HTML report to a file named `output_file_my_code.html`.

                    4. **View the Results**: You can view the HTML report in any web browser by opening the generated HTML file.
                    """)
                ], width=6),
                dbc.Col([
                    html.H4("C++ Program Execution"),
                    dcc.Markdown("""
                    To execute the C++ program for the simulation, follow these steps:

                    1. **Compile the Program**: Ensure you have a C++ compiler and MPI installed. Compile the program using a command like:
                       ```sh
                       mpic++ -o fluidchen main.cpp -I/path/to/Case.hpp
                       ```

                    2. **Run the Program**: Use the command line to run the compiled program with the required arguments. The program expects at least one argument:
                       - The path to the input file.

                       Here is the general format of the command:
                       ```sh
                       mpiexec -n <number_of_processes> ./fluidchen <path_to_input_file>
                       ```
                    For example, if you have the compiled program saved as fluidchen and the input file as input_data.dat in the current directory, and you want to use 4 MPI processes, the command would look like this:

                    ```sh
                    mpiexec -n 4 ./fluidchen input_data.dat
                    ```

                    3. **Check the Output**: After running the command, the program will:
                       - Run the simulation using the provided input file.
                       - Generate a log file named `Simulation_log.txt`.
                       - Output the running time of the simulation.

                    4. **View the Results**: You can view the log file in a text editor to see detailed logs and running time.
                    """)
                ], width=6)  # Adjust width as needed
            ], className='mt-3')
        ], width=12)
    ], className='mt-3')
], fluid=True)

# Callbacks from the first script
@app.callback(
    Output('file-selector', 'options'),
    [Input('case-study-selector', 'value')]
)
def set_files_options(selected_case_study):
    if selected_case_study:
        files = get_files_in_case_study(selected_case_study)
        return [{'label': file, 'value': file} for file in files]
    return []

@app.callback(
    Output('file-display', 'children'),
    [Input('file-selector', 'value'),
     Input('case-study-selector', 'value')]
)
def display_file_content(selected_file, selected_case_study):
    if selected_file and selected_case_study:
        file_path = os.path.join(case_studies_path, selected_case_study, selected_file)
        with open(file_path, 'r') as file:
            content = file.read()
        return html.Pre(content)
    return "Please select a file to display its content."

@app.callback(
    [Output('cpp-content', 'children'),
     Output('python-content', 'children')],
    [Input('cpp-file', 'value'),
     Input('python-file', 'value')]
)
def display_files(cpp_file, python_file):
    cpp_content = ""
    python_content = ""

    if cpp_file:
        with open(os.path.join(cpp_dir, cpp_file), 'r') as file:
            cpp_content = dcc.Markdown(f'```cpp\n{file.read()}\n```')

    if python_file:
        with open(os.path.join(python_dir, python_file), 'r') as file:
            python_content = dcc.Markdown(f'```python\n{file.read()}\n```')

    return cpp_content, python_content

def all_optimized_false(row):
    optimized_cols = [col for col in row.index if '_optimized' in col]
    return all(row[col] == False for col in optimized_cols)

@app.callback(
    Output('non-optimized-plot', 'figure'),
    [Input('case-study-dropdown', 'value')]
)
def update_non_optimized_plot(case_study):
    if case_study is None:
        return {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [
                    {
                        'text': '',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 20}
                    }
                ]
            }
        }

    df = data[case_study]
    simulation_time = df.iloc[0, -2]
    non_optimized_df = df[df.apply(all_optimized_false, axis=1)]
    first_six_columns = non_optimized_df.iloc[:, :6].sort_values(by=0, axis=1, ascending=False).T

    max_y_value = first_six_columns[0].max()
    y_range = [0, max_y_value * 1.1]

    fig = px.bar(first_six_columns, x=first_six_columns.index, y=first_six_columns[0],
                 title=f"Fully Non-Optimized Functions (Simulation Time: {simulation_time} seconds)",
                 labels={'x': 'Functions', 'y': 'Seconds'},
                 color=first_six_columns[0], color_continuous_scale='Reds',
                 text=first_six_columns[0], range_y=y_range)
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Functions",
        yaxis_title="Seconds",
        barmode='group',
        template='plotly_white',
        coloraxis_colorbar=dict(
            title='[s]',
            tickvals=[],
            ticktext=[]
        )
    )
    
    return fig

@app.callback(
    Output('optimized-plot', 'figure'),
    [Input('case-study-dropdown', 'value'),
     Input('function-dropdown', 'value')]
)
def update_optimized_plot(case_study, selected_functions):
    if case_study is None:
        return {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [
                    {
                        'text': '',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 20}
                    }
                ]
            }
        }

    df = data[case_study]

    filtered_df = df.copy()
    for func in selected_functions:
        optimized_col = f'{func}_optimized'
        if optimized_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[optimized_col] == True]
        else:
            return {
                'data': [],
                'layout': {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [
                        {
                            'text': 'Column not found',
                            'xref': 'paper',
                            'yref': 'paper',
                            'showarrow': False,
                            'font': {'size': 20}
                        }
                    ]
                }
            }

    for col in filtered_df.columns:
        if '_optimized' in col and col.replace('_optimized', '') not in selected_functions:
            filtered_df = filtered_df[filtered_df[col] == False]

    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [
                    {
                        'text': 'No measurements made',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 20}
                    }
                ]
            }
        }

    simulation_time = filtered_df.iloc[0, -2]

    function_columns = [col for col in filtered_df.columns if '_optimized' not in col and col != 'simulation_time']
    filtered_df = filtered_df[function_columns].reset_index(drop=True)

    filtered_df = filtered_df.iloc[:, :-1]
    threshold = 0.1 * simulation_time
    filtered_df01 = filtered_df.loc[:, filtered_df.iloc[0] >= threshold]

    if len(filtered_df01.columns) < 4:
        threshold = 0.01 * simulation_time
        filtered_df001 = filtered_df.loc[:, filtered_df.iloc[0] >= threshold]

        if len(filtered_df001.columns) < 4:
            threshold = 0.001 * simulation_time
            filtered_df = filtered_df.loc[:, filtered_df.iloc[0] >= threshold]
        else:
            filtered_df = filtered_df001
    else:
        filtered_df = filtered_df01

    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [
                    {
                        'text': 'No measurements made',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 28}
                    }
                ]
            }
        }

    filtered_df = filtered_df.sort_values(by=0, axis=1, ascending=False).T

    if 0 not in filtered_df.columns:
        return {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [
                    {
                        'text': 'No measurements made',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 28}
                    }
                ]
            }
        }

    max_y_value = filtered_df[0].max()
    y_range = [0, max_y_value * 1.1]

    fig = px.bar(filtered_df, x=filtered_df.index, y=filtered_df[0],
                 title=f"Selected Functions Optimized (Simulation Time: {simulation_time} seconds)",
                 labels={'x': 'Functions', 'y': 'Seconds'},
                 color=filtered_df[0], color_continuous_scale='Reds',
                 text=filtered_df[0], range_y=y_range)
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Functions",
        yaxis_title="Seconds",
        barmode='group',
        template='plotly_white',
        coloraxis_colorbar=dict(
            title='[s]',
            tickvals=[],
            ticktext=[]
        )
    )
    
    return fig

@app.callback(
    Output('html-output', 'children'),
    [Input('case-study-dropdown', 'value'),
     Input('function-dropdown', 'value'),
     Input('show-html-report', 'value')]
)
def display_html_report(case_study, selected_functions, show_html_report):
    if case_study is None or not selected_functions or 'show' not in show_html_report:
        return ""

    df = data[case_study]

    filtered_df = df.copy()
    for func in selected_functions:
        optimized_col = f'{func}_optimized'
        if optimized_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[optimized_col] == True]
        else:
            return "Column not found"

    for col in filtered_df.columns:
        if '_optimized' in col and col.replace('_optimized', '') not in selected_functions:
            filtered_df = filtered_df[filtered_df[col] == False]

    if filtered_df.empty:
        return "No measurements made"

    html_file_path = filtered_df.iloc[0, -1]
    if pd.isna(html_file_path):
        return "No HTML file available for this row."

    try:
        with open(html_file_path, 'r') as file:
            result_html = file.read()
    except FileNotFoundError:
        return f"HTML file not found: {html_file_path}"

    return html.Iframe(srcDoc=result_html, style={"width": "100%", "height": "600px"})


@app.callback(
    [Output("explanation-collapse", "is_open"),
     Output("explanation-content", "children")],
    [Input("explanation-dropdown", "value")],
)
def toggle_collapse(selected_value):
    if selected_value is None:
        return False, None
    elif selected_value == 'calc_Fluxes':
        explanation = """
        ### Explanation for Flux Calculation Functions

        The flux calculation functions are responsible for computing the fluxes of the fluid variables, which are essential for the numerical solution of the Navier-Stokes equations. These functions are part of the `Fields` class and provide three different implementations:

        1. **Naive Implementation (`calculate_fluxes_naive`)**: This method uses explicit Python loops to iterate over each fluid cell and compute the fluxes based on the convection and diffusion terms. It does not use any optimization techniques.
        2. **Vectorized Implementation (`calculate_fluxes_vectorized`)**: This method uses NumPy's array operations to handle the flux calculations more efficiently. It leverages vectorized operations for parallel processing of array elements.
        3. **Numba-Optimized Implementation (`calculate_fluxes_numba`)**: This method uses the Numba library to compile the flux calculation with just-in-time (JIT) compilation, which can significantly speed up the execution.

        Here is the code listing for each implementation:
        """
        code_naive = """
        ```python
        def calculate_fluxes_naive(self, grid: Grid):
            for elem in grid.fluid_cells():
                i = elem.i()
                j = elem.j()
                if i != 0 and i != grid.size_x() + 1 and j != grid.size_x() + 1:
                    a = Discretization.convection_u(self._U, self._V, i, j)
                    self._F[i, j] = self._U[i, j] + self._dt * ((self._nu * Discretization.laplacian(self._U, i, j)) - a + self._gx)
                    b = Discretization.convection_v(self._U, self._V, i, j)
                    self._G[i, j] = self._V[i, j] + self._dt * ((self._nu * Discretization.laplacian(self._V, i, j)) - b + self._gy)
                    if grid.getUseTemp():
                        self.set_f(i, j, self._F[i, j] - self._gx * self._dt)
                        self.set_g(i, j, self._G[i, j] - self._gy * self._dt)
                        self.set_f(i, j, self._F[i, j] - self._beta * (self._dt / 2) * (self._T[i, j] + self._T[i + 1, j]) * self._gx)
                        self.set_g(i, j, self._G[i, j] - self._beta * (self._dt / 2) * (self._T[i, j] + self._T[i, j + 1]) * self._gy)
        ```
        """
        code_vectorized = """
        ```python
        def calculate_fluxes_vectorized(self, grid: Grid):
            U, V = self._U, self._V
            dt, nu = self._dt, self._nu
            gx, gy = self._gx, self._gy
            T = self._T if grid.getUseTemp() else None
            beta = self._beta if grid.getUseTemp() else 0

            F = np.copy(U)
            G = np.copy(V)

            fluid_mask = grid.get_fluid_cells_mask()

            conv_u = Discretization.optimized_convection_u(U, V, grid)
            conv_v = Discretization.optimized_convection_v(U, V, grid)

            laplacian_U = Discretization.optimized_laplacian(U)
            laplacian_V = Discretization.optimized_laplacian(V)

            F[fluid_mask] = U[fluid_mask] + dt * (nu * laplacian_U[fluid_mask] - conv_u[fluid_mask] + gx)
            G[fluid_mask] = V[fluid_mask] + dt * (nu * laplacian_V[fluid_mask] - conv_v[fluid_mask] + gy)

            if grid.getUseTemp():
                F[fluid_mask] -= gx * dt
                G[fluid_mask] -= gy * dt

                T_shifted_right = np.roll(T, -1, axis=0)
                T_shifted_up = np.roll(T, -1, axis=1)

                F[fluid_mask] -= beta * (dt / 2) * (T[fluid_mask] + T_shifted_right[fluid_mask]) * gx
                G[fluid_mask] -= beta * (dt / 2) * (T[fluid_mask] + T_shifted_up[fluid_mask]) * gy

            self._F = F
            self._G = G
        ```
        """
        code_numba = """
        ```python
        @jit(nopython=True)
        def _calculate_fluxes_numba(U, V, F, G, T, dt, nu, gx, gy, alpha, beta, fluid_mask, dx, dy, gamma, use_temp):
            for i in range(1, U.shape[0] - 1):
                for j in range(1, U.shape[1] - 1):
                    if fluid_mask[i, j]:
                        a = _convection_u_numba(U, V, i, j, dx, dy, gamma)
                        F[i, j] = U[i, j] + dt * ((nu * _laplacian_numba(U, i, j, dx, dy)) - a + gx)
                        b = _convection_v_numba(U, V, i, j, dx, dy, gamma)
                        G[i, j] = V[i, j] + dt * ((nu * _laplacian_numba(V, i, j, dx, dy)) - b + gy)

                        if use_temp:
                            F[i, j] -= gx * dt
                            G[i, j] -= gy * dt
                            F[i, j] -= beta * (dt / 2) * (T[i, j] + T[i + 1, j]) * gx
                            G[i, j] -= beta * (dt / 2) * (T[i, j] + T[i, j + 1]) * gy
            return F, G

        def calculate_fluxes_numba(self, grid: Grid):
            self._F, self._G = _calculate_fluxes_numba(self._U, self._V, self._F, self._G, self._T, self._dt, self._nu, self._gx, self._gy, self._alpha, self._beta, grid.get_fluid_cells_mask(), grid.dx(), grid.dy(), Discretization._gamma, grid.getUseTemp())
        ```
        """
        return True, html.Div([
            dcc.Markdown(explanation),
            dbc.Row([
                dbc.Col(dcc.Markdown(code_naive), width=4),
                dbc.Col(dcc.Markdown(code_vectorized), width=4),
                dbc.Col(dcc.Markdown(code_numba), width=4)
            ])
        ])
    elif selected_value == 'sor_Solve':
        explanation = """
        ### Explanation for `sor_Solve` function

        The `sor_Solve` function is part of the `SOR` class and is responsible for solving the pressure Poisson equation using the Successive Over-Relaxation (SOR) method. This method is used to iteratively solve the system of linear equations arising from the discretization of the Poisson equation.

        The `SOR` class provides three different implementations of the SOR solver:

        1. **Naive Implementation (`solve_naive`)**: This method uses a simple loop to iterate over each fluid cell and update the pressure field. It does not use any optimization techniques and is intended for comparison with the optimized versions.
        2. **Numba-Optimized Implementation (`solve_numba`)**: This method uses the Numba library to compile the SOR algorithm with just-in-time (JIT) compilation, which can significantly speed up the execution. This method precomputes some values and uses a helper function to apply the SOR update rule.

        Here is the code listing for both `solve_naive` and `solve_numba`:
        """
        code_solve_naive = """
        ```python
        def solve_naive(self, field, grid):
            dx = grid.dx()
            dy = grid.dy()

            coeff = self._omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))

            for currentCell in grid.fluid_cells():
                i = currentCell.i()
                j = currentCell.j()

                if i != 0 and i != grid.size_x() + 1 and j != 0 and j != grid.size_y() + 1:
                    field.set_p(i, j, (1.0 - self._omega) * field.p(i, j) + coeff * (Discretization.sor_helper(field.p_matrix(), i, j) - field.rs(i, j)))

            rloc = 0.0

            for currentCell in grid.fluid_cells():
                i = currentCell.i()
                j = currentCell.j()

                if i != 0 and i != grid.size_x() + 1 and j != 0 and j != grid.size_y() + 1:
                    val = Discretization.laplacian(field.p_matrix(), i, j) - field.rs(i, j)
                    rloc += (val * val)

            return rloc
        ```
        """
        code_solve_numba = """
        ```python
        @jit(nopython=True)
        def solve_n(P, RS, fluid_mask, coeff, omega, dx, dy):
            for i in range(1, P.shape[0] - 1):
                for j in range(1, P.shape[1] - 1):
                    if fluid_mask[i, j]:
                        P[i, j] = (1.0 - omega) * P[i, j] + coeff * (_sor_helper_n(P, i, j, dx, dy) - RS[i, j])

            rloc = 0.0

            for i in range(1, P.shape[0] - 1):
                for j in range(1, P.shape[1] - 1):
                    if fluid_mask[i, j]:
                        val = _laplacian_numba(P, i, j, dx, dy) - RS[i, j]
                        rloc += (val * val)

            return rloc

        def solve_numba(self, field, grid):
            dx = grid.dx()
            dy = grid.dy()

            coeff = self._omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))

            P = field.p_matrix()
            RS = field.rs_matrix()
            fluid_mask = grid.get_fluid_cells_mask()

            return solve_n(P, RS, fluid_mask, coeff, self._omega, dx, dy)
        ```
        """
        return True, html.Div([
            dcc.Markdown(explanation),
            dbc.Row([
                dbc.Col(dcc.Markdown(code_solve_naive), width=6),
                dbc.Col(dcc.Markdown(code_solve_numba), width=6)
            ])
        ])
    elif selected_value == 'calc_temp':
        explanation = """
        ### Explanation for Temperature Calculation Functions

        The temperature calculation functions are responsible for updating the temperature field based on the fluid flow and thermal conditions. These functions are part of the `Fields` class and provide three different implementations:

        1. **Naive Implementation (`calculate_temperatures_naive`)**: This method uses explicit Python loops to iterate over each fluid cell and update the temperature based on the Laplacian and convective terms. It does not use any optimization techniques.
        2. **Vectorized Implementation (`calculate_temperatures_vectorized`)**: This method uses NumPy's array operations to handle the temperature calculations more efficiently. It leverages vectorized operations for parallel processing of array elements.
        3. **Numba-Optimized Implementation (`calculate_temperatures_numba`)**: This method uses the Numba library to compile the temperature calculation with just-in-time (JIT) compilation, which can significantly speed up the execution.

        Here is the code listing for each implementation:
        """
        code_naive = """
        ```python
        def calculate_temperatures_naive(self, grid):
            _Ttmp = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
            for elem in grid.fluid_cells():
                i = elem.i()
                j = elem.j()
                if i != 0 and j != 0 and i != grid.size_x() + 1 and j != grid.size_y() + 1:
                    _Ttmp[i, j] = self._T[i, j] + self._dt * (
                        self._alpha * Discretization.laplacian(self._T, i, j)
                        - Discretization.convection_t(self._U, self._V, self._T, i, j)
                    )
            self._T = _Ttmp
        ```
        """
        code_vectorized = """
        ```python
        def calculate_temperatures_vectorized(self, grid):
            fluid_mask = grid.get_fluid_cells_mask()
            _Ttmp = np.zeros_like(self._T)
            _Ttmp[fluid_mask] = self._T[fluid_mask] + self._dt * (
                self._alpha * Discretization.optimized_laplacian(self._T)[fluid_mask] -
                Discretization.optimized_convection_t(self._U, self._V, self._T)[fluid_mask]
            )
            self._T = _Ttmp
        ```
        """
        code_numba = """
        ```python
        @jit(nopython=True)
        def _calculate_temperatures_numba(U, V, T, Ttmp, dt, alpha, fluid_mask, dx, dy, gamma):
            for i in range(1, T.shape[0] - 1):
                for j in range(1, T.shape[1] - 1):
                    if fluid_mask[i, j]:
                        Ttmp[i, j] = T[i, j] + dt * (
                            alpha * _laplacian_numba(T, i, j, dx, dy)
                            - _convection_t_numba(U, V, T, i, j, dx, dy, gamma)
                        )
            return Ttmp

        def calculate_temperatures_numba(self, grid):
            _Ttmp = np.zeros((grid.size_x() + 2, grid.size_y() + 2))
            _Ttmp = _calculate_temperatures_numba(self._U, self._V, self._T, _Ttmp, self._dt, self._alpha, grid.get_fluid_cells_mask(), grid.dx(), grid.dy(), Discretization._gamma)
            self._T = _Ttmp
        ```
        """
        return True, html.Div([
            dcc.Markdown(explanation),
            dbc.Row([
                dbc.Col(dcc.Markdown(code_naive), width=4),
                dbc.Col(dcc.Markdown(code_vectorized), width=4),
                dbc.Col(dcc.Markdown(code_numba), width=4)
            ])
        ])
    elif selected_value == 'calc_Vel':
        explanation = """
        ### Explanation for Velocity Calculation Functions

        The velocity calculation functions are responsible for updating the velocity components based on the pressure gradient and the fluxes. These functions are part of the `Fields` class and provide three different implementations:

        1. **Naive Implementation (`calculate_velocities_naive`)**: This method uses explicit Python loops to iterate over each fluid cell and update the velocity components. It does not use any optimization techniques.
        2. **Vectorized Implementation (`calculate_velocities_vectorized`)**: This method uses NumPy's array operations to handle the velocity calculations more efficiently. It leverages vectorized operations for parallel processing of array elements.
        3. **Numba-Optimized Implementation (`calculate_velocities_numba`)**: This method uses the Numba library to compile the velocity calculation with just-in-time (JIT) compilation, which can significantly speed up the execution.

        Here is the code listing for each implementation:
        """
        code_naive = """
        ```python
        def calculate_velocities_naive(self, grid: Grid):
            for elem in grid.fluid_cells():
                i = elem.i()
                j = elem.j()
                self.set_u(i, j, self._F[i, j] - (self._dt / grid.dx()) * (self._P[i + 1, j] - self._P[i, j]))
                self.set_v(i, j, self._G[i, j] - (self._dt / grid.dy()) * (self._P[i, j + 1] - self._P[i, j]))
        ```
        """
        code_vectorized = """
        ```python
        def calculate_velocities_vectorized(self, grid: Grid):
            P = self._P
            F = self._F
            G = self._G
            dt = self._dt
            dx = grid.dx()
            dy = grid.dy()

            P_shifted_right = np.roll(P, -1, axis=0)
            P_shifted_up = np.roll(P, -1, axis=1)

            U = np.copy(self._U)
            V = np.copy(self._V)

            fluid_mask = grid.get_fluid_cells_mask()

            U[fluid_mask] = F[fluid_mask] - (dt / dx) * (P_shifted_right[fluid_mask] - P[fluid_mask])
            V[fluid_mask] = G[fluid_mask] - (dt / dy) * (P_shifted_up[fluid_mask] - P[fluid_mask])

            self._U = U
            self._V = V
        ```
        """
        code_numba = """
        ```python
        @jit(nopython=True)
        def _calculate_velocities_numba(P, F, G, U, V, dt, dx, dy, fluid_mask):
            dtdx = dt / dx
            dtdy = dt / dy
            for i in range(P.shape[0] - 1):
                for j in range(P.shape[1] - 1):
                    if fluid_mask[i, j]:
                        U[i, j] = F[i, j] - dtdx * (P[i + 1, j] - P[i, j])
                        V[i, j] = G[i, j] - dtdy * (P[i, j + 1] - P[i, j])
            return U, V

        def calculate_velocities_numba(self, grid):
            P = self._P
            F = self._F
            G = self._G
            dt = self._dt
            dx = grid.dx()
            dy = grid.dy()
            U = np.copy(self._U)
            V = np.copy(self._V)
            fluid_mask = grid.get_fluid_cells_mask()

            U, V = _calculate_velocities_numba(P, F, G, U, V, dt, dx, dy, fluid_mask)

            self._U = U
            self._V = V
        ```
        """
        return True, html.Div([
            dcc.Markdown(explanation),
            dbc.Row([
                dbc.Col(dcc.Markdown(code_naive), width=4),
                dbc.Col(dcc.Markdown(code_vectorized), width=4),
                dbc.Col(dcc.Markdown(code_numba), width=4)
            ])
        ])
    elif selected_value == 'calc_Rs':
        explanation = """
        ### Explanation for Right-Hand Side (RS) Calculation Functions

        The RS calculation functions are responsible for computing the right-hand side of the pressure Poisson equation (PPE), which is essential for the numerical solution of the Navier-Stokes equations. These functions are part of the `Fields` class and provide three different implementations:

        1. **Naive Implementation (`calculate_rs_naive`)**: This method uses explicit Python loops to iterate over each fluid cell and compute the RS term based on the fluxes. It does not use any optimization techniques.
        2. **Vectorized Implementation (`calculate_rs_vectorized`)**: This method uses NumPy's array operations to handle the RS calculations more efficiently. It leverages vectorized operations for parallel processing of array elements.
        3. **Numba-Optimized Implementation (`calculate_rs_numba`)**: This method uses the Numba library to compile the RS calculation with just-in-time (JIT) compilation, which can significantly speed up the execution.

        Here is the code listing for each implementation:
        """
        code_naive = """
        ```python
        def calculate_rs_naive(self, grid: Grid):
            for elem in grid.fluid_cells():
                i = elem.i()
                j = elem.j()
                if i != 0 and j != 0 and i != grid.size_x() + 1 and j != grid.size_y() + 1:
                    self.set_rs(i, j, 1 / self._dt * ((self._F[i, j] - self._F[i - 1, j]) / grid.dx() + (self._G[i, j] - self._G[i, j - 1]) / grid.dy()))
        ```
        """
        code_vectorized = """
        ```python
        def calculate_rs_vectorized(self, grid: Grid):
            fluid_mask = grid.get_fluid_cells_mask()
            dF_dx = (self._F - np.roll(self._F, shift=1, axis=0)) / grid.dx()
            dG_dy = (self._G - np.roll(self._G, shift=1, axis=1)) / grid.dy()
            self._RS = 1 / self._dt * (dF_dx + dG_dy)
        ```
        """
        code_numba = """
        ```python
        @jit(nopython=True)
        def _calculate_rs_numba(F, G, RS, dt, fluid_mask, dx, dy):
            for i in range(1, F.shape[0] - 1):
                for j in range(1, F.shape[1] - 1):
                    if fluid_mask[i, j]:
                        RS[i, j] = 1 / dt * ((F[i, j] - F[i - 1, j]) / dx + (G[i, j] - G[i, j - 1]) / dy)
            return RS

        def calculate_rs_numba(self, grid: Grid):
            self._RS = _calculate_rs_numba(self._F, self._G, self._RS, self._dt, grid.get_fluid_cells_mask(), grid.dx(), grid.dy())
        ```
        """
        return True, html.Div([
            dcc.Markdown(explanation),
            dbc.Row([
                dbc.Col(dcc.Markdown(code_naive), width=4),
                dbc.Col(dcc.Markdown(code_vectorized), width=4),
                dbc.Col(dcc.Markdown(code_numba), width=4)
            ])
        ])
    elif selected_value == 'applyPressure':
        explanation = """
        ### Explanation for `applyPressure` function in `FixedWallBoundary`

        The `applyPressure` method in the `FixedWallBoundary` class is responsible for applying pressure boundary conditions to the cells that are part of the fixed wall boundary. This method iterates over each cell in the boundary and sets the pressure based on the position of the cell relative to the boundary.

        #### Performance Optimization of `applyPressure_vectorized`

        The `applyPressure_vectorized` method is an optimized version of `applyPressure` that leverages vectorized operations provided by NumPy to handle the boundary conditions more efficiently. Vectorized operations allow for parallel processing of array elements, which can significantly speed up computations compared to traditional Python loops.

        Here is the code listing for both `applyPressure` and `applyPressure_vectorized`:
        """
        code_applyPressure = """
        ```python
        def applyPressure(self, field: Fields):
            p = field._P

            for elem in self._cells:
                i = elem.i()
                j = elem.j()

                if (elem.is_border(BorderPosition.TOP)):
                    if (elem.is_border(BorderPosition.RIGHT)):
                        field.set_p(i, j, (p[i, j+1] + p[i+1, j]) / 2)
                    elif (elem.is_border(BorderPosition.LEFT)):
                        field.set_p(i, j, (p[i, j+1] + p[i-1, j]) / 2)
                    elif (elem.is_border(BorderPosition.BOTTOM)):
                        field.set_p(i, j, (p[i, j+1] + p[i, j-1]) / 2)
                    else:
                        field.set_p(i, j, p[i, j+1])
                        
                elif (elem.is_border(BorderPosition.BOTTOM)):
                    if (elem.is_border(BorderPosition.RIGHT)):
                        field.set_p(i, j, (p[i+1, j] + p[i, j-1]) / 2)
                    elif (elem.is_border(BorderPosition.LEFT)):
                        field.set_p(i, j, (p[i, j-1] + p[i-1, j]) / 2)
                    else:
                        field.set_p(i, j, p[i, j-1])
                        
                elif (elem.is_border(BorderPosition.RIGHT)):
                    if (elem.is_border(BorderPosition.LEFT)):
                        field.set_p(i, j, (p[i+1, j] + p[i-1, j]) / 2)
                    else:
                        field.set_p(i, j, p[i+1, j])
                        
                elif (elem.is_border(BorderPosition.LEFT)):
                    field.set_p(i, j, p[i-1, j])
        ```
        """
        code_applyPressure_vectorized = """
        ```python
        def applyPressure_vectorized(self, field: Fields):
            i = self._i
            j = self._j
            top = self._top
            bottom = self._bottom
            left = self._left
            right = self._right

            p = field._P

            # Top border
            top_right = np.logical_and(top, right)
            top_left = np.logical_and(top, left)
            top_bottom = np.logical_and(top, bottom)

            for idx in np.where(top_right)[0]:
                field._P[i[idx], j[idx]] = (p[i[idx], j[idx] + 1] + p[i[idx] + 1, j[idx]]) / 2

            for idx in np.where(top_left)[0]:
                field._P[i[idx], j[idx]] = (p[i[idx], j[idx] + 1] + p[i[idx] - 1, j[idx]]) / 2

            for idx in np.where(top_bottom)[0]:
                field._P[i[idx], j[idx]] = (p[i[idx], j[idx] + 1] + p[i[idx], j[idx] - 1]) / 2

            for idx in np.where(top & ~top_right & ~top_left & ~top_bottom)[0]:
                field._P[i[idx], j[idx]] = p[i[idx], j[idx] + 1]

            # Bottom border
            bottom_right = np.logical_and(bottom, right)
            bottom_left = np.logical_and(bottom, left)

            for idx in np.where(bottom_right)[0]:
                field._P[i[idx], j[idx]] = (p[i[idx] + 1, j[idx]] + p[i[idx], j[idx] - 1]) / 2

            for idx in np.where(bottom_left)[0]:
                field._P[i[idx], j[idx]] = (p[i[idx], j[idx] - 1] + p[i[idx] - 1, j[idx]]) / 2

            for idx in np.where(bottom & ~bottom_right & ~bottom_left)[0]:
                field._P[i[idx], j[idx]] = p[i[idx], j[idx] - 1]

            # Right border
            right_left = np.logical_and(right, left)

            for idx in np.where(right_left)[0]:
                field._P[i[idx], j[idx]] = (p[i[idx] + 1, j[idx]] + p[i[idx] - 1, j[idx]]) / 2

            for idx in np.where(right & ~right_left)[0]:
                field._P[i[idx], j[idx]] = p[i[idx] + 1, j[idx]]

            # Left border
            for idx in np.where(left & ~right_left)[0]:
                field._P[i[idx], j[idx]] = p[i[idx] - 1, j[idx]]
        ```
        """
        rest_of_explanation = """
        ### Explanation

        1. **Traditional Loop (`applyPressure`)**:
        - This method uses a loop to iterate over each cell in the boundary.
        - For each cell, it checks the border positions (top, bottom, left, right) and sets the pressure based on the neighboring cells.
        - This approach is straightforward but can be slow for large datasets due to the overhead of Python loops and condition checks.

        2. **Vectorized Approach (`applyPressure_vectorized`)**:
        - This method uses NumPy's array operations to handle the boundary conditions.
        - It first computes logical arrays (`top_right`, `top_left`, etc.) to identify cells that are at specific corners or edges.
        - It then uses `np.where` to find indices of cells that meet specific conditions and applies the pressure boundary conditions in a vectorized manner.
        - This approach leverages the underlying C and Fortran routines in NumPy, which are highly optimized for performance.

        ### Performance Benefits

        - **Parallel Processing**: NumPy operations are implemented in C and can process large arrays in parallel, making them much faster than Python loops.
        - **Reduced Overhead**: Eliminating the need for Python loops and condition checks reduces the overhead associated with function calls and branching.
        - **Memory Efficiency**: NumPy arrays are more memory-efficient than Python lists, especially for large datasets.

        By using `applyPressure_vectorized`, the boundary conditions can be applied much faster, especially for large grids, making the simulation more efficient and scalable.
        """

        # Explanation of __init__ function
        init_explanation = """
        ### Explanation of `__init__` Function in `FixedWallBoundary`

        The `__init__` method in the provided code snippet is a constructor for the `FixedWallBoundary` class that initializes an object with a list of `Cell` objects. This method is designed to store and organize information about each cell, which is likely used in a computational fluid dynamics (CFD) simulation or similar application where cells represent discrete elements in a grid. Here's the code for the `__init__` method:

        ```python
        def __init__(self, cells: List[Cell]):
            self._cells = cells
            self._cell_info = np.array([
                [cell.i(), cell.j()] + cell.border_info()
                for cell in cells
            ])
            self._i = self._cell_info[:, 0]
            self._j = self._cell_info[:, 1]
            self._top = self._cell_info[:, 2]
            self._bottom = self._cell_info[:, 3]
            self._left = self._cell_info[:, 4]
            self._right = self._cell_info[:, 5]
        ```

        This method performs the following tasks:

        1. **Initialization of `_cells` Attribute**:
        ```python
        self._cells = cells
        ```
        This line stores the list of `Cell` objects passed to the constructor in the `_cells` attribute of the class. This attribute will hold all the cell objects that the class needs to manage.

        2. **Creation of `_cell_info` Array**:
        ```python
        self._cell_info = np.array([
            [cell.i(), cell.j()] + cell.border_info()
            for cell in cells
        ])
        ```
        This line creates a 2D NumPy array named `_cell_info`. Each row in this array corresponds to a cell and contains the following information:
        - `cell.i()`: The row index of the cell in the grid.
        - `cell.j()`: The column index of the cell in the grid.
        - `cell.border_info()`: Additional information about the cell's borders, which is obtained by calling the `border_info()` method on each cell. This method likely returns a list or tuple containing information about whether the cell has borders on the top, bottom, left, and right sides.

        3. **Extraction of Specific Cell Information**:
        ```python
        self._i = self._cell_info[:, 0]
        self._j = self._cell_info[:, 1]
        self._top = self._cell_info[:, 2]
        self._bottom = self._cell_info[:, 3]
        self._left = self._cell_info[:, 4]
        self._right = self._cell_info[:, 5]
        ```
        These lines extract specific columns from the `_cell_info` array and store them in separate attributes of the class:
        - `self._i`: An array of row indices for all cells.
        - `self._j`: An array of column indices for all cells.
        - `self._top`, `self._bottom`, `self._left`, `self._right`: Arrays indicating whether each cell has a border on the respective side.

        This setup is particularly useful in the context of a CFD simulation or similar applications where the pressure needs to be applied to cells based on their positions and border conditions. For example, in the `applyPressure` method, you might use these attributes to determine how pressure affects the cells, especially at the boundaries where the presence or absence of a border could significantly affect the flow dynamics.
        """

        return True, html.Div([
            dcc.Markdown(explanation),
            dbc.Row([
                dbc.Col(dcc.Markdown(code_applyPressure), width=6),
                dbc.Col(dcc.Markdown(code_applyPressure_vectorized), width=6)
            ]),
            dcc.Markdown(rest_of_explanation),
            dcc.Markdown(init_explanation)
        ])
    else:
        explanation = "Explanation for the selected function will be displayed here."
        code = "Code for the selected function will be displayed here."

    return True, html.Div([
        dcc.Markdown(explanation),
        dcc.Markdown(code)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
