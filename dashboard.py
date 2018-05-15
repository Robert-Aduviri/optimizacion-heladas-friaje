import os
import seaborn as sns
from IPython.display import display
import ipywidgets as widgets

from utils import load_file, run_genetic_algorithm, plot_log, decode_chromosome, plot_graph, chromosome_fitness

sns.set()

optimization_log = None

def render_dashboard():
    inp_pop_size = widgets.IntSlider(100, 50, 300, description='Population size',
                          style={'description_width': 'initial'})
    inp_n_generations = widgets.IntSlider(20, 5, 100, description='# Generations',
                          style={'description_width': 'initial'})

    sel_dataset_options = {file[:-4]: [file, file.replace('.txt', '_ampl.mod')] \
                   for file in sorted(os.listdir('data')) \
                   if file.startswith('test_') and \
                      file.endswith('.txt')}
    sel_dataset = widgets.Dropdown(
        options = sel_dataset_options,
        description='Dataset',
        value=sel_dataset_options[sorted(list(sel_dataset_options))[0]]
    )

    btn_run_ga   = widgets.Button(description='Run genetic algorithm')
    btn_run_glpk = widgets.Button(description='Run GLPK solver')
    output_log    = widgets.Output()
    output_plots  = widgets.Output()

    def run_ga(e):
        global optimization_log
        with output_log:
            E, D, W, d, g, v, t, a, c = load_file(f'data/{sel_dataset.value[0]}')
            population, solutions, log = run_genetic_algorithm(
                                    E, D, W, d, g, v, t, a, c, \
                                    pop_size=inp_pop_size.value, 
                                    n_generations=inp_n_generations.value, \
                                    n_solutions=5, crossover_p=0.5, \
                                    mutation_p=0.2)  
            optimization_log = log
        output_log.clear_output(wait=True)
    btn_run_ga.on_click(run_ga)

    btn_plot_opt_log  = widgets.Button(description='Plot optimization log')
    btn_plot_solution = widgets.Button(description='Plot solution')

    def plot_opt_log(e):
        with output_plots:
            if optimization_log:            
                ax = plot_log(optimization_log, optimal_value=None, min_max=False)
                display(ax.figure)
        output_plots.clear_output(wait=True)

    btn_plot_opt_log.on_click(plot_opt_log)

    options = widgets.VBox([inp_pop_size,
                            inp_n_generations,
                            sel_dataset])
    actions = widgets.VBox([btn_run_ga,
                            btn_run_glpk])
    plots = widgets.VBox([btn_plot_opt_log,
                          btn_plot_solution])
    input_panel  = widgets.HBox([options, actions, plots])
    output_panel = widgets.HBox([output_log, output_plots])
    return widgets.VBox([input_panel,
                  output_panel])