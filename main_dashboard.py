import os
import time
import subprocess as sp
from collections import namedtuple
import seaborn as sns
from IPython.display import display
import ipywidgets as widgets

from multiproduct_dashboard import read_dataset, export_dataset, run_multiobjective_GA, print_best_solutions, export_dataset, read_dataset, generate_ampl, print_solver_solution

sns.set()

Dataset = namedtuple('Dataset', ['nodes', 'supplies', 'demands', 'costs', 'capacities'])
dataset = None
population = None
datasets = None

def render_dashboard():
    
    ### MENU

    space = widgets.Box(layout=widgets.Layout(width='80px'))
    vspace = widgets.Box(layout=widgets.Layout(height='20px'))
    vbox_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    justify_content='space-around',
                    width='100%',
                    max_width='1000px',
                    margin='0 auto',
                    border='solid')
    
    #### GENERATE DATASET

    def gen_dataset(e):
        global dataset
        global datasets
        nodes, supplies, demands, costs, capacities = generate_dataset(
                n_nodes_input.value, n_levels_input.value, total_supplies=[400]*n_items_input.value, 
                total_demands=[500]*n_items_input.value, transp_costs=(100,1000), 
                random_state=42)   
        dataset = Dataset(nodes, supplies, demands, costs, capacities)
        export_dataset(dataset, 'test.txt')
        datasets = {file: file \
                       for file in sorted(os.listdir('datasets/evaluation')) \
                       if file.endswith('.txt')}
        dataset_dropdown.options = datasets
        generate_dataset_btn.description = 'Generated dataset!'
        time.sleep(0.8)
        generate_dataset_btn.description = 'Generate dataset'
    
    n_nodes_input = widgets.IntSlider(15, 5, 30, 1, description='# Nodes')
    n_levels_input = widgets.IntSlider(4, 3, 8, 1, description='# Levels')
    n_items_input = widgets.IntSlider(3, 1, 7, 1, description='# Items')
    generate_dataset_btn = widgets.Button(description='Generate dataset')

    aux_box1 = widgets.HBox([n_nodes_input, space, n_levels_input])
    aux_box2 = widgets.HBox([n_items_input, space, generate_dataset_btn])

    generate_dataset_box = widgets.VBox([aux_box1, aux_box2])

    generate_dataset_btn.on_click(gen_dataset)

    #### LOAD DATASET

    def load_dataset(e):
        global dataset
        dataset = read_dataset(dataset_dropdown.value)
        load_dataset_btn.description = 'Loaded dataset!'
        time.sleep(0.8)
        load_dataset_btn.description = 'Load dataset'
    
    datasets = {file: file \
                       for file in sorted(os.listdir('datasets/evaluation')) \
                       if file.endswith('.txt')}
    dataset_dropdown = widgets.Dropdown(
        options = datasets,
        description='Dataset',
        value=datasets[sorted(list(datasets))[0]]
    )

    load_dataset_btn = widgets.Button(description='Load Dataset')
    load_dataset_box = widgets.HBox([dataset_dropdown, space, load_dataset_btn])

    load_dataset_btn.on_click(load_dataset)

    ### GENETIC ALGORITHM
    
    def export_ga_solutions(e):
        global dataset
        global population
        print_best_solutions(population, dataset, 'fronts.out')
        export_ga_solutions_btn.description = 'Solutions exported!'
        time.sleep(0.8)
        export_ga_solutions_btn.description = 'Export solutions'
    
    def run_genetic_algorithm(e):
        global dataset
        global population
        pop, hof, log, toolbox = run_multiobjective_GA(dataset.nodes, 
                                dataset.supplies, dataset.demands,
                                dataset.costs, dataset.capacities, 
                                pop_size=population_size_input.value, 
                                n_generations=n_generations_input.value,
                                early_stopping_rounds=early_stopping_rounds_input.value,
                                print_log=False, plot_log=True, 
                                plot_pop=True, plot_fairness=True,
                                log_output=log_output, pop_output=pop_output,
                                fairness_output=fairness_output)
        population = pop
    
    population_size_input = widgets.IntSlider(100, 50, 300, 10, description='Population size',
                              style={'description_width': 'initial'})
    n_generations_input = widgets.IntSlider(50, 1, 300, 1, description='# Generations',
                              style={'description_width': 'initial'})
    early_stopping_rounds_input = widgets.IntSlider(10, 1, 50, 1, description='Early stopping rounds',
                              style={'description_width': 'initial'})
    run_genetic_algorithm_btn = widgets.Button(description='Run Genetic Algorithm',
                                          style={'description_width': 'initial'},
                                          layout=widgets.Layout(width='20%'))
    export_ga_solutions_btn = widgets.Button(description='Export solutions')

    aux_box3 = widgets.HBox([population_size_input, n_generations_input, early_stopping_rounds_input])
    aux_box4 = widgets.HBox([run_genetic_algorithm_btn, space, export_ga_solutions_btn])

    run_genetic_algorithm_btn.on_click(run_genetic_algorithm)
    export_ga_solutions_btn.on_click(export_ga_solutions)

    #### PLOTS

    log_output = widgets.Output() 
    pop_output = widgets.Output()
    fairness_output = widgets.Output()

    plots_hbox_layout = widgets.Layout(display='flex', flex_flow='row',
                        align_items='center', justify_content='space-around',
                        width='100%', max_width='800px', margin='0 auto')

    plots_vbox_layout = widgets.Layout(display='flex', flex_flow='column',
                        align_items='center', justify_content='space-around',
                        width='100%', max_width='800px', margin='0 auto')

    pop_fairness_box = widgets.HBox([pop_output, fairness_output], layout=plots_hbox_layout)
    plots_box = widgets.VBox([log_output, pop_fairness_box], layout=plots_vbox_layout)

    ga_tab = widgets.VBox([aux_box3, aux_box4, plots_box], layout=widgets.Layout(width='900px'))

    ### SOLVER

    def run_solver(e):
        global dataset
        generate_ampl(dataset.nodes, dataset.supplies, dataset.demands, dataset.costs,
                      dataset.capacities, weight='10000', 
                      output_file='datasets/evaluation/model.mod')
        sp.run("~/Downloads/ampl.linux64/ampl < run.ampl", shell=True)
        process = sp.Popen("~/Downloads/model-couenne/couenne model.nl", shell=True, stdout=sp.PIPE)
        with solver_output:
            for c in iter(lambda: process.stdout.read(1), b''): 
                print(c.decode(), end='')
        solver.clear_output(wait=True)
    
    def export_solver_solution(e):
        global dataset
        print_solver_solution(dataset, 'solver.out')
        export_solver_solution_btn.description = 'Solution exported!'
        time.sleep(0.8)
        export_solver_solution_btn.description = 'Export solution'
    
    solver_output = widgets.Output()
    objective_weight_inp = widgets.IntSlider(4, 0, 8, 1, description='$f_2(x)$ weight ($10^x$)',
                              style={'description_width': 'initial'})
    run_solver_btn = widgets.Button(description='Run Couenne Solver',
                                          style={'description_width': 'initial'},
                                          layout=widgets.Layout(width='20%'))
    export_solver_solution_btn = widgets.Button(description='Export solution')

    aux_box5 = widgets.HBox([run_solver_btn, space, export_solver_solution_btn])

    solver_tab = widgets.VBox([objective_weight_inp, aux_box5, solver_output], 
                              layout=widgets.Layout(width='900px'))


    export_solver_solution_btn.on_click(export_solver_solution)
    run_solver_btn.on_click(run_solver)

    ## DASHBOARD

    menu = widgets.VBox([generate_dataset_box, load_dataset_box])
    tabs = widgets.Tab([ga_tab, solver_tab])
    tabs.set_title(0, 'Genetic Algorithm')
    tabs.set_title(1, 'Solver')
    dashboard = widgets.VBox([vspace, menu, tabs, vspace], layout=vbox_layout)
    return dashboard