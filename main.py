import package_installer
from Backtest_Class import Backtester
from Portfolio_Strategy_Class import Config, Strategy_type, Optimize_method, Portf_Strategy, \
    Index_names
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from datetime import datetime
import matplotlib.pyplot as plt


# -------------------- BUTTONS FUNCTION -----------------------------------------------------------
def calc_number_hit():
    """
    Function that calculate the number of hit from our backtest
    """
    str_start = cal.get()
    str_end = cal2.get()
    dt_start = datetime.strptime(str_start, '%m/%d/%y')
    dt_end = datetime.strptime(str_end, '%m/%d/%y')
    # calculate
    int_business_days = round((dt_end - dt_start).days * 5 / 7 / 60)
    # print in the text box the number of hits with this configuration
    output_text.insert("end", f"Careful! The number of hits with this configuration is: {int_business_days + 3}\n")


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=10, header_color='#40466e',
                     row_colors=['#f1f1f2', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    Function that print a dataframe as PNG
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    # Browse all element of the dataframe
    for element, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if element[0] == 0 or element[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[element[0] % len(row_colors)])
    'Output'
    return ax.get_figure(), ax


def get_values():
    """
    Fill our configuration class with the inputs of the user
    """
    start_date = cal.get()
    startegy_type = combo2.get()
    method_optimize = combo3.get()
    rebalance_period = combo4.get()
    main_index = combo5.get()
    end_date = cal2.get()

    bl_create_PORT = bool(bl_var.get())
    output_text.delete('1.0', tk.END)
    output_text.insert("end", "Last selected portfolio settings: \n")
    output_text.insert("end", f"-> Start date: {start_date}\n")
    output_text.insert("end", f"-> End date: {end_date}\n")
    output_text.insert("end", f"-> Strategy type: {startegy_type}\n")
    output_text.insert("end", f"-> Optimization mode: {method_optimize}\n")
    output_text.insert("end", f"-> Rebalance frequency (days): {rebalance_period}\n")
    output_text.insert("end", f"-> Main index: {main_index}\n")
    output_text.insert("end", f"-> PORT file generation: {bl_create_PORT}\n")

    start_date = datetime.strptime(start_date, '%m/%d/%y')
    end_date = datetime.strptime(end_date, '%m/%d/%y')

    conf = Config(
        start_ts=start_date,
        end_ts=end_date,
        optimize_method=method_optimize,
        main_index=main_index,
        strategy_type=startegy_type,
        rebalance_period=int(rebalance_period),
        bl_generate_outputPORT=bl_create_PORT, )
    # bl_index_comparaison=bl_index_comparaison,)
    return conf


def launch_script():
    """
    Method that launch all the code to bakctest the strategy of the user
    """
    # get the arguments for the configuration
    conf = get_values()
    # Create the strategy
    strat = Portf_Strategy(conf, )
    # Backtest the factor
    backtest = Backtester(conf, strat)
    # Plot all outputs
    backtest.plot_levels()
    output = backtest.df_global_output
    output.to_excel("Factor and Index performance.xlsx")
    # a = output.to_string()
    output = output.round(decimals=4)
    fig, ax = render_mpl_table(output, header_columns=0, col_width=1.75)
    plt.show()
    fig.savefig("Factor_and_index_analysis.png")


# ----------------------- INTERFACE CODE ---------------------------------------------------------------------------
root = tk.Tk()
root.title("Bloomberg API")
root.geometry("575x650")
style = ttk.Style(root)
style.theme_use('clam')
style.configure('TLabel', background='black', foreground='white')
style.configure('TFrame', background='black')

style.configure('TCombobox', selectbackground='black', fieldbackground='black', foreground='orange', background='grey',
                font=("Georgia", 17, "bold", 'white'))

# Create frame for input widgets
input_frame = ttk.Frame(root, relief="raised", borderwidth=30)
input_frame.pack(fill="both", expand=True)

input_frame.pack(fill="both", expand=True)
root.option_add('*TCombobox*Listbox.foreground' % input_frame, 'orange')
root.option_add('*TCombobox*Listbox.background' % input_frame, 'black')
root.option_add('*TCombobox*Listbox.justify', 'center')
root.option_add('*TCombobox*.justify', 'center')
label = tk.Label(input_frame, text="BLOOMBERG™ API : PORTFOLIO CONSTRUCTOR", font=("Minion Pro SmBd", 13, "bold"),
                 background="black", foreground="white")
label.place(x=50, y=-25)

# Create calendar widget and label
label2 = ttk.Label(input_frame, text="START DATE", foreground="orange", font=("Georgia", 10, "bold"))
label2.grid(row=2, column=0, sticky="e", pady=5, padx=5)
cal = DateEntry(input_frame, width=20, background='black',
                foreground='orange', justify='center', borderwidth=2, year=2021)
cal.grid(row=2, column=1, pady=5, padx=5)
cal.set_date(datetime(2020, 1, 17))
# Create calendar widget and label
label6 = ttk.Label(input_frame, text="END DATE", foreground="orange", font=("Georgia", 10, "bold"))
label6.grid(row=3, column=0, sticky="e", pady=5, padx=5)
cal2 = DateEntry(input_frame, width=20, background='black',
                 foreground='orange', justify='center', borderwidth=2, year=2023)
cal2.grid(row=3, column=1, pady=5, padx=5)
cal2.set_date(datetime(2023, 1, 2))

# Create first combobox and label
label2 = ttk.Label(input_frame, text="STRATEGY", foreground="orange", font=("Georgia", 10, "bold"))
label2.grid(row=4, column=0, sticky="e", pady=5, padx=5)
combo2 = ttk.Combobox(input_frame, justify='center',
                      values=[Strategy_type.VALUE.value, Strategy_type.MOMENTUM.value, Strategy_type.SIZE.value],
                      width=20)
combo2.grid(row=4, column=1, pady=5, padx=5)
combo2.set(Strategy_type.VALUE.value)

# Create first combobox and label
label3 = ttk.Label(input_frame, text="OPTIMIZATION MODE", foreground="orange", font=("Georgia", 10, "bold"))
label3.grid(row=5, column=0, sticky="e", pady=5, padx=5)
combo3 = ttk.Combobox(input_frame, justify='center',
                      values=[Optimize_method.MINVVAR.value, Optimize_method.SHARPE.value,
                              Optimize_method.DIVERSIFY.value], width=20)
combo3.grid(row=5, column=1, pady=5, padx=5)
combo3.set(Optimize_method.SHARPE.value)

# Create first combobox and label
label4 = ttk.Label(input_frame, text="REBALANCE PERIOD (DAYS)", foreground="orange", font=("Georgia", 10, "bold"))
label4.grid(row=6, column=0, sticky="e", pady=5, padx=5)
combo4 = ttk.Combobox(input_frame, justify='center', values=[20, 30, 40, 50, 60, 90], width=20)
combo4.grid(row=6, column=1, pady=5, padx=5)
combo4.set(40)

# Create first combobox and label
label5 = ttk.Label(input_frame, text="MAIN INDEX", foreground="orange", font=("Georgia", 10, "bold"))
label5.grid(row=7, column=0, sticky="e", pady=5, padx=5)
combo5 = ttk.Combobox(input_frame, justify='center',
                      values=[Index_names.SX5E.value, Index_names.DAX.value, Index_names.FTSE.value,
                              Index_names.CAC.value, Index_names.NDX.value, Index_names.SPX.value], width=20)
combo5.grid(row=7, column=1, pady=5, padx=5)
combo5.set(Index_names.SX5E.value)

label7 = ttk.Label(input_frame, text="GENERATE PORT FILE", foreground="orange", font=("Georgia", 10, "bold"))
label7.grid(row=9, column=0, sticky="e", pady=5, padx=5)
bl_var = tk.IntVar()
combo7 = tk.Checkbutton(input_frame, variable=bl_var, width=1, bg='black', foreground="black",
                        highlightbackground="black", highlightcolor="BLACK", activebackground='black',
                        activeforeground='black')
combo7.grid(row=9, column=1, pady=5, padx=5)

get_values_button = tk.Button(input_frame, bg='#d47c24', fg='white', text="EVALUATE NB HITS", width=25,
                              command=calc_number_hit, font=("Helvetica", 11, "bold"))
get_values_button.place(x=160, y=225)

launch_button = tk.Button(input_frame, bg='#10942d', fg='white', text="LAUNCH SCRIPT", width=25,
                          font=("Helvetica", 11, "bold"), command=lambda: launch_script())

launch_button.place(x=160, y=275)

output_text = tk.Text(root, height=10, foreground="orange", background="black", font=("Georgia", 12), borderwidth=0,
                      highlightthickness=0)
output_text.pack(padx=4, pady=6, fill="both", expand=True)
output_text.insert("end", "Welcome to the Bloomberg API portfolio constructor !\n")
output_text.insert("end", "---> Made by Thomas & Hicham - version 2.0.5 <--- \n\n")
output_text.insert("end", "Please select your options and press 'LAUNCH SCRIPT'\n")

root.mainloop()
