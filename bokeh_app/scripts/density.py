# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

from scipy.stats import gaussian_kde

from bokeh.plotting import figure
from bokeh.models import (CategoricalColorMapper, HoverTool,ColumnDataSource, Panel,
						  FuncTickFormatter, SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,
								  Tabs, CheckboxButtonGroup,
								  TableColumn, DataTable, Select, Button, TextInput)
from bokeh.layouts import column, row, WidgetBox
from bokeh.palettes import Category20_16


def density_tab(flights):
	# Dataset for density plot based on carriers, range of delays,
	# and bandwidth for density estimation
	def make_dataset(carrier_list, range_start, range_end, bandwidth):

		xs = []
		ys = []
		colors = []
		labels = []

		for i, carrier in enumerate(carrier_list):
			subset = flights[flights['name'] == carrier]
			subset = subset[subset['arr_delay'].between(range_start,
														range_end)]

			kde = gaussian_kde(subset['arr_delay'], bw_method=bandwidth)

			# Evenly space x values
			x = np.linspace(range_start, range_end, 100)
			# Evaluate pdf at every value of x
			y = kde.pdf(x)

			# Append the values to plot
			xs.append(list(x))
			ys.append(list(y))

			# Append the colors and label
			colors.append(airline_colors[i])
			labels.append(carrier)

		new_src = ColumnDataSource(data={'x': xs, 'y': ys,
										 'color': colors, 'label': labels})

		return new_src

	def make_plot(src):
		p = figure(plot_width=700, plot_height=700,
				   title='Density Plot of Arrival Delays by Airline',
				   x_axis_label='Delay (min)', y_axis_label='Density')

		p.multi_line('x', 'y', color='color', legend='label',
					 line_width=3,
					 source=src)

		# Hover tool with next line policy
		hover = HoverTool(tooltips=[('Carrier', '@label'),
									('Delay', '$x'),
									('Density', '$y')],
						  line_policy='next')

		# Add the hover tool and styling
		p.add_tools(hover)

		p = style(p)

		return p

	def make_new_plot(src):
		p = figure(plot_width=700, plot_height=700,
				   title='Density Plot of Arrival Delays by Airline',
				   x_axis_label='Delay (min)', y_axis_label='Density')

		p.multi_line('x', 'y', color='color', legend='label',
					 line_width=3,
					 source=src)

		# Hover tool with next line policy
		hover = HoverTool(tooltips=[('Carrier', '@label'),
									('Delay', '$x'),
									('Density', '$y')],
						  line_policy='next')

		# Add the hover tool and styling
		p.add_tools(hover)

		p = style(p)

		return p

	def plot_success_rate(src2):



	def update(attr, old, new):
		# List of carriers to plot
		carriers_to_plot = [carrier_selection.labels[i] for i in
							carrier_selection.active]

		# If no bandwidth is selected, use the default value
		if bandwidth_choose.active == []:
			bandwidth = None
		# If the bandwidth select is activated, use the specified bandwith
		else:
			bandwidth = bandwidth_select.value

		new_src = make_dataset(carriers_to_plot,
							   range_start=range_select.value[0],
							   range_end=range_select.value[1],
							   bandwidth=bandwidth)

		src.data.update(new_src.data)

	def style(p):
		# Title
		p.title.align = 'center'
		p.title.text_font_size = '20pt'
		p.title.text_font = 'serif'

		# Axis titles
		p.xaxis.axis_label_text_font_size = '14pt'
		p.xaxis.axis_label_text_font_style = 'bold'
		p.yaxis.axis_label_text_font_size = '14pt'
		p.yaxis.axis_label_text_font_style = 'bold'

		# Tick labels
		p.xaxis.major_label_text_font_size = '12pt'
		p.yaxis.major_label_text_font_size = '12pt'

		return p



	def draw_simulation():

		# List of carriers to plot
		carriers_to_plot = [carrier_selection.labels[i] for i in
							carrier_selection.active]

		# If no bandwidth is selected, use the default value
		if bandwidth_choose.active == []:
			bandwidth = None
		# If the bandwidth select is activated, use the specified bandwith
		else:
			bandwidth = bandwidth_select.value

		new_src = make_dataset(carriers_to_plot,
							   range_start=4,
							   range_end=12,
							   bandwidth=bandwidth)

		src2.data.update(new_src.data)




	# Carriers and colors
	available_carriers = list(set(flights['name']))
	available_carriers.sort()

	airline_colors = Category20_16
	airline_colors.sort()

	# Carriers to plot
	carrier_selection = CheckboxGroup(labels=available_carriers,
									  active=[0, 1])
	carrier_selection.on_change('active', update)

	range_select = RangeSlider(start=-60, end=180, value=(-60, 120),
							   step=5, title='Range of Delays (min)')
	range_select.on_change('value', update)

	# Initial carriers and data source
	initial_carriers = [carrier_selection.labels[i] for
						i in carrier_selection.active]

	# Bandwidth of kernel
	bandwidth_select = Slider(start=0.1, end=5,
							  step=0.1, value=0.5,
							  title='Bandwidth for Density Plot')
	bandwidth_select.on_change('value', update)

	# Whether to set the bandwidth or have it done automatically
	bandwidth_choose = CheckboxButtonGroup(
		labels=['Choose Bandwidth (Else Auto)'], active=[])
	bandwidth_choose.on_change('active', update)



	# FUNDS_EARMARKED_FOR_GOAL
	fund_input = TextInput(value="100000", title="Funds for Goal:")

	contribution_input = TextInput(value="1000", title="Contribution Amount:")

	income_input = TextInput(value="100000", title="Annual Salary:")

	fga_input = TextInput(value="100000", title="Future Goal Amount:")


	retirementage_select = Slider(start = 20, end = 100, step = 1, value = 1, title = 'Projected Retirement Age')
	#retirementage_select.on_change('value', update)

	planning_select = Slider(start = 90, end = 120, step = 1, value = 1, title = 'Planning_Horizon')
	#planning_select.on_change('value', update)


	# Risk Tolerance
	risk_select = Select(title="Risk Tolerance:", value="Moderate", options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"])


	age_select = Slider(start = 20, end = 100, step = 1, value = 1, title = 'Age')
	#age_select.on_change('value', update)

	b = Button(label='Simulate')
	b.on_click(draw_simulation)


	# Make the density data source
	src = make_dataset(initial_carriers,
					   range_start=range_select.value[0],
					   range_end=range_select.value[1],
					   bandwidth=bandwidth_select.value)

	# Make the density data source
	src2 = make_dataset(initial_carriers,
					   range_start=range_select.value[0],
					   range_end=range_select.value[1],
					   bandwidth=bandwidth_select.value)

	# Make the density plot
	p = make_plot(src)

	p2 = make_new_plot(src2)

	# Add style to the plot
	p = style(p)

	p2 = style(p2)


	# Put controls in a single element
	controls = WidgetBox(carrier_selection, range_select,
						 bandwidth_select, bandwidth_choose, fund_input, contribution_input, income_input, fga_input, retirementage_select, planning_select, risk_select, age_select , b)

	# Create a row layout
	layout = row(controls, p, p2)

	# Make a tab with the layout
	tab = Panel(child=layout, title='Density Plot')


	return tab



	# # FUNDS_EARMARKED_FOR_GOAL
	# fund_input = TextInput(value="100000", title="Funds for Goal:")
	#
	# contribution_input = TextInput(value="1000", title="Contribution Amount:")
	#
	# income_input = TextInput(value="100000", title="Annual Salary:")
	#
	# fga_input = TextInput(value="100000", title="Future Goal Amount:")
	#
	#
	# retirementage_select = Slider(start = 20, end = 100, step = 1, value = 1, title = 'Projected Retirement Age')
	# #retirementage_select.on_change('value', update)
	#
	# planning_select = Slider(start = 90, end = 120, step = 1, value = 1, title = 'Planning_Horizon')
	# #planning_select.on_change('value', update)
	#
	#
	# # Risk Tolerance
	# risk_select = Select(title="Risk Tolerance:", value="Moderate", options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"])
	#
	#
	# age_select = Slider(start = 20, end = 100, step = 1, value = 1, title = 'Age')
	# #age_select.on_change('value', update)
	#
	# b = Button(label='Simulate')
	# b.on_click(draw_simulation)


	# to be deleted=============================================================

