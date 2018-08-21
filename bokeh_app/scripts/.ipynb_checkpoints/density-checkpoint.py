from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from bokeh.plotting import figure
from bokeh.models import (CategoricalColorMapper, HoverTool,ColumnDataSource, Panel, FuncTickFormatter, SingleIntervalTicker, LinearAxis)
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,Tabs, CheckboxButtonGroup, TableColumn, DataTable, Select, Button, TextInput)
from bokeh.layouts import column, row, WidgetBox
from bokeh.palettes import Category20_16
import tensorflow as tf
from six.moves import range, zip
import numpy as np
import zhusuan as zs
import numpy.linalg as NA
import pickle

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
			subset = subset[subset['arr_delay'].between(range_start, range_end)]
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

		new_src = ColumnDataSource(data={'x': xs, 'y': ys, 'color': colors, 'label': labels})
		return new_src

	def make_plot(src):
		p = figure(plot_width=700, plot_height=700,
				   title='Density Plot of Arrival Delays by Airline',
				   x_axis_label='Delay (min)', y_axis_label='Density')

		p.multi_line('x', 'y', color='color', legend='label', line_width=3, source=src)

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
        #x_train_standardized = pickle.load(open("x_train.pkl", "rb"))
        #y_train_standardized = pickle.load(open("y_train.pkl", "rb"))
        #x_test_standardized = pickle.load(open("x_test.pkl", "rb"))
        #y_test_standardized = pickle.load(open("y_test.pkl", "rb"))
    
        #model_file_path = 'modelsave/'
        
        #y_pred = predict(model_file_path, x_train_standardized, y_train_standardized, x_test_standardized, y_test_standardized)
        
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
        
        #x_train_standardized = pickle.load(open("x_train.pkl", "rb"))
        #y_train_standardized = pickle.load(open("y_train.pkl", "rb"))
        #x_test_standardized = pickle.load(open("x_test.pkl", "rb"))
        #y_test_standardized = pickle.load(open("y_test.pkl", "rb"))

        #model_file_path = 'modelsave/'
        #predict(model_file_path, x_train_standardized, y_train_standardized, x_test_standardized, y_test_standardized)
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


    @zs.reuse('model')
    def bayesianNN(observed, x, n_x, layer_sizes, n_particles):
        with zs.BayesianNet(observed=observed) as model:
            ws = []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mu = tf.zeros([1, n_out, n_in + 1])
                ws.append(
                    zs.Normal('w' + str(i), w_mu, std=1.,
                              n_samples=n_particles, group_ndims=2))

            # forward
            ly_x = tf.expand_dims(
                tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]), 3)
            for i in range(len(ws)):
                w = tf.tile(ws[i], [1, tf.shape(x)[0], 1, 1])
                ly_x = tf.concat(
                    [ly_x, tf.ones([n_particles, tf.shape(x)[0], 1, 1])], 2)
                ly_x = tf.matmul(w, ly_x) / tf.sqrt(tf.to_float(tf.shape(ly_x)[2]))
                if i < len(ws) - 1:
                    ly_x = tf.nn.relu(ly_x)

            y_mean = tf.squeeze(ly_x, [2, 3])
            y_logstd = tf.get_variable('y_logstd', shape=[],
                                       initializer=tf.constant_initializer(0.))
            y = zs.Normal('y', y_mean, logstd=y_logstd)

        return model, y_mean



    def mean_field_variational(layer_sizes, n_particles):
        with zs.BayesianNet() as variational:
            ws = []
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                w_mean = tf.get_variable(
                    'w_mean_' + str(i), shape=[1, n_out, n_in + 1],
                    initializer=tf.constant_initializer(0.))
                w_logstd = tf.get_variable(
                    'w_logstd_' + str(i), shape=[1, n_out, n_in + 1],
                    initializer=tf.constant_initializer(0.))
                ws.append(
                    zs.Normal('w' + str(i), w_mean, logstd=w_logstd,
                              n_samples=n_particles, group_ndims=2))
        return variational

    def predict(model_file_path, x_train_standardized, y_train_standardized, x_test_standardized, y_test_standardized):
        tf.set_random_seed(1237)
        np.random.seed(1234)

        x_train = x_train_standardized
        y_train = y_train_standardized
        x_test = x_test_standardized
        y_test = y_test_standardized

        N, n_x = x_train.shape

        # Define model parameters
        n_hiddens = [50]

        # Build the computation graph
        n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
        x = tf.placeholder(tf.float32, shape=[None, n_x])
        y = tf.placeholder(tf.float32, shape=[None])
        layer_sizes = [n_x] + n_hiddens + [1]  # layer_sizes is 

        w_names = ['w' + str(i) for i in range(len(layer_sizes) - 1)]

        def log_joint(observed):
            model, _ = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
            log_pws = model.local_log_prob(w_names)
            log_py_xw = model.local_log_prob('y')
            return tf.add_n(log_pws) + log_py_xw * N

        variational = mean_field_variational(layer_sizes, n_particles)
        qw_outputs = variational.query(w_names, outputs=True, local_log_prob=True)
        latent = dict(zip(w_names, qw_outputs))
        lower_bound = zs.variational.elbo(
            log_joint, observed={'y': y}, latent=latent, axis=0)
        cost = tf.reduce_mean(lower_bound.sgvb())
        lower_bound = tf.reduce_mean(lower_bound)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        infer_op = optimizer.minimize(cost)

        # prediction: rmse & log likelihood
        observed = dict((w_name, latent[w_name][0]) for w_name in w_names)
        observed.update({'y': y})
        model, y_mean = bayesianNN(observed, x, n_x, layer_sizes, n_particles)
        y_pred = tf.reduce_mean(y_mean, 0)
        l2 = tf.norm(y_pred - y)/tf.norm(y)
        log_py_xw = model.local_log_prob('y')

        # Define training/evaluation parameters
        lb_samples = 10
        ll_samples = 5000
        epochs = 500
        batch_size = 10
        iters = int(np.floor(x_train.shape[0] / float(batch_size)))
        test_freq = 10

        # Add a train server object
        saver = tf.train.Saver(save_relative_paths=True)


        # Run the inference
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Restore from the latest checkpoint
            ckpt_file = tf.train.latest_checkpoint(model_file_path)
            begin_epoch = 1
            if ckpt_file is not None:
                print('Restoring model from {}...'.format(ckpt_file))
                begin_epoch = int(ckpt_file.split('.')[-2]) + 1
                saver.restore(sess, ckpt_file)

    #         for epoch in range(begin_epoch, epochs + 1):
    #             lbs = []
    #             for t in range(iters):
    #                 x_batch = x_train[t * batch_size:(t + 1) * batch_size]
    #                 y_batch = y_train[t * batch_size:(t + 1) * batch_size]
    #                 _, lb = sess.run(
    #                     [infer_op, lower_bound],
    #                     feed_dict={n_particles: lb_samples,
    #                                x: x_batch, y: y_batch})
    #                 lbs.append(lb)
    #             print('Epoch {}: Lower bound = {}'.format(epoch, np.mean(lbs)))

    #             if epoch % test_freq == 0:
            test_lb,y_test_pred, ne = sess.run(
                        [lower_bound, y_pred, l2],
                        feed_dict={n_particles: ll_samples,
                                   x: x_test, y: y_test})
            l2_error = NA.norm((y_test_pred - y_test))/NA.norm(y_test)

            print('>> TEST')
            print('>> Test lower bound = {}, l2error={}, l2_internal={}'.format(test_lb, l2_error, ne))

        return y_test_pred 




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
	controls = WidgetBox(carrier_selection, range_select, bandwidth_select, bandwidth_choose, fund_input, contribution_input, income_input, fga_input, retirementage_select, planning_select, risk_select, age_select , b)

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

