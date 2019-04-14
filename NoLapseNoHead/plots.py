import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Marc Verwoert

class Plotter:
    def __init__(self, params, params_gen, stimuli, genAgent, psi, iterations_num, plot_period=500, point_size=5):
        # set the members of the Plotter object
        self.params = params
        self.params_gen = params_gen
        self.stimuli = stimuli
        self.iterations_num = iterations_num

        # save generative agent and psi object in Plotter object
        self.genAgent = genAgent
        self.psi = psi

        # number of trials before figure(s) is/are plotted again
        self.plot_period = plot_period

        # size of points in parameter values and selected stimuli figures
        self.point_size = point_size

        # reset plotter to draw new figures
        self.reset()


    def reset(self):
        # initialize selected stimuli, parameter values and distributions and the negative log likelihood as Nones.
        self.selected_stimuli = None
        self.param_values = None
        self.param_distributions = None
        self.neg_log_likelihood = None

        # reset current trial number
        self.trial_num = 1


    def plot(self):
        if self.__isTimeToPlot():
            # pause to let pyplot draw graphs
            plt.pause(0.0001)

        # increase current trial number
        self.trial_num += 1


    # returns whether or not the figures will be drawn to the screen
    def __isTimeToPlot(self):
        return self.trial_num == 1 or (self.trial_num % self.plot_period) == 0


    def plotGenProbTable(self):
        # initialize prob table figure and plot
        prob_table_figure = plt.figure(figsize=(8, 6))
        prob_table_plot = prob_table_figure.add_subplot(1, 1, 1)

        # plot prob table
        prob_table_plot.plot(self.stimuli['rods'] * 180 / np.pi, self.genAgent.prob_table)
        prob_table_plot.set_xlabel('rod ($^\circ$)')
        prob_table_plot.set_ylabel('P(right)')
        prob_table_plot.set_ylim(0.0, 1.0)
        prob_table_plot.set_title('Generative Rod Distribution for Each Frame Orientation')


    def plotGenVariances(self):
        # get variances from generative agent
        variances = self.genAgent.calcVariances()

        # initialize variances figure and plot
        variances_figure = plt.figure(figsize=(8, 6))
        variances_plot = variances_figure.add_subplot(1, 1, 1)

        # plot variances
        #variances_plot.plot(self.stimuli['frames'] * 180 / np.pi, variances['otoliths'], label='otoliths variance')
        variances_plot.plot(self.stimuli['frames'] * 180 / np.pi, variances['context'], label='visual context variance')
        variances_plot.set_xlabel('frame ($^\circ$)')
        variances_plot.set_ylabel('variance ($^\circ$)')
        variances_plot.set_title('Variances of Otoliths and Visual Contextual Information')
        variances_plot.legend()


    def plotGenWeights(self):
        # get weights from generative agent
        weights = self.genAgent.calcWeights()

        # initialize weights figure and plot
        weights_figure = plt.figure(figsize=(8, 6))
        weights_plot = weights_figure.add_subplot(1, 1, 1)

        # plot weights
        weights_plot.plot(self.stimuli['frames'] * 180 / np.pi, weights['otoliths'], label='otoliths weight')
        weights_plot.plot(self.stimuli['frames'] * 180 / np.pi, weights['context'], label='visual context weight')
        weights_plot.set_xlabel('frame ($^\circ$)')
        weights_plot.set_ylabel('weight')
        weights_plot.set_ylim(0.0, 1.0)
        weights_plot.set_title('Relative Weights of Otoliths and Visual Contextual Information')
        weights_plot.legend()


    def plotGenPSE(self):
        # get PSEs from generative agent in degrees
        PSE = self.genAgent.calcPSE()

        # initialize PSE figure and plot
        PSE_figure = plt.figure(figsize=(8, 6))
        PSE_plot = PSE_figure.add_subplot(1, 1, 1)

        # plot PSE
        PSE_plot.plot(self.stimuli['frames'] * 180 / np.pi, PSE)
        PSE_plot.set_xlabel('frame ($^\circ$)')
        PSE_plot.set_ylabel('PSE ($^\circ$)')
        PSE_plot.set_title('Point of Subjective Equivalence for Each Frame Orientation')


    def plotStimuli(self):
        if self.selected_stimuli is None:
            self.__initStimuliFigure()

        # add stimuli to self.selected_stimuli in degrees
        rod, frame = self.psi.stim
        self.selected_stimuli['rods'].append(rod * 180.0 / np.pi)
        self.selected_stimuli['frames'].append(frame * 180.0 / np.pi)

        # only plot every self.plot_period trials
        if self.__isTimeToPlot():
            # use shorter name
            plot = self.selected_stimuli_plot

            # plot selected stimuli
            plot.clear()
            plot.scatter(np.arange(self.trial_num), self.selected_stimuli['rods'], s=self.point_size, label='rod ($^\circ$)')
            plot.scatter(np.arange(self.trial_num), self.selected_stimuli['frames'], s=self.point_size, label='frame ($^\circ$)')
            plot.set_xlabel('trial number')
            plot.set_ylabel('selected stimulus ($^\circ$)')
            plot.set_xlim(0, self.iterations_num)
            plot.set_ylim(self.stim_min, self.stim_max)
            plot.set_title('Selected Stimuli for Each Trial')
            plot.legend()


    def __initStimuliFigure(self):
        # initialize selected stimuli plot
        selected_stimuli_figure = plt.figure(figsize=(8, 6))
        self.selected_stimuli_plot = selected_stimuli_figure.add_subplot(1, 1, 1)

        # initialize selected stimuli
        self.selected_stimuli = {'rods': [], 'frames': []}

        # initialize minimum and maximum stimulus values in degrees
        self.stim_min = np.amin(self.stimuli['frames']) * 180 / np.pi
        self.stim_max = np.amax(self.stimuli['frames']) * 180 / np.pi


    def plotParameterValues(self):
        if self.param_values is None:
            self.__initParameterValuesFigure()

        param_values_MAP = self.psi.calcParameterValues('MAP')
        param_values_mean = self.psi.calcParameterValues('mean')

        # add parameter values to self.param_values
        for param in self.params.keys():
            self.param_values['MAP'][param].append(param_values_MAP[param])
            self.param_values['mean'][param].append(param_values_mean[param])

        # only draw plots every self.plot_period trials
        if self.__isTimeToPlot():
            # draw each parameter's values plot
            for param in self.params.keys():
                self.__plotParemeterValues(param)

            # add a single legend to the figure
            handles, labels = self.param_values_plots[self.params.keys()[0]].get_legend_handles_labels()
            self.param_values_figure.legend(handles, labels, loc='upper right')

            # fit all the plots to the screen with no overlapping text
            self.param_values_figure.tight_layout()


    def __initParameterValuesFigure(self):
        # initialize calculated parameter values figure and plots
        self.param_values_figure = plt.figure(figsize=(15, 8))
        plots = [self.param_values_figure.add_subplot(2, 3, i) for i in [1, 2, 4, 5, 6]]

        # initialize parameter values and parameter values plots dictionaries
        self.param_values = {'MAP': {param: [] for param in self.params.keys()},
                             'mean': {param: [] for param in self.params.keys()}}
        self.param_values_plots = {param: plots[i] for param, i in zip(self.params.keys(), range(len(self.params)))}


    def __plotParemeterValues(self, param):
        # retrieve specific parameter plot from self.param_values_plots
        plot = self.param_values_plots[param]

        # plot specific parameter values
        plot.clear()
        plot.hlines(self.params_gen[param], 1, self.iterations_num, label='generative parameter value')
        plot.scatter(np.arange(self.trial_num), self.param_values['MAP'][param], s=self.point_size, label='parameter value based on MAP')
        plot.scatter(np.arange(self.trial_num), self.param_values['mean'][param], s=self.point_size, label='expected parameter value')
        plot.set_xlabel('trial number')
        plot.set_ylabel(param)
        plot.set_xlim(0, self.iterations_num)
        plot.set_title('Calculated %s Values for Each Trial' % param)


    def plotParameterDistributions(self, projection=None):
        if self.param_distributions is None:
            self.__initParemeterDistributionsFigure(projection)

        param_distributions = self.psi.calcParameterDistributions()

        # add parameter distributions to self.param_distributions
        for param in self.params.keys():
            self.param_distributions[param].append(param_distributions[param])

        # only draw plots every self.plot_period trials
        if self.__isTimeToPlot():
            # plot the plot for each parameter using the specified projection
            for param in self.params.keys():
                if projection == '3d':
                    self.__plotParameterDistributions3D(param)
                else:
                    self.__plotParameterDistributions(param)

            # add a single legend to the figure
            handles, labels = self.param_distributions_plots[self.params.keys()[0]].get_legend_handles_labels()
            self.param_distributions_figure.legend(handles, labels, loc='upper right')

            # fit all the plots to the screen with no overlapping text
            self.param_distributions_figure.tight_layout()


    def __initParemeterDistributionsFigure(self, projection):
        # initialize calculated parameter values plots
        self.param_distributions_figure = plt.figure(figsize=(15, 8))
        plots = [self.param_distributions_figure.add_subplot(2, 3, i, projection=projection) for i in [1, 2, 4, 5, 6]]

        # initialize parameter distributions and parameter distribution plots dictionaries
        self.param_distributions = {param: [] for param in self.params.keys()}
        self.param_distributions_plots = {param: plots[i] for param, i in zip(self.params.keys(), range(len(self.params)))}


    def __plotParameterDistributions3D(self, param):
        # retrieve specific parameter plot from self.param_distributions_plots
        plot = self.param_distributions_plots[param]

        # clear the plot
        plot.clear()

        # make data for generative parameter value surface
        param_gen = self.params_gen[param]
        X_gen = np.array([[0, self.iterations_num], [0, self.iterations_num]])
        Y_gen = np.array([[param_gen, param_gen], [param_gen, param_gen]])
        Z_gen = np.array([[0.0, 0.0], [1.0, 1.0]])

        # plot generative parameter value surface
        surface_gen = plot.plot_surface(X_gen, Y_gen, Z_gen, alpha=0.25, label='generative parameter value')

        # make data for parameter distribution surface
        X, Y = np.meshgrid(np.arange(self.trial_num), self.params[param])
        Z = np.array(self.param_distributions[param]).transpose()

        # plot parameter distribution surface
        surface = plot.plot_surface(X, Y, Z, label='parameter value distribution')

        # set the other plot settings
        plot.view_init(8, 340)
        plot.set_xlabel('trial number')
        plot.set_ylabel(param)
        plot.set_zlabel('P(%s = x)' % param)
        plot.set_xlim(0, self.iterations_num)
        plot.set_zlim(0.0, 1.0)
        plot.set_title('%s Distribution for Each Trial' % param)

        # I get an error when I do not use these four lines
        surface_gen._facecolors2d = surface_gen._facecolors3d
        surface_gen._edgecolors2d = surface_gen._facecolors3d
        surface._facecolors2d = surface._facecolors3d
        surface._edgecolors2d = surface._facecolors3d


    def __plotParameterDistributions(self, param):
        # retrieve specific parameter plot from self.param_distributions_plots
        plot = self.param_distributions_plots[param]

        # plot specific parameter distribution
        plot.clear()
        plot.vlines(self.params_gen[param], 0.0, 1.0, label='generative parameter value')
        plot.plot(self.params[param], self.param_distributions[param][-1], label='parameter value distribution')
        plot.set_xlabel(param)
        plot.set_ylabel('P(%s = x)' % param)
        plot.set_ylim(0.0, 1.0)
        plot.set_title('%s Distribution for Trial %d' % (param, self.trial_num))


    def plotNegLogLikelihood(self, projection=None, responses_num=1):
        if self.neg_log_likelihood is None:
            self.__initNegLogLikelihoodFigure(projection)

        if responses_num == 1:
            # data is latest response from generative agent
            data = self.genAgent.last_response
        else:
            # data is generated by generative agent
            data = self.genAgent.getAllResponses(responses_num)

        # get negative log likelihood from psi object given the data
        neg_log_likelihood = self.psi.calcNegLogLikelihood(data)
        self.neg_log_likelihood += neg_log_likelihood.reshape(self.neg_log_likelihood.shape)

        # only draw plots every self.plot_period trials
        if self.__isTimeToPlot():
            # plot negative log likelihood using the specified projection
            if projection == '3d':
                self.__plotNegLogLikelihood3D(responses_num)
            else:
                self.__plotNegLogLikelihood(responses_num)

            # add single legend to the figure
            handles, labels = self.neg_log_likelihood_plot.get_legend_handles_labels()
            self.neg_log_likelihood_figure.legend(handles, labels, loc='upper right')


    def __initNegLogLikelihoodFigure(self, projection):
        # initialize negative log likelihood figure and plot
        self.neg_log_likelihood_figure = plt.figure(figsize=(8, 6))
        self.neg_log_likelihood_plot = self.neg_log_likelihood_figure.add_subplot(1, 1, 1, projection=projection)

        # initialize the two free parameters, self.free_param1 and self.free_param2
        free_params = [param for param in self.params.keys() if len(self.params[param]) > 1]
        if len(free_params) == 2:
            self.free_param1, self.free_param2 = free_params
        else:
            raise Exception, 'model must have exactly two free parameters'

        # initialize negative log likelihood array
        self.neg_log_likelihood = np.zeros([len(self.params[self.free_param2]), len(self.params[self.free_param1])])


    def __plotNegLogLikelihood3D(self, responses_num):
        # use shorter name
        plot = self.neg_log_likelihood_plot

        # clear the plot
        plot.clear()

        # make data for generative parameter values line
        X_gen = [self.params_gen[self.free_param1], self.params_gen[self.free_param1]]
        Y_gen = [self.params_gen[self.free_param2], self.params_gen[self.free_param2]]
        Z_gen = [np.amin(self.neg_log_likelihood), np.amax(self.neg_log_likelihood)]

        # plot generative parameter values line
        plot.plot(X_gen, Y_gen, Z_gen, label='generative parameter values')

        # make data for negative log likelihood surface
        X, Y = np.meshgrid(self.params[self.free_param1], self.params[self.free_param2])
        Z = self.neg_log_likelihood

        # plot negative log likelihood surface
        surface = plot.plot_surface(X, Y, Z, label='negative log likelihood')

        # set the other plot settings
        plot.set_xlabel(self.free_param1)
        plot.set_ylabel(self.free_param2)
        plot.set_zlabel(('-logL(%s, %s)' % (self.free_param1, self.free_param2)))

        # set the title with a suffix dependent on the type of data
        if responses_num == 1:
            title_suffix = ' for Trial %d' % self.trial_num
        else:
            title_suffix = ' for %d Responses' % (len(self.stimuli['rods']) * len(self.stimuli['frames']) * responses_num)
        plot.set_title('Negative Log Likelihood of %s and %s' % (self.free_param1, self.free_param2) + title_suffix)

        # I get an error when I do not do these two lines
        surface._facecolors2d = surface._facecolors3d
        surface._edgecolors2d = surface._facecolors3d


    def __plotNegLogLikelihood(self, responses_num):
        # shorter name
        plot = self.neg_log_likelihood_plot

        # plot generative parameter values as a point and the negative log likelihood as a contour plot
        plot.clear()
        plot.plot(self.params_gen[self.free_param1], self.params_gen[self.free_param2], marker='o', label='generative parameter values')
        plot.contourf(self.params[self.free_param1], self.params[self.free_param2], self.neg_log_likelihood)
        plot.set_xlabel(self.free_param1)
        plot.set_ylabel(self.free_param2)

        # set the title with a suffix dependent on the type of data
        if responses_num == 1:
            title_suffix = ' for Trial %d' % self.trial_num
        else:
            title_suffix = ' for %d Responses' % (len(self.stimuli['rods']) * len(self.stimuli['frames']) * responses_num)
        plot.set_title('Negative Log Likelihood of %s and %s' % (self.free_param1, self.free_param2) + title_suffix)
