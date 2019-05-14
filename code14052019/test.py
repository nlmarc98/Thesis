import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from GenerativeAgent import GenerativeAgent
from PSI_RiF import PSI_RiF
from plots import Plotter


# Marc Verwoert

# transforms sigma values into kappa values
def sig2kap(sig):  # in degrees
    sig2 = np.square(sig)
    return 3.9945e3 / (sig2 + 0.0226e3)

# M: Kappas 25
kappa_ver = np.linspace(sig2kap(2.3), sig2kap(7.4), 15)
# kappa_ver = [sig2kap(4.3)]
kappa_hor = np.linspace(sig2kap(28), sig2kap(76), 15)
# kappa_hor = [sig2kap(37)]
#tau = np.linspace(0.6, 1.0, 25)
tau = np.array([0.8])
# kappa_oto = np.linspace(sig2kap(1.4), sig2kap(3.0), 8)
kappa_oto = [sig2kap(2.2)]
print(kappa_oto)
# lapse = np.linspace(0.0, 0.1, 8)
# lapse = [0.0]

params = {'kappa_ver': kappa_ver,
          'kappa_hor': kappa_hor,
          'tau': tau,
          'kappa_oto': kappa_oto,
          }

kappa_ver_gen = sig2kap(4.3)
kappa_hor_gen = sig2kap(37)
tau_gen = 0.8
kappa_oto_gen = sig2kap(2.2)

params_gen = {'kappa_ver': kappa_ver_gen,
              'kappa_hor': kappa_hor_gen,
              'tau': tau_gen,
              'kappa_oto': kappa_oto_gen
              }

rods = np.array([-7, -4, -2, -1, 0, 1, 2, 4, 7]) * np.pi / 180
frames = np.linspace(-45, 40, 18) * np.pi / 180

# M: frame orientation 45 degrees to 40 or 45 intervals.
# M: 20 to 25 frames.
# M: Alberts et al. 18 frames.

stimuli = {'rods': rods, 'frames': frames}

# initialize generative agent
genAgent = GenerativeAgent(params_gen, stimuli)

# number of iterations of the experiment
iterations_num = 250

# initialize psi object
# M: Add iterations for prior tracker
psi = PSI_RiF(params, stimuli, iterations_num)

# initialize plotter and plot generative distribution, generative weights and the negative log likelihood
plotter = Plotter(params, params_gen, stimuli, genAgent, psi, iterations_num)
plotter.plotGenProbTable()
plotter.plotGenVariances()
plotter.plotGenWeights()
plotter.plotGenPSE()
plotter.plotNegLogLikelihood(responses_num=500)
plotter.plot()

# M: Normal adaptive use 'adaptive'
# M: Product entropy use 'product'
# M: Backward induction use 'backward'
# for stim_selection in ['adaptive', 'product', 'backward']:

for stim_selection in ['adaptive', 'product', 'backward']:
    # set stimulus selection mode and reset psi object to initial values
    psi.reset(stim_selection)

    # reset plotter to plot new figures
    plotter.reset()

    # run model for given number of iterations
    print 'inferring model ' + stim_selection + 'ly'

    for _ in trange(iterations_num):
        # get stimulus from psi object
        rod, frame = psi.stim

        # get response from the generative model
        response = genAgent.getResponses(rod, frame, 1)

        # plot selected stimuli
        plotter.plotStimuli()

        # M: Plot RMSE for thresholds; first argument is the predicted threshold,
        # second argument is the target threshold.
        # M: PSE from GenAg is the target threshold, the CDF PSE is the predicted threshold.
        # CDFArray = np.asarray(psi.CDFTable)
        # Reduce dimensionality by summing over all columns.
        # CDFArrayMean = CDFArray.mean(axis=0)
        # PSEArray = np.asarray(genAgent.PSETable)
        # plotter.plotRSME(CDFArray, PSEArray)
        # plotter.CDFPlotter(CDFArrayMean)
        # print('cdf', psi.CDFTable)

        # plotter.plotRSMELog()

        # plot updated parameter values based on mean and MAP
        plotter.plotParameterValues()

        # the parameter distributions may be plotted at most once (so comment out at least one)

        # plot parameter distributions of current trial
        # plotter.plotParameterDistributions()

        # plot parameter distributions of each trial as surfaces
        plotter.plotParameterDistributions(projection='3d')

        # the negative log likelihood may be plotted at most once (so comment out at least one)

        # plot negative log likelihood of responses thus far as a contour plot
        plotter.plotNegLogLikelihood()

        # plot negative log likelihood of responses thus far as a surface
        #plotter.plotNegLogLikelihood(projection='3d')

        # actually plot all the figures
        plotter.plot()

        # Add data to psi object
        psi.addData(response)

        # M: Keeping track of priors
        psi.trial += 1
        # print(stim_selection)
        # print('trial number', psi.trial)

        # print('Prior Array', psi.prior_array)
        # print('Prior Array length', psi.prior_array.__len__(), 'Prior Array shape', psi.prior_array.shape)

# M: Response tracker
        # print('Response Array', psi.response_array)
        # print('Response Array length', psi.response_array.__len__(), 'Response Array shape', psi.response_array.shape)

# M: Stimulus tracker
        # print('Stimulus Array', psi.stimulus_array)
        # print('Stimulus Array length', psi.stimulus_array.__len__(), 'Stimulus Array shape', psi.stimulus_array.shape)

    # do not close plots when program finishes
plt.show()

