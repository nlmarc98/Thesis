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


kappa_ver = np.linspace(sig2kap(2.3), sig2kap(7.4), 25)
# kappa_ver = [sig2kap(4.3)]
kappa_hor = np.linspace(sig2kap(28), sig2kap(76), 25)
# kappa_hor = [sig2kap(37)]
# tau = np.linspace(0.6, 1.0, 25)
tau = np.array([0.8])
# kappa_oto = np.linspace(sig2kap(1.4), sig2kap(3.0), 8)
kappa_oto = [sig2kap(2.2)]
# lapse = np.linspace(0.0, 0.1, 8)


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
frames = np.linspace(-45, 40, 20) * np.pi / 180

# M: frame orientation 45 degrees to 40 intervals.
# M: 20 to 25 frames.

stimuli = {'rods': rods, 'frames': frames}


# initialize generative agent
genAgent = GenerativeAgent(params_gen, stimuli)

# initialize psi object
psi = PSI_RiF(params, stimuli)

# number of iterations of the experiment
iterations_num = 500

# initialize plotter and plot generative distribution, generative weights and the negative log likelihood
plotter = Plotter(params, params_gen, stimuli, genAgent, psi, iterations_num)
plotter.plotGenProbTable()
plotter.plotGenVariances()
plotter.plotGenWeights()
plotter.plotGenPSE()
plotter.plotNegLogLikelihood(responses_num=500)
plotter.plot()

for stim_selection in ['adaptive', 'random']:
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

        # plot updated parameter values based on mean and MAP
        plotter.plotParameterValues()

        # the parameter distributions may be plotted at most once (so comment out at least one)

        # plot parameter distributions of current trial
        # plotter.plotParameterDistributions()

        # plot parameter distributions of each trial as surfaces
        plotter.plotParameterDistributions(projection='3d')

        # the negative log likelihood may be plotted at most once (so comment out at least one)

        # plot negative log likelihood of responses thus far as a contour plot
        # plotter.plotNegLogLikelihood()

        # plot negative log likelihood of responses thus far as a surface
        plotter.plotNegLogLikelihood(projection='3d')

        # actually plot all the figures
        plotter.plot()


        # add data to psi object
        psi.addData(response)

# do not close plots when program finishes
plt.show()
