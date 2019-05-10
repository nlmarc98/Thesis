import numpy as np
from scipy.stats import vonmises, beta
from scipy.interpolate import splev, splrep
from sklearn.utils.extmath import cartesian
from tqdm import trange
from GenerativeAgent import GenerativeAgent
from scipy.stats import entropy

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import scipy as sc
import matplotlib.pyplot as plt
from plots import Plotter


# Marc Verwoert

class PSI_RiF:
    # Create parameter space and initialize prior, likelihood and stimulus
    def __init__(self, params, stimuli, ntrials, stim_selection='adaptive'):
        # initialize parameter grids
        self.kappa_ver = params['kappa_ver']
        self.kappa_hor = params['kappa_hor']
        self.tau = params['tau']
        self.kappa_oto = params['kappa_oto']

        # M: Look-up tables in which optimal stimuli and expected info gains will be stored.
        self.entropyLookUp = []
        # self.entropyLookUp2 = []
        self.backwardTable = []

        # M: CDF look up table.
        self.CDFTable = []
        self.CDFArray = np.array([])

        # Initialize stimulus grids
        self.rods = stimuli['rods']
        self.frames = stimuli['frames']

        # dimensions of the parameter space
        self.kappa_ver_num = len(self.kappa_ver)
        self.kappa_hor_num = len(self.kappa_hor)
        self.tau_num = len(self.tau)
        self.kappa_oto_num = len(self.kappa_oto)

        # dimensions of the 2D stimulus space
        self.rod_num = len(self.rods)
        self.frame_num = len(self.frames)

        # M: Keeping track of priors in a table.
        self.ntrials = ntrials
        self.stimulus_array = np.ndarray((self.ntrials, 2))
        self.response_array = np.ndarray((self.ntrials, 1))
        self.prior_array = np.ndarray(
            (self.ntrials, self.kappa_hor_num, self.kappa_ver_num, self.tau_num, self.kappa_oto_num))

        # compute initial prior
        print 'computing prior'
        self.__computePrior()

        # pre-compute likelihood
        print 'computing likelihood'
        self.__computeLikelihood()

        # compute easier-to-use parameter data-structure
        print "computing parameter values cartesian product"
        self.__computeTheta()

        # reset psi object to initial values
        self.reset(stim_selection)

    def __computeLikelihood(self):
        # the rods I need for the cumulative density function
        theta_rod_num = 10000
        theta_rod = np.linspace(-np.pi, np.pi, theta_rod_num)

        # allocate memory for the lookup table (P)
        P = np.zeros([self.kappa_ver_num, self.kappa_hor_num, self.tau_num, self.kappa_oto_num,
                      self.rod_num, self.frame_num])

        # initialize otolith distributions before for-loops
        P_oto = [self.__calcPOto(kappa_oto, theta_rod) for kappa_oto in self.kappa_oto]
        teller=0
        #self.CDFTable=np.zeros([0,18])
        for i in trange(self.kappa_ver_num):
            for j in range(self.kappa_hor_num):
                for k in range(self.tau_num):
                    # compute the 2D rod-frame distribution for the given kappas, tau and rods
                    P_frame = self.__calcPFrame(self.kappa_ver[i], self.kappa_hor[j], self.tau[k], theta_rod)

                    for l in range(self.kappa_oto_num):
                        # compute the cumulative density of all distributions convolved
                        cdf = np.cumsum(P_frame * P_oto[l], 0) / np.sum(P_frame * P_oto[l], 0)

                        # M: initialize PSE array
                        PSE = np.zeros(self.frame_num)

                        # # M: for each frame orientation, add PSE to array
                        # for pi in range(self.frame_num):
                        #     # M: point of subjective equivalence when probability is 0.5
                        #     idx_rod = np.argmax(cdf[:, pi] > 0.5)
                        #     # M: add the PSE rod given the current frame orientation
                        #     PSE[pi] = theta_rod[idx_rod]
                        #     # M: Add the target PSE to the PSE table.
                        #     self.CDFTable.append(PSE)
                        #     # print('CDFTable', self.CDFTable)

                        # reduce cdf to |rods|, |frames| by using spline interpolation
                        cdf = self.__reduceCDF(cdf, theta_rod)

                        # teller += 1
                        PCW = cdf


                        # add distribution to look-up table
                        P[i, j, k, l] = PCW

        # reshape to |param_space|, |rods|, |frames|
        self.lookup = np.reshape(P,
                                 [self.kappa_ver_num * self.kappa_hor_num * self.tau_num * self.kappa_oto_num,
                                  self.rod_num, self.frame_num],
                                 order="F")

    def __calcPFrame(self, kappa_ver, kappa_hor, tau, theta_rod):
        # computes kappas
        kappa1 = kappa_ver - \
                 (1 - np.cos(np.abs(2 * self.frames))) * \
                 tau * \
                 (kappa_ver - kappa_hor)
        kappa2 = kappa_hor + \
                 (1 - np.cos(np.abs(2 * self.frames))) * \
                 (1 - tau) * \
                 (kappa_ver - kappa_hor)

        # for every frame orientation, calculate frame influence
        P_frame = np.empty([len(theta_rod), self.frame_num])
        for i in range(self.frame_num):
            # the context provided by the frame
            P_frame0 = vonmises.pdf(theta_rod - self.frames[i], kappa1[i])
            P_frame90 = vonmises.pdf(theta_rod - np.pi / 2 - self.frames[i], kappa2[i])
            P_frame180 = vonmises.pdf(theta_rod - np.pi - self.frames[i], kappa1[i])
            P_frame270 = vonmises.pdf(theta_rod - np.pi * 3 / 2 - self.frames[i], kappa2[i])

            # add convolved distributions to P_frame
            P_frame[:, i] = P_frame0 + P_frame90 + P_frame180 + P_frame270

        return P_frame

    def __calcPOto(self, kappa_oto, theta_rod):
        # a simple von Mises distribution centered at 0 degrees
        return vonmises.pdf(theta_rod, kappa_oto).reshape(len(theta_rod), 1)

    def __reduceCDF(self, cdf, theta_rod):
        # initialize reduced cdf with dimensions |rods|, |frames|
        cdf_reduced = np.zeros([self.rod_num, self.frame_num])

        # for every frame orientation, calculate cumulative prob for rods in self.rods
        for i in range(self.frame_num):
            # use spline interpolation to get a continuous cdf
            cdf_continuous = splrep(theta_rod, cdf[:, i], s=0)

            # select cumulative probs of rods in self.rods from continuous cdf
            cdf_reduced[:, i] = splev(self.rods, cdf_continuous, der=0)

        return cdf_reduced

    def __computeTheta(self):
        # all the combinations of all parameter values
        self.theta = cartesian([self.kappa_ver, self.kappa_hor, self.tau, self.kappa_oto]).transpose()


    def reset(self, stim_selection='adaptive'):
        # M: Trial tracker initialized
        self.trial = 0

        # calculate best next stimulus with lowest entropy or a random stimulus based on self.stim_selection
        self.stim_selection = stim_selection
        self.__calcNextStim()

    def __computePrior(self):
        # compute parameter priors
        kappa_ver_prior = self.__computeUniformPrior(self.kappa_ver)
        kappa_hor_prior = self.__computeUniformPrior(self.kappa_hor)
        kappa_oto_prior = self.__computeUniformPrior(self.kappa_oto)

        # all the combinations of all parameter prior probabilities
        theta_prior = cartesian([kappa_ver_prior, kappa_hor_prior, kappa_oto_prior])

        # turn combinations in 1D array of size |param_space| which sums to 1
        self.prior = np.prod(theta_prior, 1)

    # uniform discrete prior
    def __computeUniformPrior(self, param):
        return np.ones(len(param)) / len(param)

    # M: Max info gain upon next two trials.
    def __calcNextStim(self):
        # compute posterior

        self.paxs = np.einsum('i,ijk->ijk', self.prior, self.lookup)
        self.paxf = np.einsum('i,ijk->ijk', self.prior, 1.0 - self.lookup)

        # probabilities of rod and frame orientations
        ps = np.sum(self.paxs, 0)
        pf = np.sum(self.paxf, 0)

        # normalize posterior
        self.paxs = np.einsum('jk,ijk->ijk', 1.0 / ps, self.paxs)
        self.paxf = np.einsum('jk,ijk->ijk', 1.0 / pf, self.paxf)

        # M: Determine next stimulus adaptively, or via product entropy or via backward induction.
        # M: Determine next stimulus only on the next trial (k=1), or more (k>1).
        if self.stim_selection == 'adaptive':
            self.stim1_index, self.stim2_index = self.__calcAdaptiveStim(ps, pf)
        elif self.stim_selection == 'product':
            # M: k = self.ntrials if you want to do k-trial-ahead optimization with looking all trials ahead.
            self.stim1_index, self.stim2_index = self.__calcAdaptiveStimMultipleTrials(ps, pf, self.ntrials)
        elif self.stim_selection == 'backward':
            # M: k = self.ntrials if you want to do k-trial-ahead optimization with looking all trials ahead.
            self.stim1_index, self.stim2_index = self.__calcAdaptiveBackwardV2Trials(ps, pf, self.ntrials)
        else:
            raise Exception, 'undefined stimulus selection mode: ' + self.stim_selection

        self.stim = (self.rods[self.stim1_index], self.frames[self.stim2_index])

        # M: Add stimuli
        self.stimulus_array[self.trial, :] = self.stim
        # M: Add priors
        self.prior_array[self.trial, :, :, :, :] = np.reshape(self.prior, (
            self.kappa_hor_num, self.kappa_ver_num, self.tau_num, self.kappa_oto_num))

    def __calcAdaptiveStim(self, ps, pf):
        # cannot take the log of 0
        self.paxs[self.paxs == 0.0] = 1.0e-10
        self.paxf[self.paxf == 0.0] = 1.0e-10

        # compute expected entropy
        hs = np.einsum('ijk,ijk->jk', -self.paxs, np.log(self.paxs))
        hf = np.einsum('ijk,ijk->jk', -self.paxf, np.log(self.paxf))
        h = ps * hs + pf * hf

        # determine stimulus with smallest expected entropy
        return np.unravel_index(h.argmin(), h.shape)

    def __calcAdaptiveStimMultipleTrials(self, ps, pf, k):
        # M: k = self.ntrials if you want to do k-ahead optimization with looking all trials ahead.

        # M: Need to fill the table for trial t + 1,
        # such that the table isn't empty when used for the first time in the loop.

        # cannot take the log of 0
        self.paxs[self.paxs == 0.0] = 1.0e-10
        self.paxf[self.paxf == 0.0] = 1.0e-10

        # M: compute expected entropy for each stimulus X at trial t + 1
        hs = np.einsum('ijk,ijk->jk', -self.paxs, np.log(self.paxs))
        hf = np.einsum('ijk,ijk->jk', -self.paxf, np.log(self.paxf))
        h = ps * hs + pf * hf

        # M: Add all expected entropies for each stimulus X to a table entropyLookUp of trial t + 1
        self.entropyLookUp.append(h)

        # self.ntrials += 1

        # M: For the second-to-last trial we compute the optimal stimulus.
        # M: Since the table is already filled once, we can start the loop one trial further.
        for y in xrange(1, k):
            #print('loop 1 enter')
            for x in xrange(self.entropyLookUp.__len__()):
                #print('loop 2 enter')
                # cannot take the log of 0
                self.paxs[self.paxs == 0.0] = 1.0e-10
                self.paxf[self.paxf == 0.0] = 1.0e-10

                # M: Compute expected entropy h for h(t+1) via h(t+1) = h(t+1) * (h(1),...,h(t-1))
                # M: Thus we calculate the expected entropy but now for stimulus X at trial t+1
                hs = np.einsum('ijk,ijk->jk', -self.paxs, np.log(self.paxs))
                hf = np.einsum('ijk,ijk->jk', -self.paxf, np.log(self.paxf))
                h = ps * hs + pf * hf
                # M: Then we multiply the expected entropy at trial t+1 with
                # M: the expected product entropy of all previous computed trials (each table element, x-1).
                # M: For this we use the fact that whatever transition to the next trial we made,
                # M: the maximized expected information gain
                # M: from that state on can be looked up from the table previously filled,
                # M: since it contains the product of all previous trial expected entropies.

                h = h * self.entropyLookUp[x-1]
                del self.entropyLookUp[:]
                # self.entropyLookUp=[]

                self.entropyLookUp.append(h)

        # M: Determine the stimulus with smallest expected product entropy.
        return np.unravel_index(h.argmin(), h.shape)

    def __calcAdaptiveBackwardV2Trials(self, ps, pf, k):
        # M: k = self.ntrials if you want to do k-trial-ahead optimization with looking all trials ahead.

        # M: Compute for all possible states in the t-to-last trial, or trial t + k - 1
        # the optimal stimulus using one-trial ahead optimization:  Need to fill the table,
        # such that the table isn't empty when used for the first time in the loop.

        # cannot take the log of 0
        self.paxs[self.paxs == 0.0] = 1.0e-10
        self.paxf[self.paxf == 0.0] = 1.0e-10

        # M: compute expected entropy for each stimulus X at trial t + k - 1
        hs = np.einsum('ijk,ijk->jk', -self.paxs, np.log(self.paxs))
        hf = np.einsum('ijk,ijk->jk', -self.paxf, np.log(self.paxf))
        h = ps * hs + pf * hf

        # M: Add all expected entropies for each stimulus X to a table backwardTable of trial t + (k - 1)
        self.backwardTable.append(h)

        # M: For the second-to-last trial we compute the optimal stimulus.
        # M: Since the table is already filled once, we can start the loop one trial further.

        # M: Then we compute for all possible states in the t-to-second-to-last trial.
        # the optimal stimulus using one-trial ahead optimization,
        # using backwards induction (we go one trial backwards).

        for y in xrange(1, k):
            # print('loop 1 entered')
            for x in xrange(self.backwardTable.__len__()):
                # print('loop 2 entered')
                k += -1
                # print ('k is', k)
                # cannot take the log of 0
                self.paxs[self.paxs == 0.0] = 1.0e-10
                self.paxf[self.paxf == 0.0] = 1.0e-10

                # M: We multiply the expected entropy at trial t + k - i (i is amount of trials we went backwards) with
                # M: the expected product entropy of all previous computed trials (each table element).
                # M: For this we use the fact that whatever transition to the next trial we made,
                # the maximized expected information gain from that state on,
                # can be looked up from the table previously filled (x-1),
                # since it contains the product of all previous trial expected entropies.

                hs = np.einsum('ijk,ijk->jk', -self.paxs, np.log(self.paxs))
                hf = np.einsum('ijk,ijk->jk', -self.paxf, np.log(self.paxf))
                h = ps * hs + pf * hf * self.backwardTable[x-1]

                # M: Clear the table from the previous filled values.
                # self.backwardTable = []
                del self.backwardTable[:]
                # M: Add all new expected entropies to the table.
                self.backwardTable.append(h)

        # M: Determine the stimulus with smallest expected product entropy.
        return np.unravel_index(h.argmin(), h.shape)

    def addData(self, response):
        # update prior based on response
        if response == 1:
            self.prior = self.paxs[:, self.stim1_index, self.stim2_index]
        elif response == 0:
            self.prior = self.paxf[:, self.stim1_index, self.stim2_index]
        else:
            raise Exception, 'response is ' + str(response) + ', but must be 1 or 0'

        # M: Add responses
        self.response_array[self.trial, :] = response

        # update stimulus based on posterior
        self.__calcNextStim()

    def calcParameterValues(self, mode='mean'):
        if mode == 'MAP':
            param_values = self.__calcParameterValuesMAP()
        elif mode == 'mean':
            param_values = self.__calcParameterValuesMean()
        else:
            raise Exception, 'undefined parameter value calculation mode: ' + mode

        # put parameter values in dictionary
        param_values_dict = {'kappa_ver': param_values[0],
                             'kappa_hor': param_values[1],
                             'tau': param_values[2],
                             'kappa_oto': param_values[3],
                             }

        return param_values_dict

    # calculate posterior parameter values based on MAP
    def __calcParameterValuesMAP(self):
        return self.theta[:, np.argmax(self.prior)]


    # calculate expected posterior parameter values
    def __calcParameterValuesMean(self):
        return np.matmul(self.theta, self.prior)

    def calcParameterDistributions(self):
        # M: SELF NOTE: check here wrt m in einsum
        # get posterior in right shape
        posterior = self.prior.reshape([self.kappa_ver_num, self.kappa_hor_num, self.tau_num, self.kappa_oto_num])

        param_distributions = []
        for axis in 'ijkl':
            # calculate marginalized posterior for one parameter
            param_distribution = np.einsum('ijkl->' + axis, posterior)

            # add parameter distribution to param_distributions
            param_distributions.append(param_distribution)

        # put parameter distributions in dictionary
        param_distributions_dict = {'kappa_ver': param_distributions[0],
                                    'kappa_hor': param_distributions[1],
                                    'tau': param_distributions[2],
                                    'kappa_oto': param_distributions[3]
                                    }

        return param_distributions_dict

    def calcNegLogLikelihood(self, data):
        if isinstance(data, np.ndarray):
            # compute negative log likelihood for all right, respectively left responses
            neg_log_likelihood_right_responses = np.einsum('ijk,jkl->i', -np.log(self.lookup), data)
            neg_log_likelihood_left_responses = np.einsum('ijk,jkl->i', -np.log(1.0 - self.lookup), 1.0 - data)

            # compute negative log likelihood for all responses
            neg_log_likelihood = neg_log_likelihood_right_responses + neg_log_likelihood_left_responses
            self.CDFTable.append(neg_log_likelihood)
            return neg_log_likelihood
        else:
            # compute negative log likelihood for one response
            if data == 1:
                return -np.log(self.lookup[:, self.stim1_index, self.stim2_index])
            elif data == 0:
                return -np.log(1.0 - self.lookup[:, self.stim1_index, self.stim2_index])
            else:
                raise Exception, 'response is ' + str(data) + ', but must be 1 or 0'

