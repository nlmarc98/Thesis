import numpy as np
from scipy.stats import vonmises
from scipy.interpolate import splev, splrep

#Marc Verwoert

########## Side Notes

# M: A generative agent is made in order to give responses based on a predefined parameter set:
# M: It's responses are used to update the prior.
# M: Gen. agent is build once! (inbuilt prob. distr. of giving a cw response for every rod-and-frame orientation).


############


# transforms kappa values into sigma values in degrees
def kap2sig(kap):
    return np.sqrt((3994.5 / kap) - 22.6)


class GenerativeAgent:
    # Create generative parameters and initialize probability table
    def __init__(self, params, stimuli):
        # Initialize generative parameter values
        self.kappa_ver = params['kappa_ver']
        self.kappa_hor = params['kappa_hor']
        self.tau = params['tau']
        self.kappa_oto = params['kappa_oto']


        # Initialize stimulus grids
        self.rods = stimuli['rods']
        self.frames = stimuli['frames']

        # dimensions of the 2D stimulus space
        self.rod_num = len(self.rods)
        self.frame_num = len(self.frames)

        # initialize last response
        self.last_response = -1

        # pre-compute likelihood table
        print 'computing generative distribution\n'
        self.__makeProbTable()


    def __makeProbTable(self):
        # the rods I need for the cumulative density function
        theta_rod = np.linspace(-np.pi, np.pi, 10000)

        # allocate memory for the probability table
        self.prob_table = np.zeros([self.rod_num, self.frame_num])

        # the distribution of the otoliths
        P_oto = vonmises.pdf(theta_rod, self.kappa_oto)

        # compute kappas
        kappa1, kappa2 = self.__computeKappas()

        # for every frame orientation, calculate frame influence
        for i in range(self.frame_num):
            # the context provided by the frame
            P_frame0 = vonmises.pdf(theta_rod - self.frames[i], kappa1[i])
            P_frame90 = vonmises.pdf(theta_rod - np.pi/2 - self.frames[i], kappa2[i])
            P_frame180 = vonmises.pdf(theta_rod - np.pi - self.frames[i], kappa1[i])
            P_frame270 = vonmises.pdf(theta_rod - np.pi*3/2 - self.frames[i], kappa2[i])

            # initialize convolved distributions as P_frame
            P_frame = P_frame0 + P_frame90 + P_frame180 + P_frame270

            # cumulative response distribution per frame
            cdf = np.cumsum(P_frame * P_oto) / np.sum(P_frame * P_oto)

            # reduce cdf to |rods| using spline interpolation
            cdf_continuous = splrep(theta_rod, cdf, s=0)
            cdf = splev(self.rods, cdf_continuous, der=0)

            # add lapse probability to distribution
            PCW =  cdf

            # add probabilities to look-up table
            self.prob_table[:, i] = PCW


    def __computeKappas(self):
        kappa1 = self.kappa_ver -\
                 (1.0 - np.cos(np.abs(2.0 * self.frames))) *\
                 self.tau *\
                 (self.kappa_ver - self.kappa_hor)
        kappa2 = self.kappa_hor +\
                 (1.0 - np.cos(np.abs(2.0 * self.frames))) *\
                 (1.0 - self.tau) *\
                 (self.kappa_ver - self.kappa_hor)

        return kappa1, kappa2


    # determine response_num responses for each rod-frame pair
    def getAllResponses(self, responses_num):
        # initialize responses array
        responses = np.empty([self.rod_num, self.frame_num, responses_num])

        # determine response_num responses for each given rod and frame
        for i in range(self.rod_num):
            for j in range(self.frame_num):
                responses[i, j] = self.getResponses(self.rods[i], self.frames[j], responses_num)

        return responses


    # determine response_num responses of the generative agent on a given rod and frame orientation
    def getResponses(self, stim_rod, stim_frame, responses_num):
        # find index of stimulus
        idx_rod = np.where(self.rods == stim_rod)[0]
        idx_frame = np.where(self.frames == stim_frame)[0]

        # look up probability of responding clockwise
        PCW = self.prob_table[idx_rod, idx_frame][0]

        # determine responses
        responses = np.random.binomial(1, PCW, responses_num)

        # save last response
        self.last_response = responses[-1]

        return responses


    # calculate otoliths and visual context variances in degrees
    def calcVariances(self):
        # compute otoliths variance, repeated |frames| times
        otoliths_variance = np.repeat(kap2sig(self.kappa_oto), self.frame_num)

        # compute vertical visual context variance for each frame orientation
        kappa1, _ = self.__computeKappas()
        context_variance = kap2sig(kappa1)

        # return variances in dictionary
        return {'otoliths': otoliths_variance, 'context': context_variance}


    # calculate otoliths and visual context weights
    def calcWeights(self):
        # calculate otoliths and visual context variances in degrees
        variances = self.calcVariances()

        # calculate normalizing term
        denominator = (1.0 / variances['otoliths']) + (1.0 / variances['context'])

        # compute weights with equation given in Alberts et al. (2018), displayed in Alberts et al. (2017)
        weights = {'otoliths': (1.0 / variances['otoliths']) / denominator,
                   'context': (1.0 / variances['context']) / denominator}

        return weights


    # calculate PSE for each frame orientation in degrees
    def calcPSE(self):
        # initialize PSE array
        PSE = np.zeros(self.frame_num)

        # for each frame orientation, add PSE to array
        for i in range(self.frame_num):
            # point of subjective equivalence when probability is 0.5
            idx_rod = np.argmax(self.prob_table[:, i] > 0.5)

            # add the PSE rod given the current frame orientation
            PSE[i] = self.rods[idx_rod]

        # return PSE in degrees
        return PSE * 180 / np.pi
