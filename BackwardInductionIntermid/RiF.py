#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:24:57 2019

@author: Luc
"""

import numpy as np
from scipy.stats import vonmises, norm
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from numpy.matlib import repmat

def generativeModel(a_oto,b_oto,sigma_prior,kappa_ver, kappa_hor,tau,frames,rods):
    
    # this is actually also a stimulus input (so similar to frames and rods)
    theta_head = [0*np.pi/180, 30*np.pi/180]
    
    # you might want to run a loop over head-angles
    j=1
    
    # Aocr is normally a free parameter (the uncompensated ocular counterroll)
    Aocr = 14.6*np.pi/180 # convert to radians and fixed across subjects 
    
    # compute the number of stimuli
    frame_num = len(frames)
    rod_num = len(rods)
    # head_num = len(head)

    # the theta_rod I need at high density for the cumulative density function
    theta_rod=np.linspace(-np.pi,np.pi,10000)
    
    
    # allocate memory for the lookup table (P) and for the MAP estimate
    P = np.zeros([rod_num,frame_num])
    MAP = np.zeros([frame_num])
    mu = np.zeros([frame_num])
    
    #P = np.zeros([rod_num,frame_num,head_num])
    
    
    # move through the frame vector 
    for i in range(0,frame_num):
        # the frame in retinal coordinates
        frame_retinal = -(frames[i]-theta_head[j])-Aocr*np.sin(theta_head[j]);
        # make sure we stay in the -45 to 45 deg range
        if frame_retinal > np.pi/4:
           frame_retinal = frame_retinal - np.pi/2;
        elif frame_retinal < -np.pi/4:
           frame_retinal = frame_retinal + np.pi/2;
        
        # compute how the kappa's changes with frame angle
        kappa1 = kappa_ver-(1-np.cos(np.abs(2*frame_retinal)))*tau*(kappa_ver-kappa_hor)
        kappa2 = kappa_hor+(1-np.cos(np.abs(2*frame_retinal)))*(1-tau)*(kappa_ver-kappa_hor)
    
       
        # probability distributions for the four von-mises
        P_frame1 = vonmises.pdf(-theta_rod+frame_retinal,kappa1)
        P_frame2 = vonmises.pdf(-theta_rod+np.pi/2+frame_retinal,kappa2)
        P_frame3 = vonmises.pdf(-theta_rod+np.pi+frame_retinal,kappa1)
        P_frame4 = vonmises.pdf(-theta_rod+3*np.pi/2+frame_retinal,kappa2)
        
                
        # add the probability distributions
        P_frame = (P_frame1+P_frame2+P_frame3+P_frame4)
        P_frame = P_frame/np.sum(P_frame) # normalize to one
                    
        # the otoliths have head tilt dependent noise (note a and b switched from CLemens et a;. 2009)
        #print(a_oto+b_oto*theta_head[j])
    
        P_oto = norm.pdf(theta_rod,theta_head[j],a_oto+b_oto*theta_head[j])
        
        # the prior is always oriented with gravity
        P_prior = norm.pdf(theta_rod,0,sigma_prior)
        
                        
        # compute the (cumulative) density of all distributions convolved
        # NOTE THIS IS THE HEAD ORIENTATION IN SPACE!
        M=np.multiply(np.multiply(P_oto, P_frame),P_prior)
        cdf=np.cumsum(M)/np.sum(M)
        
        # now shift the x-axis, to make it rod specific
        E_s_cumd = theta_rod-theta_head[j]+Aocr*np.sin(theta_head[j]);
        
        # now use a spline interpolation to get a continuous P(theta)
        
        
        spline_coefs=interp.splrep(E_s_cumd,cdf, s = 0)
        
        P[:,i] = interp.splev(rods,spline_coefs, der = 0)
        
        
        
        # find the MAP
        index = np.argmax(M)
        MAP[i]=-E_s_cumd[index] # negative sign to convert to 'on retina'
        index = np.argmax(cdf>0.5)
        mu[i] =-E_s_cumd[index]
                  
        # construct the P(right) matrix
        #for k in range(0,rod_num):
        #    index = np.argmax(E_s_cumd>rods[k])
        #    P[k][i]=cdf[index]
    return P, MAP, mu


# stimuli and generative parameters
nframes = 251
nrods = 11111

# stimuli should be all in radians
rods= np.linspace(-15,15.0,nrods)*np.pi/180
frames = np.linspace(-45,45,nframes)*np.pi/180.0

# parameters (in radians, but not for b_oto)
a_oto = 3.2*np.pi/180
b_oto = 0.12

sigma_prior = 10.0*np.pi/180
kappa_ver =  45.0
kappa_hor = 1.45 
tau =0.9

P, MAP, mu =generativeModel(a_oto,b_oto,sigma_prior,kappa_ver,kappa_hor,tau,frames,rods)
plt.plot(rods*180/np.pi,P)
plt.xlabel('rod [deg]')
plt.ylabel('P(right)')

plt.figure()
frames_new=frames[:,np.newaxis]
rods_new=rods[:,np.newaxis]
plt.contourf(repmat(rods_new*180/np.pi,1,nframes),repmat(frames_new.transpose()*180/np.pi,nrods,1),P)
plt.xlabel('rod [deg]')
plt.ylabel('frame [deg]')

PSE = np.zeros(len(frames))
for k in range(0,len(frames)):
     index = np.argmax(P[:,k]>0.5)
     print(index)
     PSE[k]=-rods[index]
     
plt.figure()
plt.plot(frames*180/np.pi,MAP*180/np.pi,frames*180/np.pi,mu*180/np.pi)
