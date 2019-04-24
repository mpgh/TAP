## -*- coding: utf-8 -*-
"""RoboTAP for REA

 Collection of routines to update the RoboNet database tables
 Keywords match the class model fields in ../robonet_site/events/models.py
"""

# Import dependencies
import warnings
import os
from operator import itemgetter
import time
import numpy as np
from jdcal import gcal2jd
from astropy import units as u
from astropy.coordinates import SkyCoord
import subprocess

#sync ARTEMiS
#cmd=['/usr/bin/bash','artemis.sync']
#subprocess.call(cmd)


field_dict={'ROME-FIELD-01':[ 267.835895375 , -30.0608178195 , '17:51:20.6149','-30:03:38.9442' ],
            'ROME-FIELD-02':[ 269.636745458 , -27.9782661111 , '17:58:32.8189','-27:58:41.758' ],
            'ROME-FIELD-03':[ 268.000049542 , -28.8195573333 , '17:52:00.0119','-28:49:10.4064' ],
            'ROME-FIELD-04':[ 268.180171708 , -29.27851275 , '17:52:43.2412','-29:16:42.6459' ],
            'ROME-FIELD-05':[ 268.35435 , -30.2578356389 , '17:53:25.044','-30:15:28.2083' ],
            'ROME-FIELD-06':[ 268.356124833 , -29.7729819283 , '17:53:25.47','-29:46:22.7349' ],
            'ROME-FIELD-07':[ 268.529571333 , -28.6937071111 , '17:54:07.0971','-28:41:37.3456' ],
            'ROME-FIELD-08':[ 268.709737083 , -29.1867251944 , '17:54:50.3369','-29:11:12.2107' ],
            'ROME-FIELD-09':[ 268.881108542 , -29.7704673333 , '17:55:31.4661','-29:46:13.6824' ],
            'ROME-FIELD-10':[ 269.048498333 , -28.6440675 , '17:56:11.6396','-28:38:38.643' ],
            'ROME-FIELD-11':[ 269.23883225 , -29.2716684211 , '17:56:57.3197','-29:16:18.0063' ],
            'ROME-FIELD-12':[ 269.39478875 , -30.0992361667 , '17:57:34.7493','-30:05:57.2502' ],
            'ROME-FIELD-13':[ 269.563719375 , -28.4422328996 , '17:58:15.2927','-28:26:32.0384' ],
            'ROME-FIELD-14':[ 269.758843 , -29.1796030365 , '17:59:02.1223','-29:10:46.5709' ],
            'ROME-FIELD-15':[ 269.78359875 , -29.63940425 , '17:59:08.0637','-29:38:21.8553' ],
            'ROME-FIELD-16':[ 270.074981708 , -28.5375585833 , '18:00:17.9956','-28:32:15.2109' ],
            'ROME-FIELD-17':[ 270.81 , -28.0978333333 , '18:03:14.4','-28:05:52.2' ],
            'ROME-FIELD-18':[ 270.290886667 , -27.9986032778 , '18:01:09.8128','-27:59:54.9718' ],
            'ROME-FIELD-19':[ 270.312763708 , -29.0084241944 , '18:01:15.0633','-29:00:30.3271' ],
            'ROME-FIELD-20':[ 270.83674125 , -28.8431573889 , '18:03:20.8179','-28:50:35.3666' ]}


def romecheck(radeg, decdeg):
    lhalf = 0.220833333333  # 26.5/(120.)
    field, rate = -1, -1
    fields = [[267.835895375, -30.0608178195, 64.0],
              [269.636745458, -27.9782661111, 49.0],
              [268.000049542, -28.8195573333, 46.0],
              [268.180171708, -29.27851275, 58.0],
              [268.35435, -30.2578356389, 64.0],
              [268.356124833, -29.7729819283, 90.0],
              [268.529571333, -28.6937071111, 72.0],
              [268.709737083, -29.1867251944, 83.0],
              [268.881108542, -29.7704673333, 83.0],
              [269.048498333, -28.6440675, 75.0],
              [269.23883225, -29.2716684211, 70.0],
              [269.39478875, -30.0992361667, 42.0],
              [269.563719375, -28.4422328996, 49.0],
              [269.758843, -29.1796030365, 67.0],
              [269.78359875, -29.63940425, 61.0],
              [270.074981708, -28.5375585833, 61.0],
              [270.81, -28.0978333333, -99.0],
              [270.290886667, -27.9986032778, 52.0],
              [270.312763708, -29.0084241944, 48.0],
              [270.83674125, -28.8431573889, 49.0]]
    for idx in range(len(fields)):
        if radeg < fields[idx][0] + lhalf and\
           radeg > fields[idx][0] - lhalf and\
           decdeg < fields[idx][1] + lhalf and\
           decdeg > fields[idx][1] - lhalf:
            return idx, fields[idx][2],'ROME-FIELD-'+str(idx+1).zfill(2)
    return field, rate, 'None'

def event_in_season(t0):
    if t0>7844.5 and t0<8026.5:
        return True
    elif t0>8209.5 and t0<8391.5:
        return True
    elif t0>8574.5 and t0<8756.5:
        return True
    else:
        return False

def read_events():
    modelpath='./model'
    dirlist=os.listdir(modelpath)
    dirlist=[x for x in dirlist if 'align' in x and 'B19' in x]
    active_events={}

    for entry in dirlist:
        event=str.split(entry,'.')[0]
        if os.path.exists(os.path.join(modelpath,event+'.align')):
            filein=open(os.path.join(modelpath,entry))
            for fentry in filein:
                if 'OI' in fentry and 'O' in event:
                    alignpars=str.split(fentry)
                if 'KI' in fentry and 'K' in event:
                    alignpars=str.split(fentry)
            filein.close()

            filein=open(os.path.join(modelpath,event+'.model'))
            psplpars=str.split(filein.readline())
            filein.close()
            gfac=abs(float(alignpars[2]))
            ibase=float(alignpars[1])
            ftot=10.0**(-0.4*ibase)
            fspar=ftot/(1.0+gfac)
            fbpar=gfac*fspar
            t0par=float(psplpars[3])
            tepar=float(psplpars[5])
            u0par=float(psplpars[7])

            c=SkyCoord(psplpars[0]+' '+psplpars[1],unit=(u.hourangle,u.deg))
            rapar=float(c.ra.deg)
            decpar=float(c.dec.deg)
            fld1,cad2,fname=romecheck(rapar, decpar)
            if fld1>-1:
                active_events[event]=[u0par,tepar,t0par,fspar,fbpar,fname]
    return active_events

def romerea_visibility_3sites_40deg(julian_date):
    """
    The ROME REA visibility function calculates the available
    time for the ROME field center for 3 sites (CTIO, SAAO, SSO)
    and an altitude limit of 40 degrees.
    The input requires the requested time in JD-2450000
    The output provides the visibility in hours;
    """
    yearoffset = round((7925. - julian_date) / 365.25)
    julian_date_reference_year = julian_date + yearoffset * 365.25
    par = [7.90148098e+03, 7.94700334e+03, 9.38610177e-03, -7.31977214e+01,
           9.74954459e-01, -9.32858229e-03, 7.51131184e+01]

    if julian_date_reference_year < par[0]:
        return min((julian_date_reference_year * par[2] + par[3]) * 24., par[4] * 24.)
    if julian_date_reference_year > par[1]:
        return min((julian_date_reference_year * par[5] + par[6]) * 24., par[4] * 24.)
    if julian_date_reference_year >= par[0] and julian_date_reference_year <= par[1]:
        return par[4] * 24.
    return 0.


def calculate_exptime_romerea(magin):
    """
    This function calculates the required exposure time
    for a given iband magnitude (e.g. OGLE I which also
    roughly matches SDSS i) based on a fit to the empiric
    RMS diagram of DANDIA light curves from 2016. The
    output is in seconds.
    """
    if magin < 14.7:
        mag = 14.7
    else:
        mag = magin
    lrms = 0.14075464 * mag * mag - 4.00137342 * mag + 24.17513298
    snr = 1.0 / np.exp(lrms)
    # target 4% -> snr 25
    return round((25. / snr)**2 * 300., 1)


def omegarea(time_requested, u0_pspl, te_pspl, t0_pspl, fs_pspl, fb_pspl):
    """
    This function calculates the priority for ranking
    microlensing events based on the planet probability psi
    as defined by Dominik 2009 and estimates the cost of
    observations based on an empiric RMS estimate
    obtained from a DANDIA reduction of K2C9 Sinistro
    observations from 2016. It expects the overhead to be
    60 seconds and also return the current Paczynski
    light curve magnification.
    """
    usqr = u0_pspl**2 + ((time_requested - t0_pspl) / te_pspl)**2
    pspl_deno = (usqr * (usqr + 4.))**0.5
    psip = 4.0 / (pspl_deno) - 2.0 / (usqr + 2.0 + pspl_deno)
    amp = (usqr + 2.) / pspl_deno
    mag = -2.5 * np.log10(fs_pspl * amp + fb_pspl)
    # 60s overhead
    return psip / (0.63 * calculate_exptime_romerea(mag) + 60.), amp,psip


def psplrea(u):
    """
    Calculates the magnification for a given source-lens
    separation u (PSPL)
    """
    usqr = float(u)**2
    pspl_deno = (usqr * (usqr + 4.))**0.5
    amp = (usqr + 2.) / pspl_deno
    return amp


def assign_tap_priorities(logger):
    """
    This function runs TAP and updates entries in the database.
    It only calculates the priority for active events if the reported Einstein time
    stays below 210 days assuming that ROME observations will
    characterise the event sufficiently. For the start events with A>50 are triggered
    as anomalies. TAP itself does not request observations, but sets flags
    pointing to relevant events. Blending and baseline parameters live with the
    datafile and need to be present for a succesful run. Currently, MOA parameters
    are not provided by ARTEMiS and require another processing step.
    """

    ut_current = time.gmtime()
    t_current = gcal2jd(ut_current[0], ut_current[1], ut_current[2])[
        1] - 49999.5 + ut_current[3] / 24.0 + ut_current[4] / (1440.)
    full_visibility = romerea_visibility_3sites_40deg(t_current)
    daily_visibility = 2.8 * full_visibility * 300. / 3198.

    tap=[]
    for event in active_events:
        u0_pspl = active_events[event][0]
        te_pspl = active_events[event][1]
        t0_pspl = active_events[event][2]
        fs_pspl = active_events[event][3]
        fb_pspl = active_events[event][4]
        omega_now, amp_now,psip_now = omegarea(
                t_current, u0_pspl, te_pspl, t0_pspl, fs_pspl, fb_pspl)
        err_omega = 0.
        omega_peak, amp_peak,psip_peak = omegarea(
                t0_pspl, u0_pspl, te_pspl, t0_pspl, fs_pspl, fb_pspl)

        # SAMPLING TIME FOR REA IS 1h
        tsamp = 1.
        imag = -2.5 * np.log10(fs_pspl * amp_now + fb_pspl)
        texp = calculate_exptime_romerea(imag)
        wcost1m = daily_visibility / tsamp * ((60. + texp) / 60.)
        err_omega = 0.
        ibase_pspl = -2.5 * np.log10(fs_pspl + fb_pspl)
        if event_in_season(t0_pspl):# and amp_now>1.34:

            if te_pspl<300.:
                print(event,fb_pspl/fs_pspl)
                tap.append([event,u0_pspl,amp_now,omega_now,texp,active_events[event][5],psip_now,ibase_pspl])
            else:
                print(event,' g= ',fb_pspl/fs_pspl)
    sortedtap=sorted(tap,key=itemgetter(3),reverse=True)
    for entry in sortedtap:
        print(entry[0],entry[1],entry[2],entry[3])
    return sortedtap
    #lock_state = log_utilities.lock(script_config, 'unlock', log)
    #log_utilities.end_day_log(log)

def run_tap_prioritization(sortedlist):

    """
    Sort events on RoboTAP and check a request can be made with 
    an assumed REA-LOW time allocation of 300 hours.
    For very high priority events A_now>500, the anomaly status
    can be set (not implemented yet).
    """

    ut_current = time.gmtime()
    t_current = gcal2jd(ut_current[0], ut_current[1], ut_current[2])[
        1] - 49999.5 + ut_current[3] / 24.0 + ut_current[4] / (1440.)
    full_visibility = 100000000.# romerea_visibility_3sites_40deg(t_current)
    daily_visibility = 100000000.#1.4 * full_visibility * 300. / 3198.*10
    full_visibility = romerea_visibility_3sites_40deg(t_current)
    daily_visibility = 2.8 * full_visibility * 300. / 3198.
    toverhead = 60.
    trun = 0.
    print(t_current)
    # CHECK ALLOCATED TIME AND SET MONITOR
    for idx in range(len(sortedlist)):
        tsys = 24. * (float(sortedlist[idx][4]) + toverhead) / 3600.
        if trun + tsys < daily_visibility:
            print('TO BE QUEUED',sortedlist[idx][0],sortedlist[idx][1],sortedlist[idx][2],sortedlist[idx][3],sortedlist[idx][4],field_dict[sortedlist[idx][5]],sortedlist[idx][6],field_dict[sortedlist[idx][5]],sortedlist[idx][7])
            trun = trun + tsys
        else:
            if trun + tsys < 5.*daily_visibility:
                print('REJECTED (visibility)',sortedlist[idx][0],sortedlist[idx][1],sortedlist[idx][2],sortedlist[idx][3],sortedlist[idx][4],field_dict[sortedlist[idx][5]],sortedlist[idx][6])


if __name__ == '__main__':
    active_events=read_events()
    print(active_events)
    eventlist=assign_tap_priorities(active_events)
    print(eventlist)
    run_tap_prioritization(eventlist)
