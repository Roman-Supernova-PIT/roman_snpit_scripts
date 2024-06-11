#!/usr/bin/env python
#
# Created Oct 4 2023 by R.Kessler
#
# Read Roman filter transmissions from multi-column file
# (downloaded from
#   https://github.com/GalSim-developers/GalSim/blob/releases/2.5/share/roman/Roman_effarea_20210614.txt
# + compute zero point avg/min/max/range. 
# + Optionally write filter-transmission files for SNANA. 
#
# Note that wavelength units are microns and are converted here to Angstroms. 
# Original use was for the Roman-DESC Image-sim project.
#
# May 2024: 
#   + update with zeropoint calculation based on effective area and AB spectrum.
# 
# Jun 10 2024
#   + specify effective area file as wildcard and loop over all 18 SCAs;
#     Report average/min/max/stddev.
#
# ===================

import os, sys, glob
import numpy as np

#TRANS_INPUT   =  "Roman_effarea_20210614.txt"
TRANS_INPUT   =  "Roman_effarea_tables_20240327/Roman_effarea_v8_SCA*_20240301.ecsv"

PREFIX_OUTPUT = "ROMAN"

AREA_FUDGE  = 1.0E4  # /3.757   # from M.Jarvis slack

WRITE_SNANA_FILTER_TRANS = False

BAND_MAP = {
    'F062' : 'R062' ,
    'F087' : 'Z087' ,
    'F106' : 'Y106' ,
    'F129' : 'J129' ,
    'F158' : 'H158' ,
    'F184' : 'F184' ,
    'F146' : 'W146' ,
    'F213' : 'K213' 
}

# ==================================================
# ==================================================
    
def open_filter_trans_files(line0):

    # split first line of Roman-trans file and read list of filters,
    # then open each file
    column_list = line0.split()

    fp_list = []
    filter_list = []
    for colname in column_list:
    
        if colname == 'Wave':
            fp_list.append(None)
            filter_list.append(None)
            continue
        
        if 'ism' in colname: continue
                

        if WRITE_SNANA_FILTER_TRANS:
            trans_file = f"{PREFIX_OUTPUT}_{colname}.dat"
            print(f"\t Open {trans_file}")
            f = open(trans_file,"wt")
            f.write(f"# wave trans\n")        
            fp_list.append(f)
        else:
            fp_list.append(None)

        filter_list.append(colname)
        
    return fp_list, filter_list

def compute_zp(filt,tr_dict):

    LAM_ARRAY   = np.array(tr_dict['LAM_ARRAY'])  
    TRANS_ARRAY = np.array(tr_dict['TRANS_ARRAY'])
    TRANS_ARRAY = TRANS_ARRAY * AREA_FUDGE
    
    LAM_BIN     = LAM_ARRAY[5] - LAM_ARRAY[4]  # bin size, A

    h        = 6.6260755e-27  # Planck constant, erg/sec
    c        = 2.99792458e18  # speed of light, A/sec
    AB_spec  = 3.631E-20 * c / (LAM_ARRAY*LAM_ARRAY)  # AB spec, erg/sec/A
    AB_nphot = AB_spec / (h*c)  # Nphot per lam bin
    
    AB_sum  = LAM_BIN * np.sum(AB_nphot * LAM_ARRAY * TRANS_ARRAY) 
    mag     = 2.5*np.log10(AB_sum)

    zp = mag
    #print(f"\t ZP_AB({filt}) = {zp:.3f}")
    return zp
    

def process_effarea(effarea_file):

    print(f" Process {effarea_file}")

    with open(effarea_file,"rt") as f:
        contents = f.readlines()

    FIRST_LINE = True
    FILTER_TRANS_DICT = {}
    
    for line in contents:
        if line[0] == '#' : continue
        if FIRST_LINE:
            fp_list, filter_list = open_filter_trans_files(line)
            FIRST_LINE = False
            #print(f" Init FILTER_TRANS_DICT for {filter_list}")
            for filt in filter_list:                
                FILTER_TRANS_DICT[filt] = { 'LAM_ARRAY': [],  'TRANS_ARRAY' : []}
            continue
        
        val_list_orig = line.split()
        lam_micron = float(val_list_orig[0])
        lam_A      = 10000.0 * lam_micron
        icol = -1
        for fp, filt in zip(fp_list,filter_list):
            icol += 1
            if icol==0 : continue
            trans = float(val_list_orig[icol])
            if trans > 0.0 :
                if WRITE_SNANA_FILTER_TRANS:
                    fp.write(f"{lam_A:8.2f}  {trans:.5f}   {filt}\n")
                FILTER_TRANS_DICT[filt]['LAM_ARRAY'].append(lam_A)
                FILTER_TRANS_DICT[filt]['TRANS_ARRAY'].append(trans)   

    # --------------
    # compute ZP

    filt_list = []
    zp_list   = []
    
    for filt, tr_dict in FILTER_TRANS_DICT.items():
        if filt is None: continue
        zp = compute_zp(filt, tr_dict)
        zp = f"{zp:.4f}"
        filt_list.append(filt)
        zp_list.append(float(zp))

    print(f" ZP_AB({filt_list}) = {zp_list}")

    return filt_list, zp_list
    
# ===================================
if __name__ == "__main__":

    effarea_file_list = glob.glob(TRANS_INPUT)

    zp_sca_dict = {}

    for effarea_file in effarea_file_list:
        filt_list, zp_list = process_effarea(effarea_file)
        
        # convert zp_list over filters into zp list over sca
        for filt, zp in zip(filt_list, zp_list):
            if filt in zp_sca_dict:
                zp_sca_dict[filt].append(zp)
            else:
                zp_sca_dict[filt] = []
    
    # print and compute stats for each band
    n_effarea = len(effarea_file_list)
    print(f"\n  # Average ZP among {n_effarea} effective area files:")
    print(f"\n  # BAND   <ZP>   ZP_MIN-ZP_MAX   ZP_RANGE")
    for filt, zp_sca_list in zp_sca_dict.items():
        zp_avg = np.mean(zp_sca_list) 
        zp_min = np.min(zp_sca_list) 
        zp_max = np.max(zp_sca_list) 
        zp_range = zp_max - zp_min
        band  = BAND_MAP[filt]
        print(f"    {band}  {zp_avg:.3f}  {zp_min:.3f}-{zp_max:.3f}     {zp_range:.2f}")

    # == END ===
