#!/usr/bin/env python
#
# Created Jan 29 2021 by R.Kessler  
# Translate H18 notebook to script using config file for inputs.
#
# Re-start Feb 2024 under Roman-PIT
# Goal is to quickly generate multiple survey strategies (i.e., SIMLIB files)
# for Core Community Survey (CCS) group.
#
# Beware that there are two types of config file inputs,
# and the code here internally figures out the type.
# First is the nominal config defining the survey, tiers, analysis, etc ...
# The 2nd type is for "copy+modify" model to make additional config-inputs
# with substitution changes specified by instructions in the copy+modify 
# config file.
#

import os, sys, yaml, re, argparse, math, random, logging, glob
import numpy as np
from scipy.interpolate import interp1d


# - - - - - - 
SNDATA_ROOT = os.environ['SNDATA_ROOT']
CWD         = os.getcwd()

# define BAND columns in config file
ICOL_BAND_SYMBOL   =  0
ICOL_BAND_LAMAVG   =  1  # mean wavelength, Ang
ICOL_BAND_ZPsec    =  2  # AB ZP per sec
ICOL_BAND_NEA      =  3  # NEA computed from PSF (pixels)
ICOL_BAND_ZODIAC   =  4  # Zodiac noise: e/sec/pixel
ICOL_BAND_THERMAL  =  5  # Thermal noise: e/sec/pixel

# define TIER columms in config file
ICOL_TIER_NAME      = 0  # e.g, SHALLOW or DEEP
ICOL_TIER_RA        = 1  # RA(deg) at center of field
ICOL_TIER_DEC       = 2  # DEC(deg) at center of field
ICOL_TIER_BANDS     = 3  # list of bands
ICOL_TIER_REL_AREA  = 4  # relative area
ICOL_TIER_DT_VISIT  = 5  # time between visits, days
ICOL_TIER_zSNRMATCH = 6  # redshift to match SNRMAX
#??ICOL_TIER_AREA      = 7  # area (sq deg) computed from constraints

# config block names
NAME_CONFIG_INSTRUMENT = 'CONFIG_INSTRUMENT'
NAME_CONFIG_SURVEY     = 'CONFIG_SURVEY'
NAME_CONFIG_ANALYSIS   = 'CONFIG_ANALYSIS_PREP'

# dictionary key names
KEYNAME_REL_AREA       = 'rel_area'
KEYNAME_AREA_FRAC      = 'areaFrac'
KEYNAME_DT_VISIT       = 'dt_visit'
KEYNAME_zSNRMATCH      = 'zSNRMATCH'

# miscellaneous constants
SURVEY_NAME  = "ROMAN"    # must be in $SNDATA_ROOT/SURVEY.DEF
GAIN         = 1.0         # 1 AUD = 1e-
FOURPI       = 4.0 * 3.1415926535
TSEC_PER_DAY = 24.0 * 3600.0
NSQDEGREE_SPHERE = 41253.0  # sq degrees in spehere
VALID_BAND_STRING  = "RZYJHFK"     # abort if request band is not in this list
VALID_BAND_LIST    = list(VALID_BAND_STRING)

TEMP_PREFIX  = "TEMP"
TEMP_MJD_SIMLIB = [ 60000.0, 60005.0, 60010.0 ]  # to solve for t_expose

TIME_SUM_TD = 180.0  # total TD time, days

# - - - - - - - - - - 
# define globals for analysis setup
ANALYSIS_INSTRUCTION_FILE = "ANALYSIS_INSTRUCTIONS.README"
program_submit            = "submit_batch_jobs.sh"
SAMPLE_INPUT_DIR=os.path.expandvars("$SNANA_ROMAN_ROOT/starterKits/sample_input_files")
SIMGEN_SUBMIT_TEMPLATE = f"{SAMPLE_INPUT_DIR}/SIMGEN_ROMAN_SUBMIT.INPUT"
SIMFIT_SUBMIT_TEMPLATE = f"{SAMPLE_INPUT_DIR}/SIMFIT_ROMAN_TEMPLATE.nml"

SUBMIT_ALL_FILE = "SUBMIT_ALL.LIST"
SUBMIT_ALL_LOG  = "SUBMIT_ALL.LOG"

COPY_INPUT_SIMGEN_LIST = [ 'SIMGEN_ROMAN_INSTRUMENT+HOST.INPUT', 
                           'SIMGEN_INCLUDE_*.INPUT', 
                           'SIMGEN_TRANSIENT_*.INPUT' ]
WILDCARD_INPUT_SIMGEN_TRANSIENT = 'SIMGEN_TRANSIENT_*.INPUT'

PREFIX_SUBMIT_SIM   = "SUBMIT1_SIM"
PREFIX_SUBMIT_LCFIT = "SUBMIT2_LCFIT"
PREFIX_SUBMIT_BBC   = "SUBMIT3_BBC"
PREFIX_SUBMIT_LIST = [ PREFIX_SUBMIT_SIM, PREFIX_SUBMIT_LCFIT, PREFIX_SUBMIT_BBC ]

SIMGEN_PREFIX = "SIMGEN_TRANSIENT"

ROWKEY_SIMLIB_OBS      =    'S:'
ROWKEY_SIMLIB_REJECT   =    'XXX:'  # reject from random dither

# ==========================================
def get_args():
    parser = argparse.ArgumentParser()

    msg = "name of input file"
    parser.add_argument("input_file", help=msg, nargs="?", default=None)


    msg = "skip to analysis prep"
    parser.add_argument("-s", "--skip_to_analysis", help=msg, action="store_true")
 
    # - - - - -
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return parser.parse_args()

    # end get_args

def get_inputs(input_file):
    line_list = []
    with open(input_file, "r") as f:
        for line in f:
            line_list.append(line)
    config = yaml.safe_load("\n".join(line_list))

    # if no CONFIG_SURVEY key, then bail now ... assume instead
    # that it's a list of instrictions to create more input config files.
    if NAME_CONFIG_SURVEY not in config:
        return config

    # convert TIERS rows into useful lists
    config_survey = config[NAME_CONFIG_SURVEY]

    config_survey['TIER_INPUT_LIST'] = []
    for tier_input_string in config_survey['TIERS']:
        tier_input_list = parse_tier_inputs(tier_input_string)
        config_survey['TIER_INPUT_LIST'].append(tier_input_list)

    # store MJD_RANGE_TOTAO as survey input
    n_season = len(config_survey['MJD_SEASON'])
    MJD_MIN = config_survey['MJD_SEASON'][0].split()[0]
    MJD_MAX = config_survey['MJD_SEASON'][n_season-1].split()[1]
    config_survey['MJD_RANGE_TOTAL']  = [ int(MJD_MIN), int(MJD_MAX) ]

    # read CONFIG_INSTRUMENT file and attach contents to config
    # as if the file contents were in the input file
    config_instr_file = os.path.expandvars(config['CONFIG_INSTRUMENT_FILE'])
    line_list = []
    with open(config_instr_file, "r") as f:
        for line in f:
            if line[0] != '#':
                line_list.append(line)
    config_instr = yaml.safe_load("\n".join(line_list))
    config[NAME_CONFIG_INSTRUMENT] = config_instr
 
    return config
    # end get_inputs
# ============================
def setup_logging():

    #logging.basicConfig(level=logging.DEBUG,

    #fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    fmt = "%(message)s"
    logging.basicConfig(level=logging.INFO,  format = fmt)
    return
    # end setup_logging

def mkdir_output(input_config_survey):
    outdir = input_config_survey['OUTDIR']

    if os.path.exists(outdir):
        return

    logging.info(f" Create outdir : {outdir}")
    os.mkdir(outdir)
    return

def init_band_dict(config_instr):

    band_dict          = {}

    for band in VALID_BAND_LIST:
        band_dict[band] = {
            'symbol'    : band,
            'lamavg'    : config_instr['FILTER_WAVELENGTH'][band],
            'zpsec'     : config_instr['ZEROPOINT'][band],
            'nea'       : config_instr['NEA'][band],
            'zodiac'    : config_instr['ZODIAC'][band],
            'thermal'   : config_instr['THERMAL'][band],
        }

    band_dict['read_noise_params' ] = config_instr['READ_NOISE']
    # - - - -

    return band_dict
    # end init_band_dict

def parse_tier_inputs(tier_input_string):

    # if input tier_input_string = 
    #   "SHALLOW   10  -10  RZYJ   [3,2,1]    [ 5, 8]   [0.8, 0.9]"
    # then return tier_input_list
    #   [ 'SHALLOW', 10, -10, 'RZYJ',  [3,2,1] , [ 5, 8] , [0.8, 0.9] ]
    #
    # This specialized parsing avoid forcing the user to enter
    # extraneous brackets and commas in the input config file.

    full_list          = tier_input_string.split()
    temp_list_of_lists = re.findall("\[(.*?)\]", tier_input_string )  
    only_list_of_lists = [] 
    for tmp_list in temp_list_of_lists:
        new_list = [float(x) for x in tmp_list.split(',') ]
        only_list_of_lists.append(new_list)

    tier_input_list = []
    tier_input_list.append( full_list[ICOL_TIER_NAME] )
    tier_input_list.append( full_list[ICOL_TIER_RA] )
    tier_input_list.append( full_list[ICOL_TIER_DEC] )
    tier_input_list.append( full_list[ICOL_TIER_BANDS] )

    tier_input_list.append(only_list_of_lists[0])
    tier_input_list.append(only_list_of_lists[1])
    tier_input_list.append(only_list_of_lists[2])

    #sys.exit(f"\n xxx tier_input_list = {tier_input_list} ")
    return tier_input_list

    # end parse_tier_items

def normalize_rel_area_list(input_config):

    # normalize relative areas so that rel-area sum over tiers = 1 for
    # each re-area on list.

    tier_input_list = input_config['TIER_INPUT_LIST']

    # quickly comput sum of rel_area to normalize them to sum to 1
    rel_area_tot    = [0.0] * 10
    for tier_input in tier_input_list:  
        rel_area_list   = tier_input[ICOL_TIER_REL_AREA]
        i = 0
        for rel_area in rel_area_list:
            rel_area_tot[i]  += float(rel_area)
            i += 1

    n_area       = len(rel_area_list)
    rel_area_tot = rel_area_tot[0:n_area]


    rel_area_dict = {}
    for tier_input in tier_input_list:
        tier_name       = tier_input[ICOL_TIER_NAME]
        tmp_list        = tier_input[ICOL_TIER_REL_AREA]
        rel_area_dict[tier_name] = []
        for rel_area, rel_tot in zip(tmp_list,rel_area_tot):
            rel_area /= rel_tot
            rel_area_dict[tier_name].append(0.001*int(rel_area*1000.))

    return rel_area_dict
    # end normalize_rel_area_list

def init_tier_dict(input_config_survey):
    
    # parse input config (already read, contents passed as input arg) 
    # and return dictionary of tier info

    tier_input_list    = input_config_survey['TIER_INPUT_LIST']
    tier_dict          = {}
    tier_name_list     = []

    logging.info(" INIT TIER DICTIONARY: ")

    # get sum of MJD ranges over seasons
    tsum_season = 0.0
    MJD_SEASON = []
    n_season   = 0
    for mjd_range in input_config_survey['MJD_SEASON']:
        mjd_min = float(mjd_range.split()[0])
        mjd_max = float(mjd_range.split()[1])
        tsum_season += (mjd_max - mjd_min)
        MJD_SEASON.append( [mjd_min, mjd_max] )
        n_season  += 1

    logging.info(f"\t Sum of season durations: {tsum_season} days.")

    random_reject = input_config_survey.setdefault('RANDOM_REJECT_OBS', 0.0)
    logging.info(f"\t Randonly reject {random_reject} of observations")

    # - - - - 

    rel_area_dict = normalize_rel_area_list(input_config_survey)

    all_bands = ''

    logging.info("\n\t  TIER    BANDS    DT_VISIT    Nvisit       AreaFrac        zSNRMATCH")
    logging.info("\t -----------------------------------------------------------------------")

    for tier_input in tier_input_list :
        name            = tier_input[ICOL_TIER_NAME]
        ra              = tier_input[ICOL_TIER_RA] 
        dec             = tier_input[ICOL_TIER_DEC] 
        bands           = tier_input[ICOL_TIER_BANDS]
        rel_area_list   = rel_area_dict[name]
        dt_visit_list   = tier_input[ICOL_TIER_DT_VISIT]
        z_list          = tier_input[ICOL_TIER_zSNRMATCH]
        all_bands      += bands

        # store info directly from config file
        tier_dict[name] = {
            'name'             : name,
            'ra'               : ra,
            'dec'              : dec,
            'bands'            : bands,
            'random_reject'    : random_reject,
            'rel_area_list'    : rel_area_list,
            'dt_visit_list'    : dt_visit_list,
            'z_list'           : z_list,
            'area_solve_list'  : -9.0,   # to be computed later
            'dumlast'     : 0
        }

        # update dict with computed info
        nband      = len(bands)
        all_bands_unique = ''.join(set(all_bands))

        nvisit_list = []
        for dt in dt_visit_list:
            nvisit     = int(tsum_season / dt )  + 1
            nvisit_list.append(nvisit)

        tier_dict[name].update({
            'nband'             : nband,
            'nvisit_list'       : nvisit_list,
            'ntile_solve'       : -9,   # to be computed later
            'dumlast'           : 0
        })
        tier_name_list.append(name)

        info_line = f"{name:8}  {bands}   " \
                    f"{dt_visit_list} days    {nvisit_list}     {rel_area_list}   {z_list}"
        logging.info(f"\t {info_line}")

    tier_dict['LIST']             = tier_name_list
    tier_dict['ALL_FILTERS']      = all_bands_unique
    tier_dict['MJD_SEASON']       = MJD_SEASON
    tier_dict['TIME_SUM_OBS']     = input_config_survey['TIME_SUM_OBS']
    tier_dict['TIME_SUM_SEASON']  = tsum_season
    tier_dict['N_SEASON']         = n_season

    logging.info("")
    return tier_dict

    # end init_tier_dict
    
def get_index3d_dict(input_config_survey):

    tier_input_list = input_config_survey['TIER_INPUT_LIST']
    
    # pick if lists from first tier since all tiers must have same shape
    rel_area_list = tier_input_list[0][ICOL_TIER_REL_AREA]
    dt_visit_list = tier_input_list[0][ICOL_TIER_DT_VISIT]
    z_list        = tier_input_list[0][ICOL_TIER_zSNRMATCH]
    
    index3d_dict = {
        KEYNAME_REL_AREA  : [],
        KEYNAME_DT_VISIT  : [],
        KEYNAME_zSNRMATCH : [],
        'dumLast'   : 0
    }

    n3d = 0
    na = len(rel_area_list)
    nt = len(dt_visit_list)
    nz = len(z_list)
    for j0 in range(0,na):
        for j1 in range(0,nt):
            for j2 in range(0,nz):
                index3d_dict[KEYNAME_REL_AREA].append(j0)
                index3d_dict[KEYNAME_DT_VISIT].append(j1)
                index3d_dict[KEYNAME_zSNRMATCH].append(j2)
                n3d += 1
    
    logging.info(f" 3D index list size = {n3d} for rel_area, dt_visit, zSNRMATCH ")
    return na, nt, nz, index3d_dict

    # end get_index3d_dict

def solve_exposure_time(tier_name, tier_dict, band_dict, input_config, iz):

    # Create temporary SIMLIB with only a few epochs, and
    # each LIBID corresponds to user FORCE_EXPOSURE from config file.
    # Then scoop up SIMGEN_DUMP table and interpolate SNR vs. Exposure time.
    # Ouput is t_expose added to tier_dict.
    
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_survey = input_config[NAME_CONFIG_SURVEY]

    z         = tier_dict[tier_name]['z_list'][iz]
    bands     = tier_dict[tier_name]['bands'] # full band list for this tier
    band_list = list(bands)

    pixsize = float(input_config_instr['PARAMS']['PIXSIZE'])
    t_slew  = float(input_config_instr['PARAMS']['TIME_SLEW'])

    # - - - - - 
    # check if zSNRMATCH input is redshift (<3) or explicit Texpose(>3)
    if z > 3.0 :  # <== it's really the exposure time, so nothing to solve
        logging.info(f"\t last {tier_name} input column is T_expose (not zSNRMAX) " \
                     f"-> nothing to solve.")
        t_expose = z
        t_expose_list = [ t_expose ] * len(band_list)
        tier_dict[tier_name].update( { 't_expose_list' : t_expose_list } )
        tier_dict[tier_name].update( { 't_slew'        : t_slew } )
        return

    # - - -  continue with solving for Texpose to get SNRMAX at zSNRMAX
    outdir       = input_config_survey['OUTDIR']
    tier_name_noslash = tier_name.split('/')[0]
    SIMLIB_FILE  = f"{outdir}/{TEMP_PREFIX}_{tier_name_noslash}_{outdir}.SIMLIB"

    TEXPOSE_LIST = input_config_survey['FORCE_TEXPOSE_LIST'].split()
    TEXPOSE_LIST = [int(i) for i in TEXPOSE_LIST]  # string -> integer 
    n_expose     = len(TEXPOSE_LIST)
    NGENTOT_LC   = n_expose * input_config_survey['FORCE_NGEN']

    # create temp simlib file where each LIBID = exposure time
    logging.info(f" Open {SIMLIB_FILE} to measure SNRMAX vs. TEXPOSE z={z}")
    with open(SIMLIB_FILE,"wt") as f:

        f.write(f"DOCUMENTATION: \n");
        f.write(f"  PURPOSE: temp simlib to map SNRMAX vs. Texpose \n");
        f.write(f"DOCUMENTATION_END: \n\n");

        f.write(f"SURVEY:   {SURVEY_NAME} \n")
        f.write(f"FILTERS:  {VALID_BAND_STRING} \n")
        f.write(f"PIXSIZE:  {pixsize} \n")
        f.write(f"PSF_UNIT: NEA_PIXEL  # Noise-equiv-Area instead of PSF \n")
        f.write(f"BEGIN LIBGEN")
        
        for texpose in TEXPOSE_LIST :
            write_temp_simlib_header(f, texpose, tier_name, band_list )
            for band in band_list :
                simlib_dict = \
                    compute_simlib(band, texpose, tier_name, band_dict )
                write_temp_simlib_mjds(f, simlib_dict)
            f.write(f"END_LIBID: {texpose}\n")

    # run simulation
    t0 = TEMP_MJD_SIMLIB[1]
    version         = f"{TEMP_PREFIX}_{tier_name_noslash}_{outdir}"

    sim_input_file = input_config_survey['FORCE_SIMGEN_INPUT_FILE']
    log_file       = f"{version}_SIM.LOG"
    sim_args       = f"GENVERSION {version} "
    sim_args      += f"GENRANGE_REDSHIFT  {z} {z} "
    sim_args      += f"GENRANGE_PEAKMJD   {t0} {t0} "
    sim_args      += f"SIMLIB_FILE {SIMLIB_FILE} "
    sim_args      += f"NGENTOT_LC {NGENTOT_LC} "
    sim_args      += f"HOSTLIB_NREPEAT_GALID_SNPOS {n_expose} " 

    cmd = f"snlc_sim.exe {sim_input_file} {sim_args} > {outdir}/{log_file}"


    logging.info(f" Run sim to create SIMGEN_DUMP ... (please be patient)")
    os.system(cmd)
    
    # analyze simgen dump file to get t_expose    
    simgen_dump_file = f"$SNDATA_ROOT/SIM/{version}/{version}.DUMP"
    logging.info(f" Analyze {simgen_dump_file}") 
    simgen_dump_file = os.path.expandvars(simgen_dump_file)

    t_expose_calc_list = interp_texpose(simgen_dump_file, input_config, 
                                   band_list, band_dict, z)
    

    TEXPOSE_MIN = int(input_config_survey['TEXPOSE_MIN'])
    t_expose_list = []
    for t_expose in t_expose_calc_list:
        if t_expose < TEXPOSE_MIN: t_expose = TEXPOSE_MIN
        t_expose_list.append(t_expose)

    tier_dict[tier_name].update( { 't_expose_list' : t_expose_list } )
    tier_dict[tier_name].update( { 't_slew'        : t_slew } )

    #print(f" xxx tier_dict =  {tier_dict} \n")
    logging.info(f"")

    # end solve_exposure_time


def interp_texpose(simgen_dump_file, input_config, band_list, band_dict, zSNRMAX):

    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_survey = input_config[NAME_CONFIG_SURVEY]

    # Read simgen_dump_file (from temp sim) and interpolate
    # t_expose vs. snrmax to find t_expose at input force_snrmax.
    
    TEXPOSE_MIN = int(input_config_survey['TEXPOSE_MIN'])
    texpose = []

    TEXPOSE_LIST = input_config_survey['FORCE_TEXPOSE_LIST'].split()
    TEXPOSE_LIST = [int(i) for i in TEXPOSE_LIST]  # string -> integer
    n_expose     = len(TEXPOSE_LIST)

    FORCE_SNRMAX_GRID = input_config_survey['FORCE_SNRMAX']
    FORCE_NGEN        = input_config_survey['FORCE_NGEN']

    dump = np.genfromtxt(simgen_dump_file, skip_header=6, 
                         names=True ,dtype=None, encoding=None)

    for band in band_list:
        KEY_TEXPOSE  = 'LIBID'  # this is the trick to track Texpose
        KEY_SNR_BAND = f'SNRMAX_{band}'
        timelist_to_avg = []

        lamrest      = band_dict[band]['lamavg'] / ( 1.0 + zSNRMAX)
        FORCE_SNRMAX = get_SNRMAX(FORCE_SNRMAX_GRID, band, lamrest)

        for i in np.arange(FORCE_NGEN):
            j0 = n_expose*i
            #j1 = n_expose*i + (n_expose-1)  # orig bug
            j1 = n_expose*i + n_expose  # bug fix Jun 3 2024

            # note that LIBID is the exposure time, so this interpolates
            # SNR vs. Texpose
            f = interp1d(dump[KEY_TEXPOSE][j0:j1],
                         dump[KEY_SNR_BAND][j0:j1],
                         kind = 'linear', fill_value = "extrapolate")

            #print(dump['LIBID'][n_expose*i], dump['LIBID'][n_expose*i+(n_expose-1)])
            time_list = np.linspace(dump[KEY_TEXPOSE][j0], 
                                    dump[KEY_TEXPOSE][j1-1], 
                                    num=500) # array of times to interpolate over

            snr_list = f(time_list)   #create new interpolated y-array

            # find index where snr_list is closest to force_snrmax
            idx = (np.abs(snr_list - FORCE_SNRMAX)).argmin()  

            closest_t   = time_list[idx]
            closest_snr = snr_list[idx]

            if time_list[idx] < TEXPOSE_MIN : 
                time_list[idx] = TEXPOSE_MIN
         
            timelist_to_avg.append(closest_t)

            if band == 'JJJ' :
                print(f" xxx -------------------------------------------- ")
                print(f" xxx {band}-{i}  idx={idx:3d}/{len(time_list)}  closest_[t,snr] = " \
                      f"{closest_t:.2f} {closest_snr:.2f} ")
                print(f" xxx j0={j0} j1={j1} dump snr = {dump[KEY_SNR_BAND][j0:j1]}\n")
                if i==0 : print(f" xxx time_list = \n{time_list} \n")
                print(f" xxx snr_list = \n{snr_list}")
                sys.stdout.flush() 
                #sys.exit("\n xxx bye. \n")
            
            #print(timelist_to_avg)
                    
        t= np.average(timelist_to_avg)

        texpose.append(t)  # Append the x value corresponding to the maximum y value
        logging.info(f"\t For {band}-band Texpose = {t:6.1f} sec for  " \
                     f"<SNRMAX> = {closest_snr:6.2f}    " \
                     f"<lamRest>={lamrest:.0f}")        
        
    #print(texpose)
    return texpose
    #end interp_texpose

def get_SNRMAX(FORCE_SNRMAX_GRID, band, lamrest ):

    # return SNRMAX corresponding to lam_band/(1+z) in rest-frame
    # FORCE_SNRMAX_GRID is of the form
    #   8  [ 1000, 4000 ]   # snrmax=8 for  1000 < lamrest < 4000
    #  10  [ 4000, 12000 ]  # snrmax=10 for 4000 < lamreset < 12000 A
    #   9  [12000, 15000 ]

    SNRMAX = -9.0

    for row in FORCE_SNRMAX_GRID:
        snrmax_tmp = float(row.split()[0])

        lamrange = re.findall("\[(.*?)\]", row )[0].split(',')
        lammin   = float(lamrange[0])
        lammax   = float(lamrange[1])
        if lamrest >= lammin and lamrest <= lammax :
            SNRMAX = snrmax_tmp
        
    if SNRMAX < 0.0 :
        msgerr = f"\nERROR: Cannot find SNRMAX coverage for lamrest = {lamrest:.1f}\n"\
                 f"Check FORCE_SNRMAX = {FORCE_SNRMAX_GRID}"
        sys.exit(f"{msgerr}")

    return SNRMAX
    # end get_SNRMAX

def write_temp_simlib_mjds(f,simlib_dict):

    band     = simlib_dict['band']
    idexpt   = 888
    zp       = simlib_dict['zp']
    zperr    = 0.0
    nea      = simlib_dict['nea']
    psf_sig  = simlib_dict['psf_sig']
    psf_sig2 = 0.0
    psf_ratio = 0.0
    sky_sig  = simlib_dict['sky_noise']
    read_sig = simlib_dict['read_noise']
    mag      = 99.0

    for mjd in TEMP_MJD_SIMLIB :
        line = f"S: {mjd:.1f} {idexpt} {band}   {GAIN:.1f}  {read_sig:5.2f} " \
               f"{sky_sig:6.2f}  {nea:6.3f}  " \
               f" {zp:6.3f} {zperr}  {mag:.1f}"
        f.write(f"{line}\n")

    # end write_libid_mjds

def write_temp_simlib_header(f, texpose, tier_name, band_list ):

    n_mjd  = len(TEMP_MJD_SIMLIB)
    n_band = len(band_list)
    nobs   = n_mjd * n_band

    # if tier_namne = 'MEDIUM/1', remove the '/1' and print only MEDIUM after FIELD key
    tier_name_noslash = tier_name.split('/')[0]

    f.write(f"\n# -------------------------------------------\n")
    f.write(f"LIBID:  {texpose}    # {texpose} sec exposures\n")
    f.write(f"FIELD:  {tier_name_noslash}      RA: 0.0    DEC: 0.0  MWEBV: 0.0 \n")
    f.write(f"NOBS: {nobs} \n")
    # end write_libid_header

def solve_area(tier_dict, band_dict, input_config, index1d_list):

    # solve for sky area in each tier
    #

    input_config_inst   = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_survey = input_config[NAME_CONFIG_SURVEY]

    ia = index1d_list[0]  # points to area frac
    it = index1d_list[1]  # points to dt_visit
    iz = index1d_list[2]  # points to zSNRMATCH

    logging.info(f"\n\t ---- SOLVE FOR AREA (ia={ia}, it={it}, iz={iz} ---- ")

    TIME_SUM_OBS = float(input_config_survey['TIME_SUM_OBS'])     # days
    t_slew       = float(input_config_instr['PARAMS']['TIME_SLEW'])  # seconds
    fov          = float(input_config_instr['PARAMS']['FOV'])  # field of view, sq deg
    tsum_per_visit = 0  # includes both tiers
    TSEC_PER_HOUR = 3600.0
    AT_sum = 0.0  # summ of area_frac * t_visit_sum 
    
    for tier_name in tier_dict['LIST'] :  
        rel_area      = tier_dict[tier_name]['rel_area_list'][ia]
        bands         = tier_dict[tier_name]['bands']
        nvisit        = tier_dict[tier_name]['nvisit_list'][it]

        band_list     = list(bands)
        nband         = len(band_list)
        t_expose_list = tier_dict[tier_name]['t_expose_list']  # seconds, all bands per visit
        t_expose_sum  = sum(t_expose_list)                     # seconds, all bands per visit
        t_slew_sum    = float(nband) * t_slew                  # seconds, all bands per visit
        t_visit_sum   = t_expose_sum + t_slew_sum              # seconds, all bands per visit
        AT_sum       += (rel_area * t_visit_sum * nvisit) 
        open_shutter_frac = t_expose_sum / t_visit_sum

        tier_dict[tier_name]['open_shutter_frac'] = open_shutter_frac
        tier_dict[tier_name]['t_visit_sum']       = t_visit_sum

        #print(f" xxx {tier_name:10s}: ia={ia} rel_area={rel_area}  nvis={nvisit}")

    # - - - - - -
    #print(f"\n\t xxx AT_sum = {AT_sum}\n")

    AREA_TOT = (TIME_SUM_OBS * TSEC_PER_DAY * fov) / AT_sum  # total sky area summed over tiers
    logging.info(f"\t Total AREA summed over tiers: {AREA_TOT:.1f} deg^2 \n")
    tier_dict['AREA_TOT'] = AREA_TOT
    # - - - - -

    # loop again and print table
    logging.info(f"\t               area            time per    Open-shutter ")
    logging.info(f"\t   Tier       (deg^2)   ntile  visit (hr)   fraction    ")
    logging.info(f"\t ----------------------------------------------------------- ")

    for tier_name in tier_dict['LIST'] :  
        
        rel_area      = tier_dict[tier_name]['rel_area_list'][ia]
        area          = AREA_TOT * rel_area
        tier_dict[tier_name]['area'] = area

        ntile             = int(area/fov)
        tier_dict[tier_name]['ntile'] = ntile

        t_visit_sum       = tier_dict[tier_name]['t_visit_sum'] * ntile / 3600.0 # hr
        open_shutter_frac = tier_dict[tier_name]['open_shutter_frac']
        z                 = tier_dict[tier_name]['z_list'][iz]

        logging.info(f"\t {tier_name:12s}  {area:5.1f}     {ntile}       {t_visit_sum:4.1f}" \
                     f"        {open_shutter_frac:5.2f}")

    return 
    # end solve_nvisit

def compute_simlib(band, t_expose, tier_name, band_dict ):
    
    verbose = False

    if verbose :
        logging.info(f" COMPUTE SIMLIB INFO for TIER = {tier_name} {band} " \
                     f"(t_expose={t_expose} sec) : ")

    read_noise_params = band_dict['read_noise_params']

    if band not in band_dict :
        sys.exit(f"\n ERROR: Invalid band={band} for TIER={tier_name}" \
                 f"\n\t Check BAND: key for valid bands.")

    zpsec    = band_dict[band]['zpsec']
    nea      = band_dict[band]['nea']
    zodiac   = band_dict[band]['zodiac']
    thermal  = band_dict[band]['thermal']

    # convert NEA to Gaussian PSF, pixels
    psf_sig = math.sqrt(nea / FOURPI)

    # convert ZP/sec to ZP for exposure
    zp      = zpsec + 2.5 * math.log10(t_expose)

    # compute sky noise per pixel
    noise_dict = {
        't_expose'  : t_expose,
        'zodiac'    : zodiac,
        'thermal'   : thermal,
        'readout'   : read_noise_params
        }
    sky_noise, read_noise = get_noise(noise_dict)

    simlib_dict = {
        'band'       : band,
        'zp'         : zp,
        'psf_sig'    : psf_sig,
        'sky_noise'  : sky_noise,
        'read_noise' : read_noise,
        'nea'        : nea
    }

    if verbose :
        logging.info(f"    Band={band} : " \
                     f"ZP={zp:.2f}  sig(PSF)={psf_sig:5.2f} pix  " \
                     f"noise(rd,sky)={read_noise:.1f},{sky_noise:.1f}e-/pix")
        
    return simlib_dict

    # end compute_simlib

def get_noise(noise_dict):

    t_expose = noise_dict['t_expose']
    zodiac   = noise_dict['zodiac']
    thermal  = noise_dict['thermal']
    readout  = noise_dict['readout']

    t_read = readout['T']
    cov0   = readout['COV0']
    cov1   = readout['COV1']

    # - - - -
    skyvar_list = []

    skyvar_zodiac  = (t_expose * zodiac)
    skyvar_list.append(skyvar_zodiac)

    skyvar_thermal = (t_expose * thermal) 
    skyvar_list.append(skyvar_thermal)

    skyvar_sum = 0.0
    for skyvar in skyvar_list:
        skyvar_sum += skyvar
        
    sky_noise = math.sqrt(skyvar_sum)

    # - - - - 
    # read noise from Eq 7 in H18:
    t_ratio       = t_expose/t_read
    tmp1          = (t_ratio - 1.0)/t_ratio
    tmp2          = 1.0 / (t_ratio + 1.0)
    cov_readout   = cov0 + (cov1*tmp1*tmp2)
    read_noise    = math.sqrt(cov_readout)
    #  - - -- 

    return sky_noise, read_noise


def write_simlib(args, input_config, tier_dict, band_dict, index1d_list ):

    input_config_inst   = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_survey = input_config[NAME_CONFIG_SURVEY]

    ia = index1d_list[0]  # points to area frac
    it = index1d_list[1]  # points to dt_visit
    iz = index1d_list[2]  # points to zSNRMATCH

    outdir      = input_config_survey['OUTDIR']
    output_file = f"{SURVEY_NAME}-a{ia:0>2}-t{it:0>2}-z{iz:0>2}.SIMLIB"
    OUTPUT_FILE = f"{outdir}/{output_file}"

    logging.info(f"  WRITE ROMAN SIMLIB FILE TO: {OUTPUT_FILE}")

    FILTERS    = VALID_BAND_STRING
# xxxband_dict['band_list_all']  # all bands, regardless if they are used
    AREA_TOT   = tier_dict['AREA_TOT']
    NLIBID_TOT = input_config_survey['NLIBID_TOT']
    PIXSIZE    = input_config_instr['PARAMS']['PIXSIZE']
    SOLID_ANGLE = FOURPI * (AREA_TOT/NSQDEGREE_SPHERE)

    # determine number of LIBIDs per tier
    nlibid_tot = 0 
    libid_tier_list = [] # specify tier for each LIBID

    for tier_name in tier_dict['LIST']:
        area   = tier_dict[tier_name]['area']
        nlibid = int(NLIBID_TOT * (area/AREA_TOT) + 0.5)
        nlibid_tot += nlibid
        tier_dict[tier_name].update({ 'nlibid' : nlibid})
        libid_tier_list.extend([ tier_name ] * nlibid )

    # randomly shuffle tier vs. LIBID to avoid statistical artifacts in sim
    random.shuffle(libid_tier_list)
    #print(f"\n xxx libid_tier_list = \n{libid_tier_list}\n")

    with open(OUTPUT_FILE,"wt") as f :
        write_docana_keys(f, args, input_config, tier_dict, index1d_list)

        # write SIMLIB header keys
        f.write(f"SURVEY:   {SURVEY_NAME}\n") 
        f.write(f"FILTERS:  {VALID_BAND_STRING} \n")
        f.write(f"PIXSIZE:  {PIXSIZE}  # arcsec \n")
        f.write(f"PSF_UNIT: NEA_PIXEL  # Noise-equiv-Area instead of PSF \n")
        f.write(f"SOLID_ANGLE:  {SOLID_ANGLE:.4f}   # (sr) sum of all tiers\n")
        f.write(f"BEGIN LIBGEN \n\n")

        # loop over each tier, which is called FIELD for SNANA
        # time_sum is used as diagnostic to check total survey time.

        LIBID = 0
        for tier_name in libid_tier_list :
            LIBID += 1
            write_simlib_libid(f, LIBID, tier_name, 
                               tier_dict, band_dict, index1d_list)
        # - - - 
        f.write("\nEND_OF_SIMLIB:  \n")

        
    return output_file
    # end write_simlib

def write_simlib_libid(f, LIBID, tier_name, tier_dict, band_dict, index1d_list ):

    # write single simlib entry for this tier
    local_dict = tier_dict[tier_name]

    ia = index1d_list[0]  # points to area frac
    it = index1d_list[1]  # points to dt_visit
    iz = index1d_list[2]  # points to zSNRMATCH

    #sys.exit(f"\n xxx local_dict = {local_dict} \n")

    t_expose_list   = local_dict['t_expose_list']
    t_slew     = local_dict['t_slew'  ]
    ntile      = local_dict['ntile'] 
    nvisit     = local_dict['nvisit_list'][it]
    ra         = local_dict['ra']
    dec        = local_dict['dec']
    bands      = local_dict['bands']      # e.g., RZYJ
    band_list  = list(bands)          # e.g., 'R', 'Z', 'Y', 'J'
    dt_visit   = local_dict['dt_visit_list'][it]    # time between visits (days)
    MJD_SEASON = tier_dict['MJD_SEASON']

    TIME_SUM_OBS    = tier_dict['TIME_SUM_OBS']
    n_season        = tier_dict['N_SEASON'] 
    time_per_season = TIME_SUM_OBS / float(n_season)
    NOBS            = 0
    line_epoch_list = []

    random_reject = local_dict['random_reject']

    for mjd_range in MJD_SEASON:
        mjd_min   = mjd_range[0]
        mjd_max   = mjd_range[1]
        nvisit_season = int((mjd_max - mjd_min)/dt_visit)

        for v in range(0,nvisit_season):
            mjd = mjd_min + (v * dt_visit)
            for band, t_expose in zip(band_list,t_expose_list):

                if random.uniform(0.0,1.0) >= random_reject:  
                    NOBS   += 1   # this is also IDEXPT
                    rowkey = ROWKEY_SIMLIB_OBS
                else:
                    rowkey = ROWKEY_SIMLIB_REJECT

                IDEXPT = NOBS
                simlib_dict = \
                    compute_simlib(band, t_expose, tier_name, band_dict )

                read_noise = simlib_dict['read_noise']
                sky_noise  = simlib_dict['sky_noise']
                psf_sig    = simlib_dict['psf_sig']
                zp         = simlib_dict['zp']
                nea        = simlib_dict['nea']
                zperr      = 0.001 # ??

                line = f"{rowkey} {mjd:10.4f} {IDEXPT:4}  {band}  {GAIN:3.1f} " \
                       f"{read_noise:6.2f} {sky_noise:6.2f} " \
                       f"{nea:8.3f}  " \
                       f"{zp:6.3f} {zperr:6.3f}  99" 
                if rowkey == ROWKEY_SIMLIB_REJECT :
                    line += ' # random reject'

                line_epoch_list.append(line)
                mjd      += (t_expose+t_slew)/TSEC_PER_DAY

    # - - - - - - - 
    # start with header info
    tier_name_noslash = tier_name.split('/')[0]

    f.write(f"# =========================================== \n")
    f.write(f"LIBID: {LIBID}\n")
    f.write(f"FIELD: {tier_name_noslash}    RA: {ra}   DEC: {dec}    \n")
#    f.write(f"TIME_TOT: {time_sum_total:.1f} days\n")
    f.write(f"NOBS: {NOBS}\n")

    f.write("#                           READ          \n")
    f.write("#     MJD  IDEXPT BAND GAIN NOISE  SKYSIG    NEA   ZPTAVG ZPTERR  MAG \n")


    for line in line_epoch_list:  
        f.write(f"{line}\n")

    f.write(f"END_LIBID: {LIBID}\n")
    f.write(f"\n")

    return 

    # end write_simlib_libid

def write_docana_keys(f, args, input_config, tier_dict, index1d_list ):
    
    input_config_survey = input_config[NAME_CONFIG_SURVEY]
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]

    input_file          = args.input_file
    TIME_SLEW           = input_config_instr['PARAMS']['TIME_SLEW']
    TIME_SUM_OBS        = input_config_survey['TIME_SUM_OBS']
    TIME_SUM_SEASON     = tier_dict['TIME_SUM_SEASON']
    AREA_TOT            = tier_dict['AREA_TOT']
    FORCE_SNRMAX        = input_config_survey['FORCE_SNRMAX']
    random_reject       = input_config_survey.setdefault('RANDOM_REJECT_OBS', 0.0)

    ia = index1d_list[0]  # points to area frac
    it = index1d_list[1]  # points to dt_visit
    iz = index1d_list[2]  # points to zSNRMATCH
    
    pad = "    "
    f.write(f"DOCUMENTATION:\n")
    f.write(f"{pad}PURPOSE:      Cadence for {SURVEY_NAME} simulation of transients\n")
    f.write(f"{pad}INTENT:       strategy studies \n")
    f.write(f"{pad}USAGE_KEY:    SIMLIB_FILE \n")
    f.write(f"{pad}USAGE_CODE:   snlc_sim.exe \n")

    f.write(f"{pad}INPUT_CONFIG: {input_file} \n")

    f.write(f"{pad}TIME_SUM_OBS:     {TIME_SUM_OBS:.1f}  " \
            f"# total observing time, days (includes slew)\n")

    f.write(f"{pad}TIME_SUM_SEASON:  {TIME_SUM_SEASON:.1f}  " \
            f"# total calendar time, days\n")

    f.write(f"{pad}RANDOM_REJECT_OBS: {random_reject}   " \
            f"# random reject fraction for obs\n") 
    f.write(f"{pad}AREA_TOT:          {AREA_TOT:.2f}   # area sum over tiers, sq deg \n")

    f.write(f"{pad}FORCE_SNRMAX: \n")
    for row in FORCE_SNRMAX:
        f.write(f"    - {row}  # SNRMAX for [lamRest range] \n")

    f.write(f"{pad}TIME_SLEW:    {TIME_SLEW}   # slew time, seconds \n")
    
    f.write("\n")

    f.write(f"{pad}TIER_INFO: " \
            f"#bands ntile nvisit Area  dt_visit NLIBID  zSNRMATCH  OpenFrac\n")

    # - - - - 
    for tier_name in tier_dict['LIST'] :
        bands    = tier_dict[tier_name]['bands']
        ntile    = tier_dict[tier_name]['ntile']
        nvisit   = tier_dict[tier_name]['nvisit_list'][it]
        dt_visit = tier_dict[tier_name]['dt_visit_list'][it]
        area     = tier_dict[tier_name]['area']
        nlibid   = tier_dict[tier_name]['nlibid']
        z        = tier_dict[tier_name]['z_list'][iz]
        open_shutter_frac = tier_dict[tier_name]['open_shutter_frac']

        if ntile == 0: continue
        f.write(f"{pad}- {tier_name:10} {bands:6} " \
                f"{ntile:3}  {nvisit:3}   "\
                f"{area:5.2f}  {dt_visit:4.1f}     {nlibid:4}     {z:4.2f}    " \
                f" {open_shutter_frac:4.2f}   # info\n" )

    # - - - - -
    f.write(f"{pad}TIER_EXPOSURE_TIMES:  # seconds/band\n")
    for tier_name in tier_dict['LIST'] :
        ntile         = tier_dict[tier_name]['ntile']
        if ntile == 0 : continue
        bands         = tier_dict[tier_name]['bands']
        t_expose_list = tier_dict[tier_name]['t_expose_list']
        t_string = ''
        for t in t_expose_list: t_string += f"{t:6.1f} "
        f.write(f"{pad}- {tier_name:10} {bands}     {t_string}   # texpose\n")

    f.write(f"DOCUMENTATION_END:\n")
    f.write(f"\n# ---------------------------------- \n\n")

    # end write_docana_keys

def remove_TEMP_files(outdir):

    logging.info(f" Remove {outdir}/{TEMP_PREFIX}*  files ")
    cmd = f"cd {outdir} ; rm {TEMP_PREFIX}* "
    os.system(cmd)

    return

def get_ncore(input_config_ana):
    # return dictionary for number of cores for each analysis stage.
    # ideally ncore = n_simlib, but cannot exceed NCORE_MAX
    
    NCORE_MAX = input_config_ana['NCORE_MAX']
    n_simlib  = input_config_ana['n_simlib']
    ncore     = n_simlib  # default

    # reduce ncore by factor of 2 until it is below NCORE_MAX
    while ncore > NCORE_MAX :
        ncore /= 2

    if ncore < 10: 
        ncore = 10  # at least 10 cores

    # set default number of cores for each stage
    ncore_sim   = ncore - 1   # -1 to better distribute each model among CPU jobs
    ncore_lcfit = ncore
    ncore_bbc   = ncore

    # try doubling ncore for slower LCFIT stage
    if ncore_lcfit < NCORE_MAX /2 :
        ncore_lcfit *= 2

    # halve number of BBC cores since this takes much less CPU than previous stages
    if ncore_bbc > 1:      
        ncore_bbc /= 2

    ncore_dict = {
        'SIM'      : int(ncore_sim),
        'LCFIT'    : int(ncore_lcfit),
        'BBC'      : int(ncore_bbc)
        # ?? WFIT ??
    }
    logging.info(f"  ncore_dict = {ncore_dict}")

    return ncore_dict
    # end get_ncore

def analysis_prep(args, input_config):

    input_config_ana    = input_config[NAME_CONFIG_ANALYSIS]
    input_config_survey = input_config[NAME_CONFIG_SURVEY]

    outdir = input_config_survey['OUTDIR']
    msg    = f"ANALYSIS PREP in {outdir}"
    print_stdout_header(msg)

    # get list of simlibs and count them
    wildcard    = f"{SURVEY_NAME}*.SIMLIB"
    simlib_list = sorted(glob.glob1(outdir,wildcard))
    n_simlib    = len(simlib_list)
    input_config_ana['n_simlib']    = n_simlib
    input_config_ana['simlib_list'] = simlib_list

    input_config_ana['ncore_dict'] = get_ncore(input_config_ana)

    # save original input file
    cp_cmd = f"cp {args.input_file} {outdir}/"
    os.system(cp_cmd)

    # and save instruent file too ...
    instr_file  = input_config['CONFIG_INSTRUMENT_FILE']
    cp_cmd = f"cp {instr_file} {outdir}/"
    os.system(cp_cmd)

    # ------------ prepare simulations ---------------

    analysis_readme_file = f"{outdir}/{ANALYSIS_INSTRUCTION_FILE}"
    logging.info(f"  Write analysis instructions to {analysis_readme_file}")
    f_readme = create_analysis_readme_file(analysis_readme_file)

    submit_file_sim   = analysis_prep_sim(input_config)
    submit_file_lcfit = analysis_prep_lcfit(input_config)
    submit_file_bbc   = analysis_prep_bbc(input_config)
    submit_list = [ submit_file_sim, submit_file_lcfit, submit_file_bbc ]

    prepare_submit_list(outdir, submit_list)

    # - - - - - - - 
    # write table of versions and associated parameters to analysis_readme_file
    write_readme_simlib_list(input_config_ana, f_readme)

    f_readme.close()
    return
    # end analysis_prep

def create_analysis_readme_file(analysis_readme_file):

    f_readme = open(analysis_readme_file,"wt")
    f_readme.write(f"# Instructions to run SIM + LCFIT + BBC + wfit \n")

    f_readme.write('#\n')
    f_readme.write(f"# To run all analysis stages:\n")
    f_readme.write(f"#   submit_batch_list.py --list_file {SUBMIT_ALL_FILE} >& {SUBMIT_ALL_LOG} & \n")
    f_readme.write(f"#   exit   (exit terminal to protect background job) \n")

    f_readme.write('#\n')
    f_readme.write(f"# To skip SIM and repeat LCFIT + BBC stages:\n")
    f_readme.write(f"#   submit_batch_list.py -l {SUBMIT_ALL_FILE} --nrowskip 1 >& {SUBMIT_ALL_LOG} & \n")
    f_readme.write(f"#   exit \n")

    f_readme.write('#\n')
    f_readme.write(f"# To skip SIM+LCFIT and repeat only BBC stage:\n")
    f_readme.write(f"#   submit_batch_list.py -l {SUBMIT_ALL_FILE} --nrowskip 2 >& {SUBMIT_ALL_LOG} & \n")
    f_readme.write(f"#   exit \n")

    # return file pointer so that other tasks can add more information
    return f_readme

def prepare_submit_list(outdir, submit_list):
    submit_all_file = f"{outdir}/{SUBMIT_ALL_FILE}"
    with open(submit_all_file,"wt") as s:
        s.write("SUBMIT_LIST:\n")
        for row in submit_list:
            s.write(f"  - {row}\n")

    return

def analysis_prep_sim(input_config):

    # create data and biasCor sim inputs to use with submit_batch_jobs.sh

    input_config_survey = input_config[NAME_CONFIG_SURVEY]
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_ana    = input_config[NAME_CONFIG_ANALYSIS]
    input_config_sim    = input_config_ana['SIM']

    outdir = input_config_survey['OUTDIR']
    
    # read contents of submit-template in starterKit area
    with open(SIMGEN_SUBMIT_TEMPLATE,"rt") as f:
        contents = f.readlines()

    # prepare two submit_batch_jobs inputs; 1) data and 2) biasCor
    SIM_BCOR_SCALE = input_config_sim['BCOR_SCALE']
    SIM_RANSEED    = input_config_sim['RANSEED'].split()

    SIM_REJECT_TRANSIENT_LIST = input_config_sim.setdefault('REJECT_TRANSIENT_LIST',['NONE'])
    if SIM_REJECT_TRANSIENT_LIST[0] == 'nonSNIa' : 
        REJECT_ALL_NONIA = True
    else:
        REJECT_ALL_NONIA = False

    SIM_PREFIX_DATA, SIM_PREFIX_BCOR = get_prefix_genversion(input_config_ana)
    INPUT_SUBMIT_SIM_DATA   = f"{outdir}/{PREFIX_SUBMIT_SIM}_{SIM_PREFIX_DATA}.INPUT"
    INPUT_SUBMIT_SIM_BCOR   = f"{outdir}/{PREFIX_SUBMIT_SIM}_{SIM_PREFIX_BCOR}.INPUT"

    
    logging.info(f"  Create {INPUT_SUBMIT_SIM_DATA}")
    logging.info(f"  Create {INPUT_SUBMIT_SIM_BCOR}")
    f_data = open(INPUT_SUBMIT_SIM_DATA,"wt")
    f_bcor = open(INPUT_SUBMIT_SIM_BCOR,"wt")

    # prepare user-input lines to add under GENOPT_GLOBAL.
    # Store key names to remove already-existing key names before
    # adding the user key.
    keylist_remove  = [ 'GENRANGE_PEAKMJD' ]  # remove line with key on this list
    genopt_global_linelist = []
    if 'GENOPT_GLOBAL' in input_config_sim :
        for key, arg in input_config_sim['GENOPT_GLOBAL'].items():
            keylist_remove.append(f"{key}:")
            genopt_global_linelist.append(f"  {key}:  {arg}")


    GENRANGE_MJD = input_config_survey['MJD_RANGE_TOTAL']
    peakmjd_min  = int(GENRANGE_MJD[0] - 30)
    peakmjd_max  = int(GENRANGE_MJD[1] + 20)
    mjd_min      = int(GENRANGE_MJD[0] - 1)
    mjd_max      = int(GENRANGE_MJD[1] + 1)
    genopt_global_linelist.append(f"  GENRANGE_PEAKMJD: {peakmjd_min} {peakmjd_max}")
    genopt_global_linelist.append(f"  GENRANGE_MJD:     {mjd_min} {mjd_max}")

    found_nonIa = False

    # - - - - - - - -
    for line in contents:
        line  = line.rstrip()  # remove trailing space and linefeed  
        line_data = line
        line_bcor = line
        wdlist = line.split()
        remove_line = False

        if len(line) > 0:
            is_comment = (line[0] == '#')
            remove_line = wdlist[0] in keylist_remove
        else:
            is_comment = False

        if 'GENVERSION_LIST' in line: 
            break   # stop copy when we reach this yaml key

        # suppress NONIa from biascor
        if SIMGEN_PREFIX not in line:
            found_nonIa = False
        if 'SIMGEN_INFILE_NON' in line:
            found_nonIa = True

        if found_nonIa :
            for reject in SIM_REJECT_TRANSIENT_LIST:
                if reject in line: line_data = '# ' + line_data
    
        #if 'FORMAT_MASK' in line:
        #    line_data = None
        #    line_bcor = line

        # double number of cores for BCOR (hard-wired)
        if 'BATCH_INFO' in line and not is_comment:
            NCORE_SIM  = input_config_ana['ncore_dict']['SIM']
            ncore_data = NCORE_SIM
            ncore_bcor = NCORE_SIM
            line_data = line.replace(wdlist[3], str(ncore_bcor) )
            line_bcor = line_data
            SBATCH_TEMPLATE = wdlist[2] # save this to use for BBC input

        # scale NGEN_UNIT for biascor (with user input)
        if 'NGEN_UNIT' in line and not is_comment :
            val_data = float(wdlist[1])
            val_bcor = val_data * SIM_BCOR_SCALE
            line_bcor = line.replace(wdlist[1], str(val_bcor) )

        # replace random seed with user-input seed
        if 'RANSEED_REPEAT' in line:
            ranseed = int(wdlist[2])
            line_data = line.replace( wdlist[2], str(SIM_RANSEED[0]) )
            line_bcor = line.replace( wdlist[2], str(SIM_RANSEED[1]) )

        # replace LOGIDR
        if 'LOGDIR' in line:
            LOGDIR        = wdlist[1]
            LOGDIR_DATA   = f"OUTPUT1_{SIM_PREFIX_DATA}"
            LOGDIR_BCOR   = f"OUTPUT1_{SIM_PREFIX_BCOR}"
            line_data = line.replace( wdlist[1], LOGDIR_DATA )
            line_bcor = line.replace( wdlist[1], LOGDIR_BCOR )
        
        if found_nonIa: 
            line_bcor = None  # always reject nonIA for biasCor
        if found_nonIa and REJECT_ALL_NONIA:
            line_data = None  # reject all nonIa if requested by user

        if remove_line:  # comment out specific nonSNIa lines to remind of its removal
            line_data = '# ' + line_data
            line_bcor = '# ' + line_bcor

        write_line_data_bcor(f_data, f_bcor, line_data, line_bcor)

        # insert SOLID_ANGLE 0 which is a flag to use SOLID_ANGLE in simlib file.
        if 'GENOPT_GLOBAL:' in line:
            line_add = "  SOLID_ANGLE:     0.0     " \
                       " # flag to use solid angle in simlib"
            write_line_data_bcor(f_data, f_bcor, line_add, line_add)

            for line_add in genopt_global_linelist :
                write_line_data_bcor(f_data, f_bcor, line_add, line_add)            

    # - - - - - - - - -
    line = "\nGENVERSION_LIST:"
    write_line_data_bcor(f_data, f_bcor, line, line)

    # explicitly write GENVERSION section for each simlib
    simlib_list = input_config_ana['simlib_list']

    # simlib file is of the form ROMAN_[string].SIMLIB,
    # so extract string to use in name of genversion

    GENVERSION_DATA_LIST = []
    GENVERSION_BCOR_LIST = []

    # tack on format mask
    WRITE_SPEC  = input_config_sim.setdefault('WRITE_SPEC', True)
    first_simlib = True

    for simlib in simlib_list :
        ind0            = len(SURVEY_NAME) + 1
        ind1            = simlib.index('.')
        string_simlib   = simlib[ind0:ind1]
        GENVERSION_DATA = f"{SIM_PREFIX_DATA}-{string_simlib}"
        GENVERSION_BCOR = f"{SIM_PREFIX_BCOR}-{string_simlib}"
        GENVERSION_DATA_LIST.append(GENVERSION_DATA)
        GENVERSION_BCOR_LIST.append(GENVERSION_BCOR)

        line_data = f"\n  - GENVERSION:  {GENVERSION_DATA}"
        line_bcor = f"\n  - GENVERSION:  {GENVERSION_BCOR}"
        write_line_data_bcor(f_data, f_bcor, line_data, line_bcor)

        line = "    GENOPT:"
        write_line_data_bcor(f_data, f_bcor, line, line)

        path_simlib = f"{CWD}/{outdir}/{simlib}"
        line = f"      SIMLIB_FILE: {path_simlib}"
        write_line_data_bcor(f_data, f_bcor, line, line)

        
        # write TAKE_SPECTRUM keys for prism (data only)
        write_take_spectrum(f_data, path_simlib, input_config)
        
        first_simlib = False

    # close submit-input files
    f_data.close()
    f_bcor.close()

    # store list of genversions
    input_config_ana['GENVERSION_DATA_LIST'] = GENVERSION_DATA_LIST
    input_config_ana['GENVERSION_BCOR_LIST'] = GENVERSION_BCOR_LIST
    input_config_ana['SBATCH_TEMPLATE']      = SBATCH_TEMPLATE

    # copy transient simgen-input files, 
    # and apply optional pre-scale using DNDZ_SCALE_NONIA key        
    cp_dir = os.path.dirname(SIMGEN_SUBMIT_TEMPLATE)
    for cp_file in COPY_INPUT_SIMGEN_LIST:
        logging.info(f"\t copy {cp_file}")
        cmd_cp = f"cp {cp_dir}/{cp_file} {outdir}/"
        os.system(cmd_cp)

    # check for prescale
    apply_sim_prescale(outdir, input_config_sim)
    return  os.path.basename(INPUT_SUBMIT_SIM_DATA)

    # end analysis_prep_sim

def write_take_spectrum(f, simlib_file, input_config):

    # write TAKE_SPECTRUM key(s) to pointer f (submit-sim file)
    # Beware that ntile and nvisit are read back from simlib file,
    # and thus association is fragile.

    input_config_survey = input_config[NAME_CONFIG_SURVEY]
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]
    simlib_base  = os.path.basename(simlib_file)

    key = 'TEXPOSE_PRISM'
    if key not in input_config_survey:
        return

    t_slew  = float(input_config_instr['PARAMS']['TIME_SLEW'])
    MJDMIN = input_config_survey['MJD_RANGE_TOTAL'][0]
    MJDMAX = input_config_survey['MJD_RANGE_TOTAL'][1]

    TEXPOSE_PRISM_LIST = input_config_survey[key]
    TEXPOSE_PRISM_DICT = {}  # list of Texpose for each field
    for str_prism in TEXPOSE_PRISM_LIST:
        field     = str_prism.split()[0]
        temp_list = re.findall("\[(.*?)\]", str_prism )[0]
        texpose_list = [float(x) for x in temp_list.split(',') ]
        TEXPOSE_PRISM_DICT[field] = texpose_list
        n_texpose_prism = len(texpose_list)

    TIME_SUM_PRISM = TIME_SUM_TD - input_config_survey['TIME_SUM_OBS']
    

    # read DOCANA block in SIMLIB to make sure we have correct info
    # about ntile and nvisit per tier
    line_list=  []
    with open(simlib_file, "r") as s:
        for line in s:
            if line.startswith("DOCUMENTATION_END:"): break
            line_list.append(line)
        docana = yaml.safe_load("\n".join(line_list))

    TIER_INFO = docana['DOCUMENTATION']['TIER_INFO'] 
    AREA_TOT  = docana['DOCUMENTATION']['AREA_TOT'] 

    logging.info(f"\t append TAKE_SPETRUM keys for {simlib_base}")

    for i_expose in range(0,n_texpose_prism):
        f.write(f"#   prism strategy {i_expose}\n")
        for str_tier in TIER_INFO:
            tier_info = str_tier.split()
            field     = tier_info[0].split('/')[0]     # e.g., SHALLOW, DEEP, etc...
            ntile     = int(tier_info[2])   # fragile alert for index !!!
            nvisit    = int(tier_info[3])   # fragile alert !!!
            area      = float(tier_info[4])
            dt_visit  = float(tier_info[5]) # fragile alert
            rel_area  = area/AREA_TOT

            # critial assumption: area fraction per tier = time fraction per tier
            t_sum_prism   = TIME_SUM_PRISM * 86400 * rel_area  # seconds
            texpose       = TEXPOSE_PRISM_DICT[field][i_expose]
            nexpose_prism = t_sum_prism/(texpose + t_slew)

            #print(f"\t\t xxx {field}  ntile={ntile} nvisit={nvisit}")
            frac_visit = nexpose_prism / (ntile * nvisit)
            ps         = 1.0/frac_visit  # prescale

            key = f"TAKE_SPECTRUM({field}/{ps:.2f}):"
            line_take_spectrum = f"{key:<30}  " \
                                 f"MJD({MJDMIN}:{MJDMAX}:{dt_visit})  " \
                                 f"TEXPOSE_ZPOLY({texpose})"
            f.write(f"      {line_take_spectrum}\n")

    return
    # end write_take_spectrum

def apply_sim_prescale(outdir, input_config_sim):

    # first, prepare prescale dictionary
    prescale_transient_dict = {}
    for str_ps in input_config_sim.setdefault('PRESCALE_TRANSIENT_LIST',[]):
        transient_name = str_ps.split('/')[0]
        ps             = str_ps.split('/')[1]
        prescale_transient_dict[transient_name] = ps

    # find all model-dependent simgen-input files
    simgen_input_list = sorted(glob.glob1(outdir,WILDCARD_INPUT_SIMGEN_TRANSIENT))

    # write DNDZ_SCALE_NON1A key to each simgen-input file where prescale is requested
    for inp in simgen_input_list:        
        for transient_name, ps in prescale_transient_dict.items():
            if transient_name in inp:
                with open(f"{outdir}/{inp}", "a+") as f:
                    scale = 1.0/float(ps)
                    f.write(f"DNDZ_SCALE_NON1A: {scale:.3f}   # append scale\n")
    return

def analysis_prep_lcfit(input_config):

    # create data and biasCor sim inputs to use with submit_batch_jobs.sh

    input_config_survey = input_config[NAME_CONFIG_SURVEY]
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_ana    = input_config[NAME_CONFIG_ANALYSIS]
    input_config_sim    = input_config_ana['SIM']

    outdir              = input_config_survey['OUTDIR']
    FITOPT_GLOBAL       = input_config_ana['LCFIT'].setdefault('FITOPT_GLOBAL',None)

    SIM_BCOR_SCALE = input_config_sim['BCOR_SCALE']

    with open(SIMFIT_SUBMIT_TEMPLATE,"rt") as f:
        contents = f.readlines()

    PREFIX_VERSION_DATA, PREFIX_VERSION_BCOR = get_prefix_genversion(input_config_ana)

    INPUT_SUBMIT_LCFIT_DATA   = f"{outdir}/{PREFIX_SUBMIT_LCFIT}_{PREFIX_VERSION_DATA}.NML"
    INPUT_SUBMIT_LCFIT_BCOR   = f"{outdir}/{PREFIX_SUBMIT_LCFIT}_{PREFIX_VERSION_BCOR}.NML"


    logging.info(f"  Create {INPUT_SUBMIT_LCFIT_DATA}")
    logging.info(f"  Create {INPUT_SUBMIT_LCFIT_BCOR}")


    GENVERSION_DATA_LIST = input_config_ana['GENVERSION_DATA_LIST']
    GENVERSION_BCOR_LIST = input_config_ana['GENVERSION_BCOR_LIST']

    f_data = open(INPUT_SUBMIT_LCFIT_DATA,"wt")
    f_bcor = open(INPUT_SUBMIT_LCFIT_BCOR,"wt")

    is_config = False
    last_line = ''

    for line in contents:
        line  = line.rstrip()  # remove trailing space and linefeed  
        line_data = line
        line_bcor = line
        wdlist    = line.split()

        if 'CONFIG:'     in line : is_config = True
        if 'DONE_CONFIG' in line : is_config = False

        # set NCORE
        if 'BATCH_INFO' in line:
            NCORE_LCFIT = input_config_ana['ncore_dict']['LCFIT']
            line_data = line.replace( wdlist[3], str(NCORE_LCFIT) )
            line_bcor = line.replace( wdlist[3], str(NCORE_LCFIT) )
            
        # replace VERSION to process in CONFIG block
        if 'VERSION:' in last_line and is_config : 
            line_data = f"  - {PREFIX_VERSION_DATA}*"
            line_bcor = f"  - {PREFIX_VERSION_BCOR}*"

        # under &SNLCINP, substitute a valid version to enable interactive testing
        # Note that fortran namelist arg includes single quotes
        if 'VERSION_PHOTOMETRY' in line and not is_config :
            arg = f"'{GENVERSION_DATA_LIST[0]}'" 
            line_data = line.replace( wdlist[2], arg )
            arg = f"'{GENVERSION_BCOR_LIST[0]}'"
            line_bcor = line.replace( wdlist[2], arg )

        # xxxx mark xxxxxx
        # set SNTYPE_LIST argument to select only true SNIa
        #if 'SNTYPE_LIST' in line:
        #    line_data = f"   SNTYPE_LIST = {SNTYPE_LIST}  ! select only true SNIa"
        #    line_bcor = line_data
        # xxxxx 

        # replace OUTDIR
        if 'OUTDIR' in line:
            OUTDIR_DATA = f"OUTPUT2_LCFIT_{PREFIX_VERSION_DATA}"
            OUTDIR_BCOR = f"OUTPUT2_LCFIT_{PREFIX_VERSION_BCOR}"
            line_data = line.replace(wdlist[1], OUTDIR_DATA)
            line_bcor = line.replace(wdlist[1], OUTDIR_BCOR)
            input_config_ana['OUTDIR_LCFIT_DATA'] = OUTDIR_DATA
            input_config_ana['OUTDIR_LCFIT_BCOR'] = OUTDIR_BCOR
            
            # write extra FITOPT_GLOBAL line before OUTDIR
            if FITOPT_GLOBAL is not None:
                line_global = f"  FITOPT_GLOBAL: {FITOPT_GLOBAL}\n"
                write_line_data_bcor(f_data, f_bcor, line_global, line_global)

        write_line_data_bcor(f_data, f_bcor, line_data, line_bcor)
        last_line = line
    
    return  os.path.basename(INPUT_SUBMIT_LCFIT_DATA)

    # end analysis_prep_lcfit

def analysis_prep_bbc(input_config):

    # create data and biasCor sim inputs to use with submit_batch_jobs.sh

    input_config_survey = input_config[NAME_CONFIG_SURVEY]
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_ana    = input_config[NAME_CONFIG_ANALYSIS]

    outdir              = input_config_survey['OUTDIR']

    PREFIX_VERSION_DATA, PREFIX_VERSION_BCOR = get_prefix_genversion(input_config_ana)
    INPUT_SUBMIT_DATA   = f"{outdir}/{PREFIX_SUBMIT_BBC}_{PREFIX_VERSION_DATA}.INPUT"

    logging.info(f"  Create {INPUT_SUBMIT_DATA}")

    prep_bbc_field_list(input_config)

    f = open(INPUT_SUBMIT_DATA,"wt")

    # there is no template input file here, so write all of the keys
    SBATCH_TEMPLATE    = input_config_ana['SBATCH_TEMPLATE']
    OUTDIR_LCFIT_DATA  = input_config_ana['OUTDIR_LCFIT_DATA']
    NCORE_BBC          = input_config_ana['ncore_dict']['BBC']

    f.write(f"CONFIG:\n")
    f.write(f"  BATCH_INFO:  sbatch  {SBATCH_TEMPLATE}  {NCORE_BBC} \n")
    f.write(f"  BATCH_WALLTIME:  '2:00:00'  \n")    
    f.write(f"  BATCH_MEM:        4GB \n\n")

    f.write(f"  INPDIR+:\n")
    f.write(f"  - {CWD}/{outdir}/{OUTDIR_LCFIT_DATA} \n")

    input_config_BBC = input_config_ana['BBC']
    key_list = [ 'INPFILE+', 'MUOPT' ]
    for key in key_list:
        if key in input_config_BBC:
            f.write(f"\n  {key}: \n")
            for item in input_config_BBC[key]:
                f.write(f"  - {item}\n")

    if 'WFIT' in input_config_ana:
        f.write(f"\n  WFITMUDIF_OPT: \n")
        arglist = input_config_ana['WFIT']['ARGLIST']
        for arg in arglist:
            f.write(f"  - {arg} \n")

    outdir_bbc = f"OUTPUT3_BBC_{PREFIX_VERSION_DATA}"
    f.write(f"\n  OUTDIR:  {outdir_bbc} \n")

    f.write(f"\n#END_YAML  \n\n")
    
    zbin_info = input_config_BBC['ZBIN_INFO'].split()
    zmin    = zbin_info[0]
    zmax    = zbin_info[1]
    nzbin   = zbin_info[2]
    powzbin = zbin_info[3]

    # - - - - - 
    key_bbc_dict = {
        'prefix'        : [ 'OVERRIDE' , "prefix for output defined by submit_batch"],
        'datafile'      : [ 'OVERRIDE' , "input FITRES file; defined by submit_batch_jobs" ],
        'surveygroup_biascor' : [ 'FOUNDATION(zbin=.02),ROMAN(zbin=.1)',  "biasCor z bins" ],
        'lensing_zpar'  : [ 0.055 ,      "add lenzing_zpar*z uncertainty to muerr " ],
        'fitflag_sigmb' : [ 1,           "vary sig_int until chi2/dof ~ 1" ],
        'redchi2_tol'   : [ 0.02,        "tolerance on chi2/dof = 1" ],
        'sig1'          : [0.11,         "initial sig_int for fit" ],
        'skip1'         : [None,          "" ],             # leave blank line
        'zmin'          : [f"{zmin}",         "min redshift" ],
        'zmax'          : [f"{zmax}",          "max redshift" ],
        'nzbin'         : [f"{nzbin}",           "number of redshift bins" ],
        'powzbin'       : [f"{powzbin}",            "z bin size proportional to (1+z)^powzbin" ],
        'min_per_zbin'  : [ 5,            "min per z bin" ],
        'x1min'         : [-3.0,          "min x1" ],
        'x1max'         : [ 3.0,          "max x1" ],
        'cmin'          : [-0.3,          "min color" ],
        'cmax'          : [0.3,           "max color" ],
        'skip2'         : [None,          "" ],
        'CUTWIN'        : [None,          "" ],  # substitute user inputs
        'skip3'         : [None,          "" ],
        'p1'            : [0.14,          "initial alpha for fit" ],
        'p2'            : [3.1,           "initial beta for fit" ],
        'p5'            : [0.0,           "initial gamma for fit" ],
        'p7'            : [10,            "initial mass step" ],
        'p9'            : [ 0.685,        "ref Omega_LAM" ],
        'p10'           : [ 0.0,          "ref Omega_k" ],
        'p11'           : [-1.0,          "ref w0" ],
        'p12'           : [ 0.0,          "ref wa" ],
        'u1'            : [1,             "float alpha" ],
        'u2'            : [0,             "fix beta to reduce outliers" ],
        'skip9'         : [None,           "" ]
    }

    
    key_cutwin = 'CUTWIN'
    for key, val_list in key_bbc_dict.items():  
        val     = val_list[0]
        comment = val_list[1]
        
        if 'skip' in key : 
            f.write(f"\n")
        elif key == key_cutwin :
            write_bbc_cutwin_keys(f, key, input_config_BBC)
        else:
            key_val = f"{key}={val}"
            f.write(f"{key_val:20s}     # {comment}\n")

            if key == 'datafile' :   # write biascor info after data
                write_biascor_keys(f,input_config_BBC)


    f.write(f"  \n")
    
    return  os.path.basename(INPUT_SUBMIT_DATA)

    # end analysis_prep_bbc

def prep_bbc_field_list(input_config):

    input_config_survey = input_config[NAME_CONFIG_SURVEY]
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]
    input_config_ana    = input_config[NAME_CONFIG_ANALYSIS]

    # copy list of tier [FIELD] names to BBC part of config.
    # Keep only unique field names after removing characters after slash.
    # E.g., MEDIUM/1, MEDIUM/2, MEDIUM/3 -> 1-element list with MEDIUM

    FIELD_LIST_UNIQUE = []
    for tier in input_config_survey['TIER_INPUT_LIST']:
        field = tier[0].split('/')[0]  # tier name is field name in sim
        if field not in FIELD_LIST_UNIQUE:
            FIELD_LIST_UNIQUE.append(field)

    input_config_ana['BBC']['FIELD_LIST'] = FIELD_LIST_UNIQUE
    return

def write_biascor_keys(f,input_config_BBC):

    # write biasCor keys to bbc-input file for SALT2mu.exe program

    NDIM_BIASCOR = 1  # 1 or 5; hard-coded for now; read user input later

    if NDIM_BIASCOR == 1 :
        logging.info(f"\t  Prepare input keys for 1D biasCor")
        f.write(f"simfile_biascor=datafile \n")
        f.write(f"opt_biascor=66           # 2=1D biascor, wgt=1/muerr^2; 64=vs.IDSAMPLE\n");

        FIELD_LIST = input_config_BBC['FIELD_LIST']
        # define unique fields after remove slash
        fieldGroup = ','.join(FIELD_LIST)
        if len(FIELD_LIST) > 1 :
            f.write(f"fieldGroup_biascor={fieldGroup}    # separate biasCor per tier\n")

    else:
        logging.info(f"\n WARNING: no biasCor specified for BBC.")

    # - - - - - - 
    f.write(f"\n")
    return

def write_bbc_cutwin_keys(f, key, input_config_BBC):
    if key in input_config_BBC:
        for arg in input_config_BBC[key]:
            f.write(f"{key}  {arg}\n")
    return 
    # end write_bbc_key

def write_line_data_bcor(f_data, f_bcor, line_data, line_bcor):

    if line_data is not None:
        f_data.write(f"{line_data} \n")

    if line_bcor is not None:
        f_bcor.write(f"{line_bcor} \n")
    return
    # end write_line_data_bcor

def update_readme(f_readme, INPUT_SUBMIT_DATA, INPUT_SUBMIT_BCOR):

    # .xyz
    if INPUT_SUBMIT_DATA is not None:
        base_file = os.path.basename(INPUT_SUBMIT_DATA)
        log_file = "OUT_" + base_file.split('.')[0] + ".LOG"            
        f_readme.write(f"# {program_submit}  {base_file}  >& {log_file} & \n")
        f_readme.write(f"#    [wait until slurm jobs to finish] \n")

    if INPUT_SUBMIT_BCOR is not None:
        base_file = os.path.basename(INPUT_SUBMIT_BCOR)
        log_file = "OUT_" + base_file.split('.')[0] + ".LOG" 
        f_readme.write(f"# {program_submit}  {base_file}  >& {log_file} & \n")
        f_readme.write(f"#    [wait until slurm jobs to finish] \n")

    f_readme.write('# \n')
    return
    # end update_readme

def  write_readme_simlib_list(input_config_ana, f_readme):
    
    # write table of simlib file and indices (ia,it.iz) to readme

    n_simlib             = input_config_ana['n_simlib']
    simlib_list          = input_config_ana['simlib_list']
    GENVERSION_DATA_LIST = input_config_ana['GENVERSION_DATA_LIST'] 

    f_readme.write(f"\n# ------------------------------------------------\n")
    f_readme.write(f"# Summary of {n_simlib} simulation jobs.\n")

    f_readme.write("\nVARNAMES: ROW  VERSION  SIMLIB_FILE  i_AREA i_TEXPOSE i_zSNRMAX \n")
    len_survey = len(SURVEY_NAME)
    rownum = 0
    for version, simlib in  zip(GENVERSION_DATA_LIST,simlib_list):
        rownum += 1
        ia = int(simlib.split("-a")[1][0:2])
        it = int(simlib.split("-t")[1][0:2])
        iz = int(simlib.split("-z")[1][0:2])
        f_readme.write(f"ROW: {rownum:3}  {version}  {simlib}   {ia:2d} {it:2d} {iz:2d} \n")

    return
    # end write_readme_simlib_list
    
def get_prefix_genversion(input_config_analysis):
    SIM_PREFIX      = input_config_analysis['SIM']['PREFIX']
    GENVERSION_DATA = f"{SIM_PREFIX}_DATA"
    GENVERSION_BCOR = f"{SIM_PREFIX}_BCOR"

    # xxx mark GENVERSION_DATA = f"{SIM_PREFIX}_DATA_{SURVEY_NAME}"
    # xxx mark GENVERSION_BCOR = f"{SIM_PREFIX}_BCOR_{SURVEY_NAME}"

    return GENVERSION_DATA, GENVERSION_BCOR
    #end prefix_genversion

def print_stdout_header(msg):
    logging.info(f"")
    logging.info(f"# ================================================================")
    logging.info(f"# ============== {msg} ===============")
    logging.info(f"# ================================================================")
    return
    

def INP_copy_and_modify(input_config):

    # input file is a list of instructions to copy and modify 
    # input config files. Thus only base config files need to be
    # manually created and this utuility creates modified versions
    # of them

    new_file_list = []
    for base_file in input_config:
        logging.info(f" Found base config file: {base_file}")
        if not os.path.exists(base_file):
            sys.exit(f"\n ERROR: cannot find base config file {base_file}")

        for new_file in input_config[base_file]:
            key_replace_dict = input_config[base_file][new_file]
            logging.info(f"\t copy+modify {new_file} ")
            copy_and_modify(base_file, new_file, key_replace_dict)
            new_file_list.append(new_file)

    # construct bash script to run all the new_file tasks
    run_script = "RUN_makeSimlibs+Analysis.sh"
    with open(run_script,"wt") as r:
        for new_file in new_file_list:
            log_file = "OUT_" + new_file.split('.')[0] + ".LOG"
            r.write(f"{sys.argv[0]}  {new_file} \\\n\t >& {log_file} &\n")

    os.system(f"chmod +x {run_script}")
    logging.info(f" Launch them all with {run_script}")

    return

def copy_and_modify(base_file, new_file, key_replace_dict):
    
    with open(base_file,"rt") as f:
        contents = f.readlines()

    # check for special FILTER replacment for each tier;
    # here unpack and store info to use below.
    filter_dict = {}
    if 'FILTER_REPLACE' in key_replace_dict:
        for row in key_replace_dict['FILTER_REPLACE']:
            filter_orig    = row.split()[0]
            filter_replace = row.split()[1]
            filter_dict[filter_orig] = filter_replace

    with open(new_file,"wt") as f:
        f.write(f"# AUTO-GENERATED by {sys.argv[0]}    \n")
        f.write(f"# FILTER_REPLACEMENTS: {filter_dict} \n")
        f.write(f"\n")

        for line_in in contents:
            line_out = line_in.strip('\n')
            wdlist   = line_out.split()
            if len(wdlist) >= 2 :
                key      = wdlist[0]
                val      = wdlist[1]
                for key_replace, val_replace in key_replace_dict.items():
                    if key_replace+':' == key:
                        line_out = line_out.replace(str(val),str(val_replace))
                        line_out += '   # REPLACED '
             
            # check for filter replacements
            for filter_orig, filter_replace in filter_dict.items():
                if filter_orig in line_out:
                    line_out = line_out.replace(filter_orig,filter_replace)
            
            f.write(f"{line_out}\n")

    return

# =============================================
if __name__ == "__main__":

    setup_logging()
    args  = get_args()
    input_config = get_inputs(args.input_file)

    command = ' '.join(sys.argv)
    logging.info(f"Command: {command}\n")

    if args.skip_to_analysis:
        analysis_prep(args, input_config)
        exit(0)
 
    if NAME_CONFIG_SURVEY not in input_config:
        INP_copy_and_modify(input_config)
        exit(0)

    input_config_survey = input_config[NAME_CONFIG_SURVEY]
    input_config_instr  = input_config[NAME_CONFIG_INSTRUMENT]

    # convert 3D indices to flattend 1D arrays for easier looping below
    na, nt, nz, index3d_dict = get_index3d_dict(input_config_survey)

    # convert user-input band info to dictionary labeled by band symbol
    band_dict = init_band_dict(input_config_instr)

    # convert tier info to dictionary labeled by tier name
    tier_dict = init_tier_dict(input_config_survey)

    mkdir_output(input_config_survey)

    # grand loop over areaFractin, time between visits and redshift to match SNR
    n_simlib = 0
    simlib_dict = {}

    for iz in range(0,nz):

        msg = f"SOLVE FOR TEXPOSE (iz={iz})"
        print_stdout_header(msg)

        # solve for exposure times
        for tier_name in tier_dict['LIST'] :
             solve_exposure_time(tier_name, tier_dict, band_dict, input_config, iz)

        for ia, it, izdum in zip(index3d_dict[KEYNAME_REL_AREA], 
                                 index3d_dict[KEYNAME_DT_VISIT],
                                 index3d_dict[KEYNAME_zSNRMATCH] ):
            if iz != izdum : continue
            index1d_list = [ ia, it, iz ]
            # solve for area containing nvisits
            solve_area(tier_dict, band_dict, input_config, index1d_list)

            # write simlib
            n_simlib += 1
            simlib_file = \
                write_simlib(args, input_config, tier_dict, band_dict, index1d_list )
            
            simlib_dict[simlib_file] = index1d_list

    logging.info("")

    #remove_TEMP_files(input_config_survey['OUTDIR'])
    logging.info(f" Done making {n_simlib} simlib files.")

    # - - - - - - - -

    analysis_prep(args,input_config)

    logging.info("DONE")

    exit(0);
    # end main

