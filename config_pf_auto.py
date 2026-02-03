import os
import shutil
from datetime import datetime

import logging

#showPlot = False
showTable = False

#######################################
# ------Model specific settings
#######################################
#### Temporal Settings
#######################################

# Start and end year of optimization
start_year = 2025
end_year = 2025

calc_start = datetime(start_year, 1, 1, 00, 00)
calc_end = datetime(end_year, 12, 31, 23, 00)

# sets number of time steps. set 'None'to get full year, else set number of time steps in one year to look at
number_of_time_steps = None #8*24 #None


# Auxiliary years (5 year steps)
aux_years = True  # True to use representative years and select the steps. False to optimize each year individually
aux_year_steps = 5  # years

# choose, if multi-period perfect foresight optimization (True) or myopic (False) 
# be aware: 'myopic' is currently not really 'myopic', but only available for one single year
# if it shall become a real myopic optimization, the myopic optimization loop must be programmed in 'laend_module_pf_auto.py'
# and accessing the results after each optimization step must be programmed in utils_pf_auto.py:
multiperiod_pf = True

#######################################
#### Set Optimization Objective
#######################################

objective = [
    'Costs',
    #'EnvCosts',
    #'climate change',
    #'JRCII',
    #'Equilibrium',
    #'acidification',
    #'climate change: biogenic',
    #'climate change: fossil',
    #'climate change: land use and land use change',
    #'ecotoxicity: freshwater',
    #'ecotoxicity: freshwater, inorganics',
    #'ecotoxicity: freshwater, organics',
    #'energy resources: non-renewable',
    #'eutrophication: freshwater',
    #'eutrophication: marine',
    #'eutrophication: terrestrial',
    #'human toxicity: carcinogenic',
    #'human toxicity: carcinogenic, inorganics',
    #'human toxicity: carcinogenic, organics',
    #'human toxicity: non-carcinogenic',
    #'human toxicity: non-carcinogenic, inorganics',
    #'human toxicity: non-carcinogenic, organics',
    #'ionising radiation: human health',
    #'land use',
    #'material resources: metals/minerals',
    #'ozone depletion',
    #'particulate matter formation',
    #'photochemical oxidant formation: human health',
    #'water use'
]
# Multiprocessing of several objectives
# True if several objectives should run in parallel to speed up the calculation time
# False for testing and to use debug mode, since debug messages are not displayed
# during multiprocessing
multiprocessing = False

#######################################
#### Scenario Excel file
#######################################

filename_configuration = ''

#######################################
#### Location specific settings
#######################################

location_name = ''

#######################################
#### TMY
#######################################

# import typical meteorological year (TMY) as downloaded from https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY for specific location
filename_tmy = 'in/pvgis/tmy_47.658_9.169_2007_2016.csv'


# #######################################
##### Location settings
# #######################################
timezone = 'Europe/Berlin',
latitude = 47.658
longitude = 9.169


#######################################
#### Fixed demand
#######################################
# Get new electricity profile and save to filename_configuration (copy of Scenario Excel in files folder)
update_electricity_demand = False
filename_el_demand = 'in/el_demand_bdew+emob.csv'
varname_el_demand = 'load_el'

# Get new heat profile and save to filename_configuration (copy of Scenario Excel in files folder)
update_heat_demand = False
filename_th_demand = 'in/heat_demand_bdew.csv'

separate_heat_water = True  # if True two load curves for space heat and hot water are generated
                            # only relevant when update_heat_demand = True

varname_th_low = 'load_th_low' # name used in scenario.xlsx for thermal load for low temperature space heat;
varname_th_high = 'load_th_high' # name used in scenario.xlsx for thermal load for hot water

# BDEW heat demand input data
ann_demands_per_type = {'efh': 0,
                        'mfh': 960990,
                        'ghd': 19613}

building_class = 10
# class of building according to bdew classification possible numbers are: 1 - 11
# according to https://www.eko-netz.de/files/eko-netz/download/3.5_standardlastprofile_bgw_information_lastprofile.pdf
#    Altbauanteil    mittl. Anteile von
#     von    bis     Altbau   Neubau
# 1  85,5%    90,5%    88,0%    12,0%
# 2  80,5%    85,5%    83,0%    17,0%
# 3  75,5%    80,5%    78,0%    22,0%
# 4  70,5%    75,5%    73,0%    27,0%
# 5  65,5%    70,5%    68,0%    32,0%
# 6  60,5%    65,5%    63,0%    37,0%
# 7  55,5%    60,5%    58,0%    42,0%
# 8  50,5%    55,5%    53,0%    47,0%
# 9  45,5%    50,5%    48,0%    52,0%
# 10 40,5%    45,5%    43,0%    57,0%
# 11                   75,0%    25,0% Durchschnitt DE
# Altbau bis 1979; auf Basis von Wohneinheiten

building_wind_class = 0  # 0=not windy, 1=windy


#######################################
#### Renewable Energy Curves
#######################################

# --- PV ---
update_pv_opt_fix = False
# PV performance as downloaded from https://re.jrc.ec.europa.eu/pvg_tools/en/#PVP for specific location, PV technology, system loss, mounting position, slope, azimuth
# crist. Si, 14 % system loss
filename_pv_opt_fix = 'in/pvgis/Timeseries_47.658_9.169_SA_1kWp_crystSi_14_36deg_4deg_2007_2016.csv'
varname_pv_1 = 'PV_flat_roof'
varname_pv_2 = 'PV_og'

update_pv_facade_fix = False
# crist. Si, 14 % Systemverlust
filename_pv_facade_fix = 'in/pvgis/Timeseries_47.658_9.169_SA2_1kWp_crystSi_14_90deg_47deg_2007_2016.csv'
varname_pv_3 = 'PV_facade'

# --- Wind ---
wind_z0_roughness = 0.1 # surface roughness length in m; bare rocks, sparely vegetated areas = 0.01; 
                        # arable land, pastures, nat. grasslands = 0.1; forest = 0.9 
                        # [https://www.researchgate.net/figure/Surface-roughness-length-m-for-each-land-use-category-in-WAsP-and-WRF-and-the_tbl1_340415172]

# --- Solar ---
# Solar Collector data (https://shop.ssp-products.at/Flachkollektor-SSP-Prosol-25-251m-)
update_Solar_Collector_data = False
varname_solar_collector_high = 'solar_thermal_FPC_high'
varname_solar_collector_low = 'solar_thermal_FPC_low'
collector_tilt = 36
collector_azimuth = 5
a_1 = 3.594  # [W/(m²K)] Thermal loss parameter k1
a_2 = 0.014  # [W/(m²K²)] Thermal loss parameter k2
eta_0 = 0.785  # Optical efficiency of the collector
temp_collector_inlet = 50  # [°C] Collectors inlet temperature (to do: should be specific for high and low temp. level)
delta_temp_n = 10  # Temperature difference between collector inlet and mean temperature (of inlet and outlet temperature♦)

# --- Heatpump ---
# heatpump air water data
update_heatpump_a_w_cop = False

varname_a_w_hp_high = 'heat_pump_a_w_high'
hp_temp_high = 55  # °C

varname_a_w_hp_low = 'heat_pump_a_w_low'
hp_temp_low = 40  # °C

a_w_hp_quality_grade = 0.4  # 0.4 is default setting for air/water heat pumps
# [°C] temperature below which icing occurs at heat exchanger
hp_temp_threshold_icing = 2
# [0<f<1] sets the relative COP drop by icing; 1 = no efficiency drop
hp_factor_icing = 0.8


#######################################
#### LCA
#######################################

#define brightway2 project to be used as reference
bw2_project = 'laend'
#define database to be used within bw2_project
bw2_database = 'own_calculations'
#define Impact Assessment Method and optionally impact category to be used
bw2_method = 'EF v3.1'

#Do NOT change this list unless you're changing the impact assessment method (bw2_method) with associated weighting & normalization
#Then also have a look at utils def determineGoalForObj and def determine CfactorForSolver 

system_impacts_index = [
    'Costs',
    'acidification',
    'climate change',
    'climate change: biogenic',
    'climate change: fossil',
    'climate change: land use and land use change',
    'ecotoxicity: freshwater',
    'ecotoxicity: freshwater, inorganics',
    'ecotoxicity: freshwater, organics',
    'energy resources: non-renewable',
    'eutrophication: freshwater',
    'eutrophication: marine',
    'eutrophication: terrestrial',
    'human toxicity: carcinogenic',
    'human toxicity: carcinogenic, inorganics',
    'human toxicity: carcinogenic, organics',
    'human toxicity: non-carcinogenic',
    'human toxicity: non-carcinogenic, inorganics',
    'human toxicity: non-carcinogenic, organics',
    'ionising radiation: human health',
    'land use',
    'material resources: metals/minerals',
    'ozone depletion',
    'particulate matter formation',
    'photochemical oxidant formation: human health',
    'water use',
    'JRCII',
    'EnvCosts',
    'Equilibrium'
]

###############################################################################
#### Normalisation & Weighting for objectives JRCII, EnvCosts, Equilibrium
###############################################################################

filename_weight_and_normalisation = 'in/Normalisation and Weighting.xlsx'
# GDP 2018 with current prices for 2010 to be in line with environmental normalisation (for calculation see GDP_2010_Euro.xlsx) (todo: auf 2021 aktualisieren)
normalization_cost_gdp = 4.63113E+13

# Weighting factor for costs, if scenario requires setting
# For objective EnvCosts weight of Environmental Footprint and costs can be choosen,
# Must be a decimal between 0 and 1 for multi-objective "EnvCosts"
weight_cost_to_env = 0.5
# Equilibrium means equal weighting of all impacts but allows for setting all weights individually (setting for environmental impacts in Excel file)
# 1/17 if equal weighting between every single goal for multi-objective "Equilibrium"
weight_cost_to_env_equilibrium = 1/17


#######################################
#### Emission constraint
#######################################

emission_constraint = False
ec_horizon = 5
ec_impact_category = 'climate change'
ec_buffer = 0.0001
ef_fuel_based_only = False


#######################################
#### Climate neutrality
#######################################

def_cn_calculate_climate_neutrality = False
def_cn_include_investment = True
def_cn_year_climate_neutrality = 2045
def_cn_fuel_based_only = False


#######################################
#### Financial settings
#######################################

# InvestCostDecrease = 0.01  # Annual cost decrease (technical progress)

InvestWacc = 1e-11  #weighted average cost of capital, part of annuity calculation based on oemof.economics.annuity
DiscountRate = 1e-11  #pay attention: if discount rate is changed to >0, environmental impacts will also be discounted
# kW (kWh for storage) investments below this value do not get passed as existing capacity to next year
Invest_min_threshold = 0.1 #only applicable to myopic
InvestTimeSteps = 1 if aux_years == False else aux_year_steps


###############################################################################
#### Technical settings
###############################################################################

# Set the logging level
log_screen_level = logging.INFO
log_file_level = logging.DEBUG


# changing the following config variables may result in errors! Proceed with caution!
ci = ['converters_in', 'renewables', 'storages', 'converters_out']

# granularity of calculation; chose 'D' for daily or 'H' for hourly
granularity = 'H'

# Solver
solver = 'gurobi'  # 'cplex', 'glpk', 'gurobi',....
solver_verbose = True  # show/hide solver output
solver_options_on = True
solver_options = {
    # When using gurobi, parameters are "threads", "feasibilityTol", "optimalityTol";
    # When using cbc, parameters should be "threads", "primalTolerance", "dualTolerance"
    'threads': 14,
    'feasibilitytol': 1e-6, #gurobi-standard: 1e-6
    'optimalitytol': 1e-6 #gurobi-standard: 1e-6
} if solver_options_on == True else {}

continue_despite_storage_issues = None


def createLogPathName(filename_f):
    now = datetime.now()
    time = str(now)
    time = time[:-7]
    time = time.replace(':', '-')

    # creates name for folder where all data is stored
    name = os.path.dirname(os.path.realpath(__file__)) + '\\' + 'runs' + '\\' + str(now.date()) + '_' + str(
        now.hour) + '-' + str(now.minute) + '-' + str(now.second)

    # if folder does not exist, creates folder and copies config.py and scenario.xlsx
    if not os.path.exists(name):
        os.makedirs(name)

        # copies config.py to runs folder
        src = os.path.dirname(os.path.realpath(__file__)) + '\\' + 'config_pf_auto.py'
        dst = name + '\\' + f'laend_config_pf_{time}.py'
        shutil.copyfile(src, dst)

        # creates subfolder for files
        if not os.path.exists(name + '\\' + 'files'):
            os.makedirs(name + '\\' + 'files')

        # creates subfolder for timeseries
        if not os.path.exists('in' + '\\' + location_name):
            os.makedirs('in' + '\\' + location_name)

            # creates subfolder for oemof dumps
        if not os.path.exists(name + '\\' + 'oemof_dumps'):
            os.makedirs(name + '\\' + 'oemof_dumps')

        # copies scenario.xlsx to files folder
        src = os.path.dirname(os.path.realpath(__file__)) + \
            '\\' + filename_f
        dst = name + '\\' 'files\\' + f'{time}_{filename_f}'
        shutil.copyfile(src, dst)

    if not os.path.exists(name + '\\logs'):
        os.mkdir(name + '\\logs')

    return name, time
