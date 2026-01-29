# import general libraries
import logging
import math
import pandas as pd
from datetime import datetime
import numpy as np
import os
import sys
import copy

# import oemof
import oemof.solph as solph
from oemof.tools import economics

# import oemof auxiliary libraries
from windpowerlib import ModelChain, WindTurbine, power_output

#    from oemof.thermal import solar_thermal_collector,
#    import oemof.thermal.compression_heatpumps_and_chillers as cmpr_hp_chiller
#    import demandlib.bdew as bdew
#    as futher potential auxiliary libraries


# import files
import config_pf_auto as config_laend

# import LCA connection
import brightway2 as bw2



########################################################
#### Functions used in main()
########################################################

def defineYearsForCalculation():
    '''
    (1) Creates a list of representative years (based on config_laend.py) AND
    (2) creates a DateTimeIndex and list of periods of representative years
        including hourly breakdown that is later used for computation

    Returns
    -------
    calc_years: list of representative years (without hourly breakdown)
    timeindex: pandas.DatetimeIndex, without considering leap days
    periods: representative years, including hourly breakdown, without considering leap days
    '''
    
    #(1) Create list of representative years:
    calc_years_lst = []
    start = config_laend.start_year
    end = config_laend.end_year
    
    calc_year = start
    if config_laend.aux_years == True:
        for i in range(1, math.floor((end - start) / config_laend.aux_year_steps) + 2):
            calc_years_lst.append(calc_year)
            calc_year = start + i * config_laend.aux_year_steps
    else:
        for i in range(1, end - start + 2):
            calc_years_lst.append(calc_year)
            calc_year = start + i * 1
            
    
    #(2) Create DateTimeIndex (timeindex) and List of Periods (periods)
    #    for later use within energy system computation
    if config_laend.number_of_time_steps == None:
        tstp = 8760
    else:
        tstp = config_laend.number_of_time_steps
    
    t_idx_series_dict = {}
    periods = []
    for i in calc_years_lst:
        #t_idx = pd.date_range(f'1/1/{i}',f'12/31/{i}/23' , freq='H')
        t_idx = pd.date_range(start=f'1/1/{i}', periods=tstp, freq='h')
        
        #as the timeindex is set by periods variable, leap day search is deprecated
        # check, if is leap year and if so, delete leap day
        #if t_idx.is_leap_year.all():
        #    t_idx = t_idx[~((t_idx.month==2) & (t_idx.day==29))]
        #---------
        
        t_idx_series = pd.Series(index=t_idx, dtype="float64")
        
        periods.append(t_idx)
        t_idx_series_dict[f't_idx_series_{i}'] = t_idx_series
        
    timeindex = pd.concat(list(t_idx_series_dict.values())).index
    
    return calc_years_lst, timeindex, periods
 
    

def writeParametersLogger(calc_years):
    '''
    Writes input Parameters to logger for start of calculation.
    
    Parameters
    ----------
    calc_years: list of representative years
    
    Returns
    -------
    '''
    
    info = {} #why is this dict needed? It seems, that it is used nowhere else
    now = datetime.now()
    logging.info('Started LAEND-pf at: ' + str(now))
    info['Calculation start time'] = now
    logging.info('***********Parameters used in calculation:*********')
    logging.info('Start year: ' + str(config_laend.start_year))
    info['Start year'] = config_laend.start_year
    logging.info('End year: ' + str(config_laend.end_year))
    info['End year'] = config_laend.end_year
    logging.info('Granularity: ' + str(config_laend.granularity))
    info['Granularity'] = config_laend.granularity
    logging.info(f'Years selected for calculation: {calc_years}')
    info['Years for calculation'] = calc_years
    info['Investment time steps (annual result will be multiplied with this number if aux years are used!) '] = config_laend.InvestTimeSteps
    info['Emission Constraint'] = config_laend.emission_constraint
    logging.info('Using emission constraints: ' + str(config_laend.emission_constraint))

    for i in config_laend.objective:
        logging.info(f'Objectives: {i}')
    logging.info('***********++++++++++++++++++++++++++++++*********')



def CompileScenario(filename):
    '''
    Read scenario excel and adjust scenario setting (e.g. remove deactivated technologies)

    Parameters
    ----------
    filename: pathname of scenario-excel

    Returns
    -------
    scenario: Dictionary, including dataframes containing techno-economic informations about energy system
    '''
    
    # Check, if Scenario Excel file exist?
    if not filename or not os.path.isfile(filename):
        raise FileNotFoundError(f'Excel data file {filename} not found.')
    
    logging.info("Importing data of Scenario Excel")
    
    #read scenario-excel
    xls = pd.ExcelFile(filename)
    scenario = {}
    for sheet in xls.sheet_names:
        if sheet != "INFO":
            scenario[sheet] = xls.parse(sheet)
        
    hourindex = scenario['timeseries']['hour']
    scenario['timeseries'] = scenario['timeseries'].set_index(hourindex)
    del scenario['timeseries']['hour']
    
    logging.info("Data from Scenario Excel imported.")
    
    #iterate through sheets and drop all deactivated
    #commodities, technologies, empty rows, columns etc.:
    keys_list = [key for key in scenario.keys() if key != "timeseries"]
        
    for key in keys_list:
        for num, x in scenario[key].iterrows():
            #drop empty rows:
            if pd.isna(x['label']):
                scenario[key] = scenario[key].drop(labels=num)
            #drop deactivated technologies etc.:
            elif x['active'] != 1:
                scenario[key] = scenario[key].drop(labels=num)
                
        #collect and drop irrelevant columns:
        un = [x for x in scenario[key].columns if str(x).__contains__('Unnamed') or \
                                                  str(x).__contains__('warnings') or \
                                                  str(x).__contains__('Dropdown list') or \
                                                  str(x).__contains__('comment')]
        for u in un:
            del scenario[key][u]
            
    #collect and drop empty "timeseries" columns:
    key = "timeseries"
    empty_columns = scenario[key].columns[(scenario[key].isna()).all()]
    scenario[key].drop(columns=empty_columns, inplace=True)
    
    logging.info("All irrelevant data dropped.")
    
    return scenario



def validateExcelInput(scenario):
    '''
    Validation of Excel-File respectively scenario dictionary.
    Should include more criteria, work in progress.

    Parameters
    ----------
    scenario: dictionary including all relevant information

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.
    '''

    logging.info('Validating Scenario Excel.')

    # general checks
    tabs = [*scenario]
    tabs.remove('timeseries')

    #check that no name ends in 4 digits (year gets added to the name later):
    for t in tabs:
        items = list(scenario[t]['label'])
        for item in items:
            if item[-4:].isnumeric():
                raise ValueError(
                    f'Labels cannot end with four integers as this will lead to confusion once the year of investment is added to the year. Please change {item} in sheet {t} and start again.')
    logging.info('Labels from excel input are ok. Validation continues...')

    # check for duplicate names
    for t in tabs:
        duplicated_labels = scenario[t][scenario[t].duplicated(subset=['label'], keep='last')]['label']
        if not duplicated_labels.empty:
            raise ValueError(f'Duplicate label entries in Excel input {t}: {duplicated_labels.tolist()}')
        else:
            logging.info('No duplicate names found. Validation continues...')
 

    # check that the buses used for the investments e.g. input and output buses are included in the sheet 'buses'
    buses = list(scenario['buses']['label'])

    for i in tabs:
        if i == 'buses':
            continue

        lst_buses_labels = ['to', 'from', 'bus_1', 'bus_2', 'bus', 'from1', 'to1', 'from2', 'to2']

        for bus_label in lst_buses_labels:
            try:
                used_buses = list(scenario[i][bus_label])
                bus_issues = [bus for bus in used_buses if bus not in buses]
                
                if set(used_buses).issubset(buses):
                    continue
                elif all(pd.isna(bus_issues)):
                    continue
                else:
                    raise ValueError(
                        f'Please check sheet {i}. You have listed a bus in column {bus_label} that is not included or set active in the bus list.')
            except KeyError:
                continue
    logging.info('All buses used as input or output have also been listed as a bus. Validation continues...') 

    # check that demand buses have excess
    demands = list(scenario['demand']['from'])
    bus_with_excess = list(scenario['buses'].loc[scenario['buses']['excess'] == 1]['label'])
    if not set(demands).issubset(bus_with_excess):
        raise ValueError(
            f'Please make sure that the buses listed as input for demands {demands} have excess capacity enabled in the buses excel sheet.')
    logging.info('Checked that demand inputs have excess capability enabled. Validation continues...')

    # check that fixed flows exist for renewables and demand
    sheets = ['demand', 'renewables']
    fixed_flows = list(scenario['timeseries'].columns)
    for sheet in sheets:
        df = scenario[sheet]
        if sheet == 'demand':
            for i, x in df.iterrows():
                if x['fixed-flow_periodically_variable'] == True:
                    if not any(x['label'] in fixed_flow for fixed_flow in fixed_flows):
                        raise ValueError(
                            'You have listed demand that does not have a corresponding fixed flow in sheet timeseries.')
                else:
                    if not any(x['label'] + '.fix' == fixed_flow for fixed_flow in fixed_flows):
                        raise ValueError(
                            f'You have listed demand {x["label"]} that does not have a corresponding fixed flow in sheet timeseries.')
        if sheet == 'renewables':
            for i, x in df.iterrows():
                if x['fixed'] == True:
                    if x['fixed-flow_periodically_variable'] == True:
                        if not any(x['label'] + '.fix' == fixed_flow or x['initial_existance'] == True for fixed_flow in fixed_flows):
                            raise ValueError(
                                f'You have listed a fixed flow in sheet {sheet} that does not have a corresponding fixed flow in sheet timeseries.')
                    else:
                        if not any(x['label'][:-3] +'.fix' == fixed_flow or x['initial_existance'] == True for fixed_flow in fixed_flows):
                            raise ValueError(
                                f'You have listed a fixed flow in sheet {sheet} that does not have a corresponding fixed flow in sheet timeseries.')      
    logging.info('Checked availability of fixed flows. Validation continues...')

    # check that lca names have no special characters / < >
    lca_columns = ['var_env1', 'inv1', 'var_env2']
    sheets = [*scenario]
    issue_chars = set(['/', '>', '<'])

    for sheet in sheets:
        if sheet in ['buses', 'timeseries']:
            continue

        for col in lca_columns:
            try:
                lca_items = list(scenario[sheet][col])

                for item in lca_items:
                    if pd.isna(item):
                        continue
                    if 1 in [c in item for c in issue_chars]:
                        raise ValueError(
                            f'Please check excel sheet {sheet}, column {col} and rename --{item}-- in Excel and database as the following characters cause errors: {issue_chars}')
            except KeyError:
                continue
    logging.info('Checked for characters that cause issues in LCA dataset names. Validation continues...')

    logging.info('Successfully checked the excel input table for the most common errors.')

   
    
def compileTMY(file_name):
    '''
    Takes pvgis tmy file and prepares it for later use

    Parameters
    ----------
    file_name: csv file of tmy data from pvgis

    Returns
    -------
    tmy : tmy file without leap day
    tmy_month_year : series of month and year that was chosen for tmy

    '''

    # create dictionary of years chosen for particular month
    my = pd.read_csv(file_name, skiprows=range(0, 3), nrows=12, sep=',')
    my.index = my.month
    del my['month']
    tmy_month_year = pd.Series(my['year']).to_dict()

    # get hourly data
    tmy = pd.read_csv(file_name, sep=',', skiprows=range(0, 16))
    tmy = tmy.drop(tmy.index[8760:])
    del tmy['time(UTC)']
    tmy.index = range(1, 8761)
    tmy.index.name = 'hour'

    return tmy, tmy_month_year

    
    
#further functions avialable within laend_v3.1



def createWindPowerPlantFixedFlow(scenario, my_weather, run_name, time):
    '''
    Create fixed flow for wind power plant, based on windpowerlib

    Parameters
    ----------
    scenario: Dictionary, including dataframes contain techno-economic informations about energy system
    my_weather: weather year
    run_name: name of currently running scenario.
    time: starting time of currently running scenario calculation.

    Returns
    -------
    scenario: Updated scenario dictionary
    '''

    logging.info('Creating fixed flow for wind power plants')

    # read from scenario-excel, which wind power plants shall be used within energy system
    items_dict = {}
    key = None
    for i,x in scenario["renewables"].iterrows():
        if x['label'][:4] == 'wind':
            if key != x['label'][:-3]: #efficiency check, that an already existing key is not written again
                key = x['label'][:-3]
                value = {
                    'turbine_type': x['info1'],
                    'hub_height': x['info2'],
                    'windPlantCapacity': x['info3']}
                items_dict[key] = value
    
    #Test, if scenario-excel does already contain wind speed values:
    logging.info('Test, if scenario-excel does already contain wind speed')
    
    for item in items_dict.keys():
        default_wind_speed_timeseries = pd.Series(scenario['timeseries'][item + '.fix'])  
    
        #if scenario-excel already contains wind speed values,
        #use these values for computation of wind turbine's power-output
        if all(np.isfinite(v) for v in default_wind_speed_timeseries):
            #windpowerlib.oedb as another option to get power-curves
            windPowerPlant = {
                'turbine_type': items_dict[item]['turbine_type'],
                'hub_height': items_dict[item]['hub_height']}
            windPowerPlant = WindTurbine(**windPowerPlant)
            power_curve_wind_speeds = windPowerPlant.power_curve['wind_speed']
            power_curve_values = windPowerPlant.power_curve['value']
                
            windPowerPlant.power_output = power_output.power_curve(
                wind_speed=default_wind_speed_timeseries,
                power_curve_wind_speeds=power_curve_wind_speeds,
                power_curve_values=power_curve_values)
    
        #if there is neither a pvgis weather file nor information by scenario-timeseries given, raise error
        elif my_weather == None and not all(np.isfinite(v) for v in default_wind_speed_timeseries):
            raise ValueError("No wind speeds defined! Either use a pvgis datafile or insert wind speed data to timeseries in scenario-excel")
        
        #if scenario-excel doesn't already contain wind speed values,
        #use PVGIS-file for computation of wind speed at specific hight and wind turbine's power output
        else:  
            timezone = 'Europe/Berlin'
            wind_z0_roughness = 0.1 # surface roughness length in m; bare rocks, sparely vegetated areas = 0.01; 
                            # arable land, pastures, nat. grasslands = 0.1; forest = 0.9 
                            # [https://www.researchgate.net/figure/Surface-roughness-length-m-for-each-land-use-category-in-WAsP-and-WRF-and-the_tbl1_340415172]

            my_weather = my_weather.copy()
            # convert tmy to required format
            my_weather.index = pd.to_datetime(my_weather.index, utc=True)
            my_weather.index = my_weather.index.tz_convert(
                    timezone)
            my_weather = my_weather.rename(columns={'WS10m': 'v_wind', 'SP': 'pressure', 'T2m': 'temp_air'})
            my_weather['temp_air'] += 273.15 # converts °C in K
            my_weather['z0'] = wind_z0_roughness
        
            # The columns of the DataFrame my_weather are a MultiIndex where the first level
            # contains the variable name as string (e.g. 'wind_speed') and the
            # second level contains the height as integer at which it applies
            # (e.g. 10, if it was measured at a height of 10 m). The index is a
            # DateTimeIndex.
    
            windHeightOfData = { # height in m over ground of measurement of single parameter
                                # 'dhi': 0,
                                # 'dirhi': 0,
                                'pressure': 0, # hight in which pressure was measured
                                'temp_air': 2, # height in which temperature was measured
                                'v_wind': 10, # height in which wind speed was measured
                                'Z0': 0} # hight in which roughness was measured
    
            # insert second column name for height of data 
            my_weather = my_weather[["v_wind", "temp_air", "z0", "pressure"]]
            my_weather.columns = [
                ["wind_speed", "temperature", "roughness_length", "pressure"],
                [
                    windHeightOfData["v_wind"], 
                    windHeightOfData["temp_air"],
                    windHeightOfData["Z0"],
                    windHeightOfData["pressure"],
                    ],
                ]
    
            # Specification of wind turbine. Data is provided in the
            # oedb turbine library
            windPowerPlant = {
                'turbine_type': items_dict[item]['turbine_type'],
                'hub_height': items_dict[item]['hub_height']}
            windPowerPlant = WindTurbine(**windPowerPlant)
    
            # For specification of wind turbine with your own data, see windpowerlib documentation
    
            # Use of ModelChain with defaul parameter
            mc_windPowerPlant = ModelChain(windPowerPlant).run_model(my_weather)
            # write power output time series to WindTurbine object
            windPowerPlant.power_output = mc_windPowerPlant.power_output
    
        wind_plant_fix = windPowerPlant.power_output/ 1000  # W --> kW
        # write fixed power output flow in kW to scenario timeseries
        scenario['timeseries'][item+'.fix'] = wind_plant_fix / items_dict[item]['windPlantCapacity']

        # fixed flow values could be exported to .csv-file by "wind_plant_fix.to_csv..."
        
    return scenario



def readWeightingNormalization(filename):
    '''
    Reads excel file about normalization and weighting.

    Parameters
    ----------
    filename: Filename with information about weighting and normalization. Filename is defined in config_laend.

    Returns
    -------
    weightEnvIn: Dataframe with weighting information.
    normalizationEnvIn: Dataframe with normalization information.
    '''

    logging.info('Reading environmental weighting and normalization from Excel')

    # validate filenname
    if isinstance(filename, str) == True:
        if filename[-5:] != '.xlsx':
            raise ValueError('Make sure your filename for weighting and normalization ends in .xlsx')
        elif os.path.isfile(filename) == False:
            raise FileNotFoundError(f'No file found for weighting and normalization with name {filename}')
    else:
        raise TypeError(f'{filename} is not a valid filename; cannot read weighting and normalisation')

    # Validate objective inputs
    for i in config_laend.objective:
        if not isinstance(i, str):
            raise TypeError('Please enter a valid objective')
        assert i in config_laend.system_impacts_index, f'{i} not in list of allowed objectives'

    weightEnvIn = pd.read_excel(filename, sheet_name='Weighting', index_col=0)
    normalizationEnvIn = pd.read_excel(filename, sheet_name='Normalisation', index_col=0)

    logging.info('Successfully read env weighting and normalization')

    return weightEnvIn, normalizationEnvIn



def addLCAData(scenario):
    '''
    Iterates through active elements and imports corresponding LCA data from brightway or an already existing excel-file.
    Multiplication with conversion factor to convert impacts to kW, kWh, m² etc. if necessary.

    Parameters
    ----------
    scenario: scenario dictionary, including all information about energy system.
    
    Returns
    -------
    scenario: updated scenario dictionary including LCA-data
    lcia_methods: list of applied Life Cycle Impact Assessment methods and the respective midpoint indicators
    lcia_units: associated units of the respective LCIA methods (e.g. climate change GWP100: kg CO2-eq.)
    '''

    logging.info('Getting LCA data')

    # Definition of brightway2 (LCA) project to be used as reference
    bw2.projects.set_current(config_laend.bw2_project)
    # Definition of eco database within chosen bw2 project to be used
    eco = bw2.Database(config_laend.bw2_database)
    eco_df = pd.DataFrame(eco, columns=['code', 'name', 'production amount', 'unit', 'type', 'location'])
    eco_df.set_index('name', inplace=True, drop=False)    
    #eco_dict = eco_df.to_dict('index')
    # Definition of Life Cycle Impact Assessment method (LCIA)
    lcia_methods = []
    lcia_units = []
    for m in bw2.methods:
        if m[0] == config_laend.bw2_method:
            lcia_methods.append(m)
            lcia_units.append(bw2.methods.data[m]['unit'])

    # prepare search for all relevant environmental data within scenario dict for LCA computation    
    keys1 = ['buses', 'commodity_sources', 'renewables', 'storages', 'converters_in', 'converters_out']
    keys2 = ['shortage_env', 'excess_env', 'var_env1', 'var_env2', 'inv1']

    #search relevant information within scenario excel and compute LCA by using brightway2
    for key1 in keys1:
        for i, x in scenario[key1].iterrows():
            for key2 in keys2:
                lca_score = []
                try:
                    #check, if environmental information are "empty" - if yes, write zeros representing no environmental impact
                    if x[key2] == 'empty':
                        for method in lcia_methods:
                            lca_score.append(0)
                        x[key2] = pd.Series(lca_score, index=[i[1] for i in lcia_methods], name='impact')  
                        
                    #if not "empty", test if an LCA computation is already available as excel file (importing excel is faster than bw2 computation)           
                    elif os.path.isfile(f'in/LCA/{config_laend.bw2_method}_{x[key2]}.xlsx'):
                        lca_score_df = pd.read_excel(f'in/LCA/{config_laend.bw2_method}_{x[key2]}.xlsx', sheet_name='impacts', header=0, 
                                                     index_col=0)
                        x[key2] = pd.Series(lca_score_df['impact'].tolist(), index=[i[1] for i in lcia_methods]) * x[f'{key2}_conversion']

                    #if no LCA computation is available as excel file, compute LCA using bw2 and write LCA results to an excel file    
                    else:
                        try:
                            eco_code = eco_df.at[x[key2], 'code']
                            logging.info(f'bw2 computation of {x[key2]}')
                            l_ele_result = []
                            for method in lcia_methods:
                                process = eco.get(eco_code)
                                functional_unit = {process:1}
                                lca = bw2.LCA(functional_unit, method)
                                lca.lci()
                                lca.lcia()
                                lca.score
                                l_ele_result.append(lca.score)
                        except KeyError:
                            error_label = x['label']
                            print(f'\n\n\n!!!!!!!!!!\nKeyError: Technology "{error_label}" contains LCA process "{x[key2]}" that is not contained in bw2 database "{config_laend.bw2_database}".\n!!!!!!!!!!\n\n\n')
                            sys.exit()
                        
                        #write results to excel
                        results_for_export = pd.DataFrame({'unit': lcia_units, 'impact': l_ele_result}, index=lcia_methods)
                        info_for_export = eco_df.loc[x[key2]]
                        info_for_export.name = 'info'
                        with pd.ExcelWriter(f'in/LCA/{config_laend.bw2_method}_{x[key2]}.xlsx') as writer:
                            info_for_export.to_excel(writer, sheet_name='information')
                            results_for_export.to_excel(writer, sheet_name='impacts')

                        #write results, offset against a conversion factor, to scenario dict
                        lca_score = pd.Series(l_ele_result, index=[i[1] for i in lcia_methods], name='impact')
                        x[key2] = lca_score * x[f'{key2}_conversion']
                            
                except KeyError:
                    continue
           
            scenario[key1].loc[i] = x           

    return scenario, lcia_methods, lcia_units



########################################################
#### Functions used in optimizeForObjective()
########################################################

def determineGoalForObj(objective):
    '''
    Sets ratio of financial and environmental cost

    Parameters
    ----------
    objective: optimization objective as string from config-file

    Returns
    -------
    dict containing financial goal weighting and environmental goal weighting
    '''

    # Validate function inputs
    if not isinstance(objective, str):
        raise TypeError('Please enter a valid objective as a string')
    assert objective in config_laend.system_impacts_index, f'{objective} not in list of allowed objectives'

    logging.info('*********** Determining Weight of Costs and Environmental Impacts *************')

    if objective == 'Costs':
        goal_financial = 1  # Costs (economical)
        goal_environmental = 0

    elif objective == 'EnvCosts':
        goal_financial = config_laend.weight_cost_to_env
        goal_environmental = 1 - goal_financial
    
    elif objective == 'Equilibrium':
        goal_financial = config_laend.weight_cost_to_env_equilibrium
        goal_environmental = 1 - goal_financial

    else:
        goal_financial = 0
        goal_environmental = 1

    logging.info(f'Successfully determined goal weighting of costs and environmental impacts:\n\
                 goal weighting costs: {goal_financial}, goal weighting environmental: {goal_environmental}')

    return {"costs": goal_financial, "env": goal_environmental}



def determineCfactorForSolver(objective):
    '''
    corrects value for solver, if normalization is not per person

    Parameters
    ----------
    objective: optimization objective as string from config-file

    Returns
    -------
    c_factor: correction factor as number (integer)
    '''

    logging.info('Determining correction factor for solver')
    c_factor = 1
    
    if 'human toxicity' in objective:
        c_factor = 1e6
    elif 'minerals' in objective:
        c_factor = 1e3
    elif objective in {'JRCII', 'EnvCosts', 'Equilibrium'}:
        c_factor = 1e17
    elif 'climate' in objective and any(keyword in objective for keyword in ['biogenic', 'land use']):
        c_factor = 100

    logging.info(f'Correction factor for solver: {c_factor}')

    return c_factor



def adaptForObjective(scenario, objective, weightEnv, normalizationEnv, goals, c_factor, calc_years):
    '''
    Adapts all (env & financial) factors in scenario to the optimization objective

    Parameters
    ----------
    scenario: scenario dictionary with all relevant data
    objective: objective as string, initially taken from config file
    weightEnv:
    normalizationEnv:
    goals: ratio of financial and environmental cost, based on objective
    c_factor: correction factor for solver if LCA values are very small
    calc_years: list of representative years (=periods) that are optimized

    Returns
    -------
    scenario_obj: adjusted copy of scenario dict containing data summarized regarding specific objective
    '''
  
    logging.info('Adapting factors to the optimization objective')
    
    if config_laend.number_of_time_steps != None:
        tstp = config_laend.number_of_time_steps
    else:
        tstp = 8760
        
    cy_list_adapted = []
    for calc_year in calc_years:
        cy_list_adapted.append(str(calc_year)[-2:])
    
    #deepcopy scenario so that original dict stays the same
    scenario_obj = copy.deepcopy(scenario)
    #prepare search and adaption for all relevant data within scenario dict   
    keys1 = ['buses', 'commodity_sources', 'renewables', 'storages', 'converters_in', 'converters_out']
    keys_env = ['shortage_env', 'excess_env', 'var_env1', 'var_env2', 'inv1']
    keys_fin = ['shortage_costs', 'excess_costs', 'variable_costs', 'om', 'invest', 'variable_input_costs', 'variable_output_costs', 'var_from1_costs', 'var_to1_costs', 'var_from2_costs']
    pot_multi_criteria = ['EnvCosts', 'JRCII', 'Equilibrium']

    #search relevant information within scenario excel and compute adaption:
    for key1 in keys1:

        if len(scenario[key1]) == 0:
            continue

        else:
            # iterate through rows of scenario
            for x in scenario_obj[key1].index:
                row = scenario_obj[key1].loc[x]
                logging.info('Adapting environmental impacts of ' + row['label'] + ' for weight and normalisation')
 
                for key_env in keys_env:
                    try:
                        if isinstance(row[key_env], pd.Series):
                            if objective == 'Costs':
                                new = 0
                            elif objective in pot_multi_criteria:
                                #multiply weighting factors with normalized LCA and sum up
                                #think about division by lifetime
                                new_list = []
                                for index in row[key_env].index:
                                    if row[key_env][index] == 0:
                                        new_list.append(0)
                                    else:
                                        new_list.append(row[key_env][index] * weightEnv.loc[index, objective] / normalizationEnv.loc[index, objective])
                                
                                new = sum(new_list)
                            
                            else:
                                new = row[key_env][objective]
                                
                            logging.info(row['label'] + ': Environmental impacts ' + key_env + ' adapted to optimization objective\nImpact w/o goal weighting and correction factor: ' + str(new))
                            
                            row[key_env] = new * goals['env'] * c_factor

                    except KeyError:
                        continue

                for key_fin in keys_fin:
                    try:
                        test = row[key_fin] #serves only as a test, if key_fin is available within row - else -> except -> continue!
                        
                        #check, if there are periodically variable EXCESS costs (e.g. emission costs) and adapt for objective
                        if row[key_fin] == 'c_avar':
                            for y in cy_list_adapted:
                                if objective == 'Costs':
                                    new = row[f'c_avar_{y}']
                                elif objective in pot_multi_criteria:
                                    new = row[f'c_avar_{y}'] / config_laend.normalization_cost_gdp
                                else:
                                    new = 0
                            
                                row[f'c_avar_{y}'] = new * goals['costs'] * c_factor
                                
                        #info: if there are completely variable EXCESS costs (not only periodically, but also within a period), they are
                        #      processed during timeseries processing (see few lines below)
                                
                        #common proceeding for everything that is not periodically variable EXCESS costs     
                        else:
                            if objective == 'Costs':
                                if not key_fin == 'invest':
                                    new = row[key_fin]
                                elif key_fin == 'invest':
                                    new = economics.annuity(row[key_fin], row['lifetime'], config_laend.InvestWacc) * row['lifetime']
                            elif objective in pot_multi_criteria:
                                if not key_fin == 'invest':
                                    new = row[key_fin] / config_laend.normalization_cost_gdp
                                elif key_fin == 'invest':
                                    new = economics.annuity(row[key_fin], row['lifetime'], config_laend.InvestWacc) * row['lifetime'] / config_laend.normalization_cost_gdp
                            else:
                                new = 0
                        
                            row[key_fin] = new * goals['costs'] * c_factor
                        
                    except KeyError:
                        continue
                    
                scenario_obj[key1].loc[x] = row

    #search relevant infromation within timeseries (timevariable_costs or timevariable_env) and compute adaption
    for col in scenario_obj['timeseries'].columns.values:
        #adapt financial data
        if col.split('.')[1] == 'timevariable_costs':
            if objective == 'Costs':
                new = pd.Series(scenario_obj['timeseries'][col])
            elif objective in pot_multi_criteria:
                new = pd.Series(scenario_obj['timeseries'][col]) / config_laend.normalization_cost_gdp
            else:
                new = pd.Series([0] * tstp)
            
            scenario_obj['timeseries'][col] = new * goals['costs'] * c_factor

        #adapt environmental data
#        elif col.split('.')[1] == 'timevariable_env':
#            if objective == 'Costs':
#                new = [0] * 8760
#            elif objective in pot_multi_criteria:
#                multiply series with weighting factor and normalization factor of respective impact category (probably carbon footprint)               


#if myopic, lifetime and om cost should be considered in annuity computation

    return scenario_obj



def createOemofNodes(scenario_obj, calc_years):
    '''
    Creates nodes (oemof objects) from scenario_obj dictionary

    Parameters
    ----------
    scenario_obj: scenario dictionary containing all relevant information adjusted to objective
    calc_years: list of representative years (=periods) that are optimized
    
    Returns
    -------
    nodes: list of created nodes that will be added to energysystem model (es)
    '''
    
    logging.info('Creating oemof objects')
    
    if not scenario_obj:
        raise ValueError('No scenario_obj provided.')
    
    if config_laend.number_of_time_steps != None:
        tstp = config_laend.number_of_time_steps
    else:
        tstp = 8760
    
    nodes = []    
    busd = {}
    cy_list_adapted = []
    for calc_year in calc_years:
        cy_list_adapted.append(str(calc_year)[-2:])
    

    ####Create buses    
    for i, x in scenario_obj['buses'].iterrows():
        bus = solph.buses.Bus(label=x['label'])
        nodes.append(bus)
        
        busd[x['label']] = bus

        if x['excess']:
            #check, if periodically variable excess_costs
            if x['excess_costs'] == 'c_avar':
                #collect periodically costs
                var_exc_costs = []
                for y in cy_list_adapted:
                    #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                    if y != cy_list_adapted[-1]:
                        var_exc_costs.extend([(x[f'c_avar_{y}'] + x['excess_env'])* config_laend.aux_year_steps] * tstp)
                    if y == cy_list_adapted[-1]:
                        var_exc_costs.extend([x[f'c_avar_{y}'] + x['excess_env']] * tstp)                        
            
            #check, if variable costs (not only periodically, but also within a period)
            elif x['excess_costs'] == 'c_var':           
                var_exc_costs = []
                for y in cy_list_adapted:
                    #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                    if y !=  cy_list_adapted[-1]:
                        var_exc_costs.extend(((scenario_obj['timeseries'][f'{x["label"]}_{y}.timevariable_costs'] + x['excess_env']) * config_laend.aux_year_steps).tolist())
                    if y == cy_list_adapted[-1]:
                        var_exc_costs.extend((scenario_obj['timeseries'][f'{x["label"]}_{y}.timevariable_costs'] + x['excess_env']).tolist())
            
            #if one and the same excess costs are valid for all periods
            else:
                var_exc_costs = []
                #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                var_exc_costs.extend([(x['excess_costs'] + x['excess_env']) * config_laend.aux_year_steps] * tstp * (len(cy_list_adapted)-1))
                var_exc_costs.extend([x['excess_costs'] + x['excess_env']] * tstp)
                
            #generate oemof object for excess (sink)
            bus_excess = solph.components.Sink(
                label=x['label'] + '_excess',
                inputs={busd[x['label']]: solph.flows.Flow(
                    variable_costs=var_exc_costs)})
            nodes.append(bus_excess)
        
            
        if x['shortage']:
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            var_shor_costs = []
            var_shor_costs.extend([(x['shortage_costs'] + x['shortage_env']) * config_laend.aux_year_steps] * tstp * (len(cy_list_adapted)-1))
            var_shor_costs.extend([x['shortage_costs'] + x['shortage_env']] * tstp)
            
            #generate oemof object for shortage (source)
            bus_shortage = solph.components.Source(
                label=x['label'] + '_shortage',
                outputs={busd[x['label']]: solph.flows.Flow(
                    variable_costs=var_shor_costs)})
            nodes.append(bus_shortage)
            
        logging.debug(x['label'] + ' (bus) created')

    
    ####Create Sources
    #collect and cluster all sources
    sources_dict = {}
    for i, x in scenario_obj['commodity_sources'].iterrows():
        if x['label'][-2:].isdigit():
            x_label = x['label'][:-3]
            if x_label in sources_dict:
                sources_dict[x_label][f'{x["label"][-2:]}'] = x #writes each year separately to sources_dict[source]
            else:
                sources_dict[x_label] = {} #if sources_dict[source] is not yet exisiting, generate!
                sources_dict[x_label][f'{x["label"][-2:]}'] = x
        else:
            sources_dict[x['label']] = {}
            sources_dict[x['label']]['allperiods'] = x
    
    #go through clustered sources and write timeseries for multi period optimization
    for src in sources_dict.keys():
        #check, if for all periods (=calculation years) respective information is available
        if not all(key == 'allperiods' for key in sources_dict[src].keys()):
            for y in cy_list_adapted:
                if not y in sources_dict[src].keys():
                    raise ValueError(f'\n\n\n!!!!!!!!!!\nPeriod 20{y} is missing in {src}.\n!!!!!!!!!!\n\n\n')

            #shorten sources[src] dict to the actual amount of periods that shall be optimized
            sources_dict[src] = {year: value for year, value in sources_dict[src].items() if year in cy_list_adapted}

        #if there are one and the same source information valid for all periods
        if all(key == 'allperiods' for key in sources_dict[src].keys()):
            timeseries_list = []
            cy = sources_dict[src]['allperiods']
            if cy['timevariable_costs'] == True:
                if not any(col.split(".")[0] == cy['label'] for col in scenario_obj['timeseries'].columns.values):
                    raise ValueError("Please make sure labels in timeseries are the same as in commodity_sources")
                for col in scenario_obj['timeseries'].columns.values:
                    if col.split('.')[0] == cy['label']:
                        #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                        timeseries_list.extend([(scenario_obj['timeseries'][col] + cy['var_env1']) * config_laend.aux_year_steps] * (len(calc_years)-1))
                        timeseries_list.extend([scenario_obj['timeseries'][col] + cy['var_env1']])
            else:                   
                #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                timeseries_list.extend([(cy['variable_costs'] + cy['var_env1']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
                timeseries_list.extend([cy['variable_costs'] + cy['var_env1']] * tstp)
        
        #if there are individual input parameters for each period
        else:
            #collect information for timeseries and generate timeseries for multi period optimization
            timeseries_list = []
            for calc_year in sources_dict[src]:
                cy = sources_dict[src][calc_year]
                if cy['timevariable_costs'] == True:
                    if not any(col.split(".")[0] == cy['label'] for col in scenario_obj['timeseries'].columns.values):
                        raise ValueError("Please make sure labels in timeseries are the same as in commodity_sources")
                    for col in scenario_obj['timeseries'].columns.values:
                        if col.split('.')[0] == cy['label']:
                            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                            if not str(calc_year) == cy_list_adapted[-1]:
                                timeseries_list.extend((scenario_obj['timeseries'][col] + cy['var_env1']) * config_laend.aux_year_steps)
                            elif str(calc_year) == cy_list_adapted[-1]:
                                timeseries_list.extend(scenario_obj['timeseries'][col] + cy['var_env1'])
                                break
                                
                else:
                    #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                    if not str(calc_year) == cy_list_adapted[-1]:
                        timeseries_list.extend([(cy['variable_costs'] + cy['var_env1']) * config_laend.aux_year_steps] * tstp)
                    elif str(calc_year) == cy_list_adapted[-1]:
                        timeseries_list.extend([cy['variable_costs'] + cy['var_env1']] * tstp)
                
        sources_dict[src]['timeseries'] = timeseries_list #timeseries_list contains financial and environmental data
        
        to = cy['to']
        source = solph.components.Source(
            label=src,
            outputs={busd[to]: solph.flows.Flow(
                variable_costs=timeseries_list)})
        nodes.append(source)
        
        logging.debug(f'{src} (source) created.')


    ####Create Sinks
    for i, x in scenario_obj['demand'].iterrows():
        #check, if demands are different for the respective years.
        #if no, write defined demand for all periods
        timeseries_list = []
        if x['fixed-flow_periodically_variable'] == False:
            for y in cy_list_adapted:
                timeseries_list.extend(scenario_obj['timeseries'][x['label'] + '.fix'])
        #if the demand is varying for the respective years, generate timeseries considering this characteristic
        else:
            for y in cy_list_adapted:
                if not any(col == x['label'] + '_' + y + '.fix' for col in scenario_obj['timeseries'].columns.values):
                    raise ValueError("Please make sure 'timeseries' is containing demand information for all periods necessary")
                timeseries_list.extend(scenario_obj['timeseries'][x['label'] + '_' + y + '.fix'])

        if x['DSM'] == False:
            sink = solph.components.Sink(
                label=x['label'],
                inputs={busd[x['from']]: solph.flows.Flow(
                    fix=timeseries_list,
                    nominal_value=x['nominal value'])})
            nodes.append(sink)
            
        elif x['DSM'] != False:
            #check, if absolute or relative upshifting capacity is given,
            #if relative value is given, derive upshifting capacity from given load profile
            if '%' in str(x['capacity_up']):
                cap_up = list(scenario_obj['timeseries'][f'{x["label"]}.fix'] * int(x['capacity_up'][:-1])/100)
                
            else:
                cap_up = [x['capacity_up']] * 8760
            
            #check, if absolute or relative downshifting capacity is given,
            #if relative value is given, derive downshifting capacity from given load profile
            if '%' in str(x['capacity_down']):
                cap_down = list(scenario_obj['timeseries'][f'{x["label"]}.fix'] * int(x['capacity_down'][:-1])/100)
            else:
                cap_down = [x['capacity_down']] * 8760
            
            #if there is a regularly occuring time without DSM potential within a day (such as baseload 
            #during night), search for them in DSM timeseries (cap_down, cap_up), and set DSM potential to 0
            if x['unflex_from'] != False and x['unflex_to'] != False:
                for day_counter in range(1, 366, 1):
                    #consider time change from winter to summer time, as through that demand curve is shifted by minus 1 hour
                    if ((x['day_timechange_march'] + 31 + 28) <= day_counter < (x['day_timechange_october'] + 31*5 + 30*3 + 28)
                    and x['day_timechange_march'] != 0 and x['day_timechange_october'] != 0):
                        time_correction = 1
                    else:
                        time_correction = 0
                        
                    #combine range from start value to midnight with range from midnight
                    #to end value
                    if x['unflex_from'] > x['unflex_to']:
                        unflex_range = list(range(int(x['unflex_from']), 24 + 1, 1)) + \
                            list(range(1, int(x['unflex_to']) + 1, 1))
                    #or combine range from start value to end value, if both are on the same day
                    else:
                        unflex_range = list(range(int(x['unflex_from']), int(x['unflex_to']) + 1, 1))
                    #define timedependent inflexibility
                    for unflex in unflex_range:
                        day_time = (day_counter - 1) * 24 + (unflex - 1 - time_correction)
                        cap_up[day_time] = 0
                        cap_down[day_time] = 0

            #if there is no DSM potential on weekends (saturday +  sunday), search in DSM timeseries
            #(cap_down, cap_up) for saturdays and sundays, and set DSM potential to 0
            if x['first_sunday'] != False:
                for day_counter in range(int(x['first_sunday']), 367, 7):
                    #Sundays
                    if day_counter < 365:
                        for day_time in range((day_counter - 1) * 24, day_counter * 24, 1):
                            cap_up[day_time] = 0
                            cap_down[day_time] = 0
                    #Saturdays:
                    if day_counter - 1 > 0:
                        for day_time in range((day_counter - 2) * 24, (day_counter - 1) * 24, 1):
                            cap_up[day_time] = 0
                            cap_down[day_time] = 0                            
            
            #write oemof node
            sink_dsm = solph.components.experimental.SinkDSM(
                label=x['label'],
                inputs={busd[x['from']]: solph.flows.Flow()},
                demand=timeseries_list,
                max_demand=1,
                shed_eligibility=False,
                capacity_up=cap_up,
                capacity_down=cap_down,
                max_capacity_down=1,
                max_capacity_up=1,
                approach=x['DSM'],
                shift_interval=int(x['delay_time']) if x['DSM']=='oemof' else None,
                delay_time=int(x['delay_time']) if x['DSM'] in ('DIW', 'DLR') else None,
                shift_time=int(x['shift_time'])/2 if x['DSM']=='DLR' else None,
                )
            nodes.append(sink_dsm)
        
        logging.debug(f'{x["label"]} (sink) created.')


    ####Create Renewables
    #collect and cluster all renewables
    renewables_dict = {}
    for i, x in scenario_obj['renewables'].iterrows():
        if x['initially_installed_capacity'] > 0:
            x_label = x['label'][:-9]
            if x_label in renewables_dict:
                renewables_dict[x_label]['existing'] = x
            else:
                renewables_dict[x_label] = {}
                renewables_dict[x_label]['existing'] = x         
        else:
            x_label = x['label'][:-3]
            if x_label in renewables_dict:
                renewables_dict[x_label][f'{x["label"][-2:]}'] = x #writes each year separately to renewables_dict[rnw]
            else:
                renewables_dict[x_label] = {} #if renewables_dict[rnw] is not yet exisiting, generate!
                renewables_dict[x_label][f'{x["label"][-2:]}'] = x
    
    #go through clustered renewables and write timeseries for multi period optimization
    for rnw in renewables_dict.keys():
        #check, if for all periods (=calculation years) respective information is available
        for calc_year in cy_list_adapted:
            if not calc_year in renewables_dict[rnw].keys():
                raise ValueError(f'\n\n\n!!!!!!!!!!\nPeriod {calc_year} is missing for {rnw}.\n!!!!!!!!!!\n\n\n')
        
        #shorten renewables[rnw] dict to the actual amount of periods that shall be optimized
        renewables_dict[rnw] = {year: value for year, value in renewables_dict[rnw].items() if year in cy_list_adapted or year == 'existing'}

        #collect information for fixed (weather) timeseries and generate timeseries for multi period optimization
        timeseries_list = []
        if renewables_dict[rnw][cy_list_adapted[0]]['fixed'] == True:
            if renewables_dict[rnw][cy_list_adapted[0]]['fixed-flow_periodically_variable'] == True:
                for calc_year in cy_list_adapted:
                    cy = renewables_dict[rnw][calc_year]
                    if not any(col.split(".")[0] == cy['label'] for col in scenario_obj['timeseries'].columns.values):
                        raise ValueError("Please make sure labels in timeseries are the same as in renewables")
                    for col in scenario_obj['timeseries'].columns.values:
                        if col.split('.')[0] == cy['label']:
                            timeseries_list.extend(scenario_obj['timeseries'][col])
            else:
                for calc_year in cy_list_adapted:
                    cy = renewables_dict[rnw][calc_year]
                    timeseries_list.extend(scenario_obj['timeseries'][cy['label'][:-3] + '.fix'])
                
        renewables_dict[rnw]['timeseries'] = timeseries_list #timeseries_list contains fixed flows (commonly based on weather data)

        #collect individual information for each period and generate lists for multi period optimization
        rnw_maximum_list = [renewables_dict[rnw][y]['max_capacity_invest'] for y in cy_list_adapted]
        rnw_ep_cost_list = [(renewables_dict[rnw][y]['invest'] + renewables_dict[rnw][y]['inv1']) * tstp/8760 for y in cy_list_adapted]
        
        #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
        rnw_varcosts_list = []
        for y in cy_list_adapted:
            if not y == cy_list_adapted[-1]:
                rnw_varcosts_list.extend([(renewables_dict[rnw][y]['variable_costs'] + renewables_dict[rnw][y]['var_env1']) * config_laend.aux_year_steps] * tstp)
            elif y == cy_list_adapted[-1]:
                rnw_varcosts_list.extend([renewables_dict[rnw][y]['variable_costs'] + renewables_dict[rnw][y]['var_env1']] * tstp)
        
        #generate oemof object for periodically individual renewable investments
        to = cy['to']
        renewable = solph.components.Source(
            label=rnw,
            outputs={busd[to]: solph.flows.Flow(
                investment=solph.Investment(
                    maximum=rnw_maximum_list,
                    minimum=0,
                    overall_maximum=cy['max_total_capacity'],
                    ep_costs=rnw_ep_cost_list, #already contains financial and environmental data
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760),
                variable_costs=rnw_varcosts_list,
                fix=timeseries_list)})
        nodes.append(renewable)
        
        logging.debug(f'{rnw} (renewable) created.')
        
        #check, if there are already existing renewable technologies and generate respective oemof object         
        if any(key == 'existing' for key in renewables_dict[rnw].keys()):
            
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            rnw_varcosts_list = []
            rnw_varcosts_list.extend([(renewables_dict[rnw]['existing']['variable_costs'] + renewables_dict[rnw]['existing']['var_env1']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            rnw_varcosts_list.extend([renewables_dict[rnw]['existing']['variable_costs'] + renewables_dict[rnw]['existing']['var_env1']] * tstp)

            #generate respective oemof object
            renewable_e = solph.components.Source(
                label=rnw+'_existing',
                outputs={busd[to]: solph.flows.Flow(
                    investment=solph.Investment(
                        maximum=0, #by maximum=0, no future investments are allowed
                        ep_costs=0,
                        existing=renewables_dict[rnw]['existing']['initially_installed_capacity'],
                        lifetime=int(renewables_dict[rnw]['existing']['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                        age=0,
                        interest_rate=0,
                        fixed_costs=renewables_dict[rnw]['existing']['om'] * tstp/8760),
                    variable_costs=rnw_varcosts_list,
                    fix=timeseries_list)})
            nodes.append(renewable_e)
            
            logging.debug(f'{rnw}_existing (renewable) created.')


    ####Create Storages
    #collect and cluster all storage technologies
    storages_dict = {}
    for i, x in scenario_obj['storages'].iterrows():
        if x['label'][-2:].isdigit():
            x_label = x['label'][:-3]
            if x_label in storages_dict:
                storages_dict[x_label][f'{x["label"][-2:]}'] = x #writes each year separately to storages_dict[stor]
            else:
                storages_dict[x_label] = {} #if storages_dict[stor] is not yet exisiting, generate!
                storages_dict[x_label][f'{x["label"][-2:]}'] = x
        elif x['initially_installed_capacity'] > 0:
            x_label = x['label'][:-9]
            if x_label in storages_dict:
                storages_dict[x_label]['existing'] = x
            else:
                storages_dict[x_label] = {}
                storages_dict[x_label]['existing'] = x         
        else:
            if x['label'] in storages_dict:
                storages_dict[x['label']]['allperiods'] = x
            else:
                storages_dict[x['label']] = {}
                storages_dict[x['label']]['allperiods'] = x

    #go through clustered storages and create oemof nodes for multi period optimization
    for stor in storages_dict.keys():
        #check, if for all periods (=calculation years) respective information is available
        if not all(key == 'allperiods' or key == 'existing' for key in storages_dict[stor].keys()):
            for calc_year in cy_list_adapted:
                if not calc_year in storages_dict[stor].keys():
                    raise ValueError(f'\n\n\n!!!!!!!!!!\nPeriod {calc_year} is missing for {stor}.\n!!!!!!!!!!\n\n\n')
            #shorten storages[stor] dict to the actual amount of periods thatt shall be optimized
            storages_dict[stor] = {year: value for year, value in storages_dict[stor].items() if year in cy_list_adapted or year == 'existing'}
      
        #if there are storage information valid for all periods
        if any(key == 'allperiods' for key in storages_dict[stor].keys()):
            cy = storages_dict[stor]['allperiods']
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            stor_varincosts_list = []
            stor_varoutcosts_list = []
            stor_varincosts_list.extend([(cy['variable_input_costs']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            stor_varoutcosts_list.extend([(cy['variable_output_costs']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            stor_varincosts_list.extend([cy['variable_input_costs']] * tstp)
            stor_varoutcosts_list.extend([cy['variable_output_costs']] * tstp)
                           
            #generate oemof object for storage investment, valid for all periods
            to = cy['bus']
            storage = solph.components.GenericStorage(
                label=stor,
                inputs={busd[to]: solph.flows.Flow(
                    variable_costs=stor_varincosts_list,
                    investment=solph.Investment(
                        ep_costs=0,
                        lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                        interest_rate=0
                        ))},
                outputs={busd[to]: solph.flows.Flow(
                    variable_costs=stor_varoutcosts_list,
                    investment=solph.Investment(
                        ep_costs=0,
                        lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                        interest_rate=0
                        ))},
                loss_rate=cy['capacity loss'],
                initial_storage_level=None,
                balanced=bool(cy['balanced']),
                max_storage_level=cy["capacity max"],
                min_storage_level=cy["capacity min"],
                invest_relation_input_capacity=cy['invest_relation_input_capacity'],
                invest_relation_output_capacity=cy['invest_relation_output_capacity'],
                inflow_conversion_factor=cy["efficiency inflow"],
                outflow_conversion_factor=cy["efficiency outflow"],
                investment=solph.Investment(
                    maximum=cy['max_capacity_invest'],
                    minimum=0,
                    overall_maximum=cy['max_total_capacity'],
                    ep_costs=(cy['invest'] + cy['inv1']) * tstp/8760,
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))
            nodes.append(storage)
            
            logging.debug(f'{stor} (storage) created.')

        #if storage information are NOT valid for all periods, BUT for each individual period    
        elif not all(key == 'existing' for key in storages_dict[stor].keys()):
            #collect individual information for each period and generate lists for multi period optimization
            stor_maximum_list = [storages_dict[stor][y]['max_capacity_invest'] for y in cy_list_adapted]
            stor_ep_cost_list = [(storages_dict[stor][y]['invest'] + storages_dict[stor][y]['inv1']) * tstp/8760 for y in cy_list_adapted]
            stor_varincosts_list = []
            stor_varoutcosts_list = []
            for y in cy_list_adapted:
                #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
                if not y == cy_list_adapted[-1]:
                    stor_varincosts_list.extend([(storages_dict[stor][y]['variable_input_costs']) * config_laend.aux_year_steps] * tstp)
                    stor_varoutcosts_list.extend([(storages_dict[stor][y]['variable_output_costs']) * config_laend.aux_year_steps] * tstp)
                elif y == cy_list_adapted[-1]:                
                    stor_varincosts_list.extend([storages_dict[stor][y]['variable_input_costs']] * tstp)
                    stor_varoutcosts_list.extend([storages_dict[stor][y]['variable_output_costs']] * tstp)
            
            cy = storages_dict[stor][calc_year]        
            to = cy['bus']
            storage = solph.components.GenericStorage(
                label=stor,
                inputs={busd[to]: solph.flows.Flow(
                    variable_costs=stor_varincosts_list,
                    investment=solph.Investment(
                        ep_costs=0,
                        lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                        interest_rate=0
                        ))},
                outputs={busd[to]: solph.flows.Flow(
                    variable_costs=stor_varoutcosts_list,
                    investment=solph.Investment(
                        ep_costs=0,
                        lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                        interest_rate=0
                        ))},
                loss_rate=cy['capacity loss'],
                initial_storage_level=None,
                balanced=bool(cy['balanced']),
                max_storage_level=cy["capacity max"],
                min_storage_level=cy["capacity min"],
                invest_relation_input_capacity=cy['invest_relation_input_capacity'],
                invest_relation_output_capacity=cy['invest_relation_output_capacity'],
                inflow_conversion_factor=cy["efficiency inflow"],
                outflow_conversion_factor=cy["efficiency outflow"],
                investment=solph.Investment(
                    maximum=stor_maximum_list,
                    minimum=0,
                    overall_maximum=cy['max_total_capacity'],
                    ep_costs=stor_ep_cost_list,
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))
            nodes.append(storage)
            
            logging.debug(f'{stor} (storage) created.')

        #if there is an already existing storage   
        if any(key == 'existing' for key in storages_dict[stor].keys()):
            cy = storages_dict[stor]['existing']
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            stor_varincosts_list = []
            stor_varoutcosts_list = []
            stor_varincosts_list.extend([(cy['variable_input_costs']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            stor_varoutcosts_list.extend([(cy['variable_output_costs']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            stor_varincosts_list.extend([cy['variable_input_costs']] * tstp)
            stor_varoutcosts_list.extend([cy['variable_output_costs']] * tstp)
            
            #generate oemof object for storage investment, valid for existing storage            
            to = cy['bus']
            storage = solph.components.GenericStorage(
                label=stor+'_existing',
                inputs={busd[to]: solph.flows.Flow(
                    variable_costs=stor_varincosts_list,
                    investment=solph.Investment(
                        ep_costs=0,
                        lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                        interest_rate=0
                        ))},
                outputs={busd[to]: solph.flows.Flow(
                    variable_costs=stor_varoutcosts_list,
                    investment=solph.Investment(
                        ep_costs=0,
                        lifetime=cy['lifetime'], #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                        interest_rate=0
                        ))},
                loss_rate=cy['capacity loss'],
                initial_storage_level=None,
                balanced=bool(cy['balanced']),
                max_storage_level=cy["capacity max"],
                min_storage_level=cy["capacity min"],
                invest_relation_input_capacity=cy['invest_relation_input_capacity'],
                invest_relation_output_capacity=cy['invest_relation_output_capacity'],
                inflow_conversion_factor=cy["efficiency inflow"],
                outflow_conversion_factor=cy["efficiency outflow"],
                investment=solph.Investment(
                    maximum=0,
                    ep_costs=0,
                    existing=cy['initially_installed_capacity'],
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    age=0,
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))
            nodes.append(storage)
            
            logging.debug(f'{stor}_existing (storage) created.')
        

    ####Create converters_in
    #collect and cluster all converter_in technologies
    converters_in_dict = {}
    for i, x in scenario_obj['converters_in'].iterrows():
        if x['label'][-2:].isdigit():
            x_label = x['label'][:-3]
            if x_label in converters_in_dict:
                converters_in_dict[x_label][f'{x["label"][-2:]}'] = x #writes each year separately to converters_in_dict[convin]
            else:
                converters_in_dict[x_label] = {} #if converters_in_dict[convin] is not yet exisiting, generate!
                converters_in_dict[x_label][f'{x["label"][-2:]}'] = x
        elif x['initially_installed_capacity'] > 0:
            x_label = x['label'][:-9]
            if x_label in converters_in_dict:
                converters_in_dict[x_label]['existing'] = x
            else:
                converters_in_dict[x_label] = {}
                converters_in_dict[x_label]['existing'] = x         
        else:
            if x['label'] in converters_in_dict:
                converters_in_dict[x['label']]['allperiods'] = x
            else:
                converters_in_dict[x['label']] = {}
                converters_in_dict[x['label']]['allperiods'] = x
    
    #go through clustered converters_in and create oemof nodes for multi period optimization
    for convin in converters_in_dict.keys():
        #check, if for all periods (=calculation years) respective information is available
        if not all(key == 'allperiods' or key == 'existing' for key in converters_in_dict[convin].keys()):
            for calc_year in cy_list_adapted:
                if not calc_year in converters_in_dict[convin].keys():
                    raise ValueError(f'\n\n\n!!!!!!!!!!\nPeriod {calc_year} is missing for {convin}.\n!!!!!!!!!!\n\n\n')
            #shorten storages[stor] dict to the actual amount of periods thatt shall be optimized
            converters_in_dict[convin] = {year: value for year, value in converters_in_dict[convin].items() if year in cy_list_adapted or year == 'existing'}

        #if there are converters_in information valid for all periods
        if any(key == 'allperiods' for key in converters_in_dict[convin].keys()):
            cy = converters_in_dict[convin]['allperiods']
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            convin_varfrom1costs_list = []
            convin_varto1costs_list = []
            convin_varto2costs_list = []
            convin_varfrom1costs_list.extend([(cy['var_from1_costs']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            convin_varto1costs_list.extend([(cy['var_to1_costs'] + cy['var_env1']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            convin_varfrom1costs_list.extend([cy['var_from1_costs']] * tstp)
            convin_varto1costs_list.extend([cy['var_to1_costs'] + cy['var_env1']] * tstp)
            try:
                convin_varto2costs_list.extend([cy['var_env2'] * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
                convin_varto2costs_list.extend([cy['var_env2']] * tstp)
            except:
                proxy = 'nothing will happen, proxy-variable is only needed for "except" task'

            #generate oemof object for converter_in investment, valid for all periods
            from1 = cy['from1']
            to1 = cy['to1']
            to2 = cy['to2']
            
            input_args = {busd[from1]: solph.flows.Flow(
                variable_costs=convin_varfrom1costs_list,
                investment=solph.Investment(
                    maximum=cy['max_capacity_invest'],
                    minimum=0,
                    overall_maximum=cy['max_total_capacity'],
                    ep_costs=(cy['invest'] + cy['inv1']) * tstp/8760,
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))}
            output_args = {busd[to1]: solph.flows.Flow(
                variable_costs=convin_varto1costs_list)}
            conversion_facts = {busd[from1]: cy['conversion_factor_f1'],
                                busd[to1]: cy['conversion_factor_t1']}
            try:
                output_args[busd[to2]] = solph.flows.Flow(
                    variable_costs=convin_varto2costs_list)
                conversion_facts[busd[to2]] = cy['conversion_factor_t2']
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
                
            nodes.append(solph.components.Converter(
                label=convin,
                inputs=input_args,
                outputs=output_args,
                conversion_factors=conversion_facts)) 
            logging.debug(f'{convin} (Converter_in) created.')

        #if converters_in information are NOT valid for all periods, BUT for each individual period
        elif not all(key == 'existing' for key in converters_in_dict[convin].keys()):
            #collect individual information for each period and generate lists for multi period optimization
            convin_maximum_list = [converters_in_dict[convin][y]['max_capacity_invest'] for y in cy_list_adapted]
            convin_ep_cost_list = [(converters_in_dict[convin][y]['invest'] + converters_in_dict[convin][y]['inv1']) * tstp/8760 for y in cy_list_adapted]
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            convin_varfrom1costs_list = []
            convin_varto1costs_list = []
            convin_varto2costs_list = []
            for y in cy_list_adapted:
                if y != cy_list_adapted[-1]:
                    convin_varfrom1costs_list.extend([(converters_in_dict[convin][y]['var_from1_costs']) * config_laend.aux_year_steps] * tstp)
                    convin_varto1costs_list.extend([(converters_in_dict[convin][y]['var_to1_costs'] + converters_in_dict[convin][y]['var_env1']) * config_laend.aux_year_steps] * tstp)
                elif y == cy_list_adapted[-1]:
                    convin_varfrom1costs_list.extend([converters_in_dict[convin][y]['var_from1_costs']] * tstp)
                    convin_varto1costs_list.extend([converters_in_dict[convin][y]['var_to1_costs'] + converters_in_dict[convin][y]['var_env1']] * tstp)
                try:
                    if y != cy_list_adapted[-1]:
                        convin_varto2costs_list.extend([(converters_in_dict[convin][y]['var_env2']) * config_laend.aux_year_steps] * tstp)
                    if y == cy_list_adapted[-1]:
                        convin_varto2costs_list.extend([converters_in_dict[convin][y]['var_env2']] * tstp)
                except:
                    proxy = 'nothing will happen, variable is only needed for "except" task'
            
            #generate oemof object for converter_in investment, valid for individual periods
            cy = converters_in_dict[convin][calc_year]
            from1 = cy['from1']
            to1 = cy['to1']
            to2 = cy['to2']
            input_args = {busd[from1]: solph.flows.Flow(
                variable_costs=convin_varfrom1costs_list,
                investment=solph.Investment(
                    maximum=convin_maximum_list,
                    minimum=0,
                    overall_maximum=cy['max_total_capacity'],
                    ep_costs=convin_ep_cost_list, #already contains weighting of financial and environmental parameters
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))}
            output_args = {busd[to1]: solph.flows.Flow(
                variable_costs=convin_varto1costs_list)} #already contains weighting of financial and environmental parameters
            conversion_facts = {busd[from1]: cy['conversion_factor_f1'],
                                busd[to1]: cy['conversion_factor_t1']}
            try:
                output_args[busd[to2]] = solph.flows.Flow(
                    variable_costs=convin_varto2costs_list)
                conversion_facts[busd[to2]] = cy['conversion_factor_t2']
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
                
            nodes.append(solph.components.Converter(
                label=convin,
                inputs=input_args,
                outputs=output_args,
                conversion_factors=conversion_facts)) 
            logging.debug(f'{convin} (Converter_in) created.')

        #if there is an already existing converters_in  
        if any(key == 'existing' for key in converters_in_dict[convin].keys()):
            cy = converters_in_dict[convin]['existing']
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            convin_varfrom1costs_list = []
            convin_varto1costs_list = []
            convin_varto2costs_list = []
            convin_varfrom1costs_list.extend([(cy['var_from1_costs']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            convin_varto1costs_list.extend([(cy['var_to1_costs'] + cy['var_env1']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            convin_varfrom1costs_list.extend([cy['var_from1_costs']] * tstp)
            convin_varto1costs_list.extend([cy['var_to1_costs'] + cy['var_env1']] * tstp)
            try:
                convin_varto2costs_list.extend([cy['var_env2'] * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
                convin_varto2costs_list.extend([cy['var_env2']] * tstp)
            except:
                proxy = 'nothing will happen, proxy-variable is only needed for "except" task'
            
            #generate oemof object for already existing converter_in
            from1 = cy['from1']
            to1 = cy['to1']
            to2 = cy['to2']
            
            input_args = {busd[from1]: solph.flows.Flow(
                variable_costs=convin_varfrom1costs_list,
                investment=solph.Investment(
                    maximum=0,
                    ep_costs=0,
                    existing=cy['initially_installed_capacity'],
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    age=0,
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))}
            output_args = {busd[to1]: solph.flows.Flow(
                variable_costs=convin_varto1costs_list)}
            conversion_facts = {busd[from1]: cy['conversion_factor_f1'],
                                busd[to1]: cy['conversion_factor_t1']}
            try:
                output_args[busd[to2]] = solph.flows.Flow(
                    variable_costs=convin_varto2costs_list)
                conversion_facts[busd[to2]] = cy['conversion_factor_t2']
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
                
            nodes.append(solph.components.Converter(
                label=convin+'_existing',
                inputs=input_args,
                outputs=output_args,
                conversion_factors=conversion_facts)) 
            logging.debug(f'{convin}_existing (Converter_in) created.')
            
           
    ####Create converters_out
    #collect and cluster all converter_out technologies
    converters_out_dict = {}
    for i, x in scenario_obj['converters_out'].iterrows():
        if x['label'][-2:].isdigit():
            x_label = x['label'][:-3]
            if x_label in converters_out_dict:
                converters_out_dict[x_label][f'{x["label"][-2:]}'] = x #writes each year separately to converters_out_dict[convout]
            else:
                converters_out_dict[x_label] = {} #if converters_out_dict[convout] is not yet exisiting, generate!
                converters_out_dict[x_label][f'{x["label"][-2:]}'] = x
        elif x['initially_installed_capacity'] > 0:
            x_label = x['label'][:-9]
            if x_label in converters_out_dict:
                converters_out_dict[x_label]['existing'] = x
            else:
                converters_out_dict[x_label] = {}
                converters_out_dict[x_label]['existing'] = x         
        else:
            if x['label'] in converters_out_dict:
                converters_out_dict[x['label']]['allperiods'] = x
            else:
                converters_out_dict[x['label']] = {}
                converters_out_dict[x['label']]['allperiods'] = x
    
    #go through clustered converters_out and create oemof nodes for multi period optimization
    for convout in converters_out_dict.keys():
        #check, if for all periods (=calculation years) respective information is available
        if not all(key == 'allperiods' or key == 'existing' for key in converters_out_dict[convout].keys()):
            for calc_year in cy_list_adapted:
                if not calc_year in converters_out_dict[convout].keys():
                    raise ValueError(f'\n\n\n!!!!!!!!!!\nPeriod {calc_year} is missing for {convout}.\n!!!!!!!!!!\n\n\n')
            #shorten converters_out[convout] dict to the actual amount of periods thatt shall be optimized
            converters_out_dict[convout] = {year: value for year, value in converters_out_dict[convout].items() if year in cy_list_adapted or year == 'existing'}
            
        #if there are converters_out information valid for all periods         
        if any(key == 'allperiods' for key in converters_out_dict[convout].keys()):
            cy = converters_out_dict[convout]['allperiods']
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            convout_varfrom1costs_list = []
            convout_varfrom2costs_list = []
            convout_varto1costs_list = []
            #write var_from1_costs:
            convout_varfrom1costs_list.extend([cy['var_from1_costs'] * config_laend.aux_year_steps] * tstp * (len(calc_years)-1)) #multiply variable_costs by period duration except last period
            convout_varfrom1costs_list.extend([cy['var_from1_costs']] * tstp) #add variable_costs for last period
            
            #write var_to1_costs:
            if cy['var_to1_costs'] == 'timevariable_costs':
                #go to timeseries and write these costs
                for col in scenario_obj['timeseries'].columns.values:
                    if col == f'{cy["label"]}.timevariable_costs':
                        for calc_year in calc_years[:-1]:
                            convout_varto1costs_list.extend((scenario_obj['timeseries'][col] + cy['var_env1']) * config_laend.aux_year_steps) #multiply variable_costs by period duration except last period
                        convout_varto1costs_list.extend(scenario_obj['timeseries'][col] + cy['var_env1'])                       
            
            elif cy['var_to1_costs'] == 'timevariable_penalty':
                for col in scenario_obj['timeseries'].columns.values:
                    if col == f'{cy["label"]}.timevariable_penalty':
                        #go to timeseries and write these costs to first reference year - for subsequent years, write last value (8760) of timeseries
                        for calc_year in calc_years[:1]:
                            convout_varto1costs_list.extend((scenario_obj['timeseries'][col] + cy['var_env1']) * config_laend.aux_year_steps) #multiply variable_costs by period duration except last period
                        for calc_year in calc_years[1:-1:1]:
                            convout_varto1costs_list.extend([(scenario_obj['timeseries'][col][8760] + cy['var_env1']) * config_laend.aux_year_steps] * tstp)
                        convout_varto1costs_list.extend([(scenario_obj['timeseries'][col][8760] + cy['var_env1'])] * tstp)
            
            else:
                convout_varto1costs_list.extend([(cy['var_to1_costs'] + cy['var_env1']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1)) #multiply variable_costs by period duration except last period
                convout_varto1costs_list.extend([cy['var_to1_costs'] + cy['var_env1']] * tstp) #add variable_costs for last period
            
            #check, if there are var_from2_costs:
            try:
                convout_varfrom2costs_list.extend([cy['var_from2_costs'] * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
                convout_varfrom2costs_list.extend([cy['var_from2_costs']] * tstp)
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
               

            #generate oemof object for converter_out investment, valid for all periods
            from1 = cy['from1']
            from2 = cy['from2']
            to1 = cy['to1']
             
            input_args = {busd[from1]: solph.flows.Flow(
                variable_costs=convout_varfrom1costs_list)}
            output_args = {busd[to1]: solph.flows.Flow(
                variable_costs=convout_varto1costs_list,
                investment=solph.Investment(
                    maximum=cy['max_capacity_invest'],
                    minimum=0,
                    overall_maximum=cy['max_total_capacity'],
                    ep_costs=(cy['invest'] + cy['inv1']) * tstp/8760,
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))}
            conversion_facts = {busd[from1]: cy['conversion_factor_f1'],
                                busd[to1]: cy['conversion_factor_t1']}
            try:
                input_args[busd[from2]] = solph.flows.Flow(
                    variable_costs=convout_varfrom2costs_list)
                conversion_facts[busd[from2]] = cy['conversion_factor_f2']
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
                 
            nodes.append(solph.components.Converter(
                label=convout,
                inputs=input_args,
                outputs=output_args,
                conversion_factors=conversion_facts)) 
            logging.debug(f'{convout} (Converter_in) created.')
        
        #if converters_out information are NOT valid for all periods, BUT for each individual period
        elif not all(key == 'existing' for key in converters_out_dict[convout].keys()):
            #collect individual information for each period and generate lists for multi period optimization
            convout_maximum_list = [converters_out_dict[convout][y]['max_capacity_invest'] for y in cy_list_adapted]
            convout_ep_cost_list = [(converters_out_dict[convout][y]['invest'] + converters_out_dict[convout][y]['inv1']) * tstp/8760 for y in cy_list_adapted]
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            convout_varfrom1costs_list = []
            convout_varfrom2costs_list = []
            convout_varto1costs_list = []
            #write var_from1_costs and try var_from2_costs:
            for y in cy_list_adapted:
                if y != cy_list_adapted[-1]:
                    convout_varfrom1costs_list.extend([converters_out_dict[convout][y]['var_from1_costs'] * config_laend.aux_year_steps] * tstp) #multiply variable_costs by period duration except last period
                elif y == cy_list_adapted[-1]:
                    convout_varfrom1costs_list.extend([converters_out_dict[convout][y]['var_from1_costs']] * tstp) #add variable_costs for last period
                try:
                    if y != cy_list_adapted[-1]:
                        convout_varfrom2costs_list.extend([converters_out_dict[convout][y]['var_from2_costs'] * config_laend.aux_year_steps] * tstp) #multiply variable_costs by period duration except last period
                    elif y == cy_list_adapted[-1]:   
                        convout_varfrom2costs_list.extend([converters_out_dict[convout][y]['var_from2_costs']] * tstp) #add variable_costs for last period
                except:
                    proxy = 'nothing will happen, variable is only needed for "except" task'
                    
            #write var_to1_costs:
            for y in cy_list_adapted:
                if converters_out_dict[convout][y]['var_to1_costs'] == 'timevariable_costs':
                    for col in scenario_obj['timeseries'].columns.values:
                        if col == f'{converters_out_dict[convout][y]["label"]}.timevariable_costs':
                            if y != cy_list_adapted[-1]:
                                convout_varto1costs_list.extend((scenario_obj['timeseries'][col] + converters_out_dict[convout][y]['var_env1']) * 
                                                                config_laend.aux_year_steps) #multiply variable_costs by period duration except last period
                                convout_varto1costs_list.extend(scenario_obj['timeseries'][col] + converters_out_dict[convout][y]['var_env1']) #add variable_costs for last period
                else:
                    convout_varto1costs_list.extend([(converters_out_dict[convout][y]['var_to1_costs'] + converters_out_dict[convout][y]['var_env1']) * config_laend.aux_year_steps] * tstp)
                    convout_varto1costs_list.extend([converters_out_dict[convout][y]['var_to1_costs'] + converters_out_dict[convout][y]['var_env1']] * tstp)

            
            #generate oemof object for converter_out investment, valid for all individual periods
            cy = converters_out_dict[convout][calc_year]
            from1 = cy['from1']
            from2 = cy['from2']
            to1 = cy['to1']
            input_args = {busd[from1]: solph.flows.Flow(
                variable_costs=convout_varfrom1costs_list)}
            output_args = {busd[to1]: solph.flows.Flow(
                variable_costs=convout_varto1costs_list,
                investment=solph.Investment(
                    maximum=convout_maximum_list,
                    minimum=0,
                    overall_maximum=cy['max_total_capacity'],
                    ep_costs=convout_ep_cost_list, #already contains weighting of financial and environmental parameters
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))} #already contains weighting of financial and environmental parameters
            conversion_facts = {busd[from1]: cy['conversion_factor_f1'],
                                busd[to1]: cy['conversion_factor_t1']}
            try:
                input_args[busd[from2]] = solph.flows.Flow(
                    variable_costs=convout_varfrom2costs_list)
                conversion_facts[busd[from2]] = cy['conversion_factor_f2']
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
                
            nodes.append(solph.components.Converter(
                label=convout,
                inputs=input_args,
                outputs=output_args,
                conversion_factors=conversion_facts)) 
            logging.debug(f'{convout} (Converter_in) created.')
        
        #if there is an already existing converters_out  
        if any(key == 'existing' for key in converters_out_dict[convout].keys()):
            cy = converters_out_dict[convout]['existing']
            #due to oemof v0.5.2 bug, workaround (multiplying variable_costs by period duration except last period) necessary
            convout_varfrom1costs_list = []
            convout_varfrom2costs_list = []
            convout_varto1costs_list = []
            convout_varfrom1costs_list.extend([cy['var_from1_costs'] * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            convout_varto1costs_list.extend([(cy['var_to1_costs'] + cy['var_env1']) * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
            convout_varfrom1costs_list.extend([cy['var_from1_costs']] * tstp)
            convout_varto1costs_list.extend([cy['var_to1_costs'] + cy['var_env1']] * tstp)
            try:
                convout_varfrom2costs_list.extend([cy['var_from2_costs'] * config_laend.aux_year_steps] * tstp * (len(calc_years)-1))
                convout_varfrom2costs_list.extend([cy['var_from2_costs']] * tstp)
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
            
            #generate oemof object for already existing converter_out
            from1 = cy['from1']
            from2 = cy['from2']
            to1 = cy['to1']
             
            input_args = {busd[from1]: solph.flows.Flow(
                variable_costs=convout_varfrom1costs_list)}
            output_args = {busd[to1]: solph.flows.Flow(
                variable_costs=convout_varto1costs_list,
                investment=solph.Investment(
                    maximum=0,
                    ep_costs=0,
                    existing=cy['initially_installed_capacity'],
                    lifetime=int(cy['lifetime']), #must be !INTEGER!, nevertheless an oemof inherent file (investment_flow_blocks.py) has a problem
                    age=0,
                    interest_rate=0, #interest rate was already considered in adaptForObjective computation. Considering here would lead to problems if env impacts are included in ep_costs
                    fixed_costs=cy['om'] * tstp/8760))}
            conversion_facts = {busd[from1]: cy['conversion_factor_f1'],
                                busd[to1]: cy['conversion_factor_t1']}
            try:
                input_args[busd[from2]] = solph.flows.Flow(
                    variable_costs=convout_varfrom2costs_list)
                conversion_facts[busd[from2]] = cy['conversion_factor_f2']
            except:
                proxy = 'nothing will happen, variable is only needed for "except" task'
                 
            nodes.append(solph.components.Converter(
                label=convout+'_existing',
                inputs=input_args,
                outputs=output_args,
                conversion_factors=conversion_facts)) 
            logging.debug(f'{convout}_existing (Converter_out) created.')


    return nodes



def processResults(results_main, results_meta, calc_years, scenario):
    '''
    Process Results so they can be exported to results excel file, also containing environmental impacts
    
    Parameters
    ----------
    results_main : dict containing the results for all nodes and flows
    results_meta : dict containing objective and information about the problem and solver
    calc_years: list of representative years (=periods) that are optimized
    scenario : scenario dictionary containing all relevant techno-ecological and -economic information.\
        Necessary for processing with optimization results.
        
    Returns
    -------
    combined: DataFrame containing investment results and variable results including LCA impacts
    investments: DataFrame containing all investment results including LCA impacts
    variable: DataFrame containing all variable results including LCA impacts, aggregeted to one year
    flow_overview: DataFrame containting all flows for all periods
    '''
    
    ####general processing
    nodes = [x for x in results_main.keys() if x[1] is None]
    flows = [x for x in results_main.keys() if x[1] is not None]

    investments = pd.DataFrame()
    variable = pd.DataFrame()
    flow_overview = pd.DataFrame()
    ####processing nodes (storages)
    nodes_list = []
    
    if config_laend.multiperiod_pf == True:
        for node in nodes:
            nodes_list.append(str(node[0]))
            if results_main[node]['period_scalars'].notna().any().any():
                if not (results_main[node]['period_scalars']['total'] == 0).all():
                    #generate an overview of the storage level for all periods
                    total_flow_series = results_main[node]['sequences']['storage_content']
                    flow_overview[f'{str(node[0])}_content-level'] = total_flow_series
                    
                    invest_name = str(node[0])
                    #iterate through period_scalars (investment results per period)
                    for y, r in results_main[node]['period_scalars'].iterrows():
                                                      
                        #multiply results with specific costs and LCA data
                        scen_stor = scenario['storages']
                        for i, t in scen_stor.iterrows():
                            if invest_name == t['label'] or f'{invest_name}_{str(y)[-2:]}' == t['label']:
                                investment_series = processing_investments(y, r, t, results_meta)
                                break
                        
                        #write results to invest DataFrame, that is afterwards exported as .xlsx file
                        investments[f'{invest_name}_{y}'] = investment_series
        
        for flow in flows:
            ####processing mulit-period investment 'flows'
            #make sure, that only technologies with investments are considered and no (storage-) flow investments overwrite already written node (capacity-) investments
            if results_main[flow]['period_scalars'].notna().any().any() and not any(str(flow[0]) == x for x in nodes_list) and not any(str(flow[1]) == x for x in nodes_list):
                if not (results_main[flow]['period_scalars']['total'] == 0).all():
                    if str(flow[0])[:3] != 'bus':
                        invest_name = str(flow[0])
                    elif str(flow[0])[:3] == 'bus':
                        invest_name = str(flow[1])       
    
                    #iterate through period_scalars (investment results per period)
                    for y, r in results_main[flow]['period_scalars'].iterrows():
                        
                        # here starts a funciton: investment_series = def collect_flowResults(y, r)
                        
                        #multiply results with specific costs and LCA data
                        finished = False
                        scen_data_list = ['renewables', 'converters_in', 'converters_out']
                        for scen_data in scen_data_list:
                            for i, t in scenario[scen_data].iterrows():
                                if invest_name == t['label'] or f'{invest_name}_{str(y)[-2:]}' == t['label']:
                                    investment_series = processing_investments(y, r, t, results_meta)
                                    finished = True
                                if finished == True:
                                    break
                            if finished == True:
                                break
                        
                        #write results to invest DataFrame, that is afterwards exported as .xlsx file
                        investments[f'{invest_name}_{y}'] = investment_series
                
            # access function of "processing_variable_flows",
            #collecting all relevant data regarding variable flows that shall be returned via the results excel
            variable, flow_overview = processing_variable_flows(variable, flow, results_main, results_meta, flow_overview, calc_years, scenario)
    
    
    #if myopic optimization is activated:
    else:
        for node in nodes:
            nodes_list.append(str(node[0]))
            if results_main[node]['scalars'].notna().any():
                if not results_main[node]['scalars'].loc['total'] == 0:
                    #generate an overview of the storage level for all periods
                    total_flow_series = results_main[node]['sequences']['storage_content']
                    flow_overview[f'{str(node[0])}_content-level'] = total_flow_series
                    
                    invest_name = str(node[0])
                    # access investment results
                    r = results_main[node]['scalars']
                    y = r.name.year
                                                      
                    #multiply results with specific costs and LCA data
                    scen_stor = scenario['storages']
                    for i, t in scen_stor.iterrows():
                        if invest_name == t['label'] or f'{invest_name}_{str(y)[-2:]}' == t['label']:
                            investment_series = processing_investments(y, r, t, results_meta)
                            break
                    
                    #write results to invest DataFrame, that is afterwards exported as .xlsx file
                    investments[f'{invest_name}_{y}'] = investment_series
                    
        for flow in flows:
            ####processing myopic investment 'flows'
            #make sure, that only technologies with investments are considered and no (storage-) flow investments overwrite already written node (capacity-) investments
            if results_main[flow]['scalars'].notna().any() and not any(str(flow[0]) == x for x in nodes_list) and not any(str(flow[1]) == x for x in nodes_list):
                if not results_main[flow]['scalars'].loc['total'] == 0:
                    if str(flow[0])[:3] != 'bus':
                        invest_name = str(flow[0])
                    elif str(flow[0])[:3] == 'bus':
                        invest_name = str(flow[1])       
    
                    # access investment results
                    r = results_main[flow]['scalars']
                    y = r.name.year
                        
                    #multiply results with specific costs and LCA data
                    finished = False
                    scen_data_list = ['renewables', 'converters_in', 'converters_out']
                    for scen_data in scen_data_list:
                        for i, t in scenario[scen_data].iterrows():
                            if invest_name == t['label'] or f'{invest_name}_{str(y)[-2:]}' == t['label']:
                                investment_series = processing_investments(y, r, t, results_meta)
                                finished = True
                            if finished == True:
                                break
                        if finished == True:
                            break
                    
                    #write results to invest DataFrame, that is afterwards exported as .xlsx file
                    investments[f'{invest_name}_{y}'] = investment_series        

            # access function of "processing_variable_flows",
            #collecting all relevant data regarding variable flows that shall be returned via the results excel
            variable, flow_overview = processing_variable_flows(variable, flow, results_main, results_meta, flow_overview, calc_years, scenario)

                    
    ####final preparation and ordering for excel export
    investments = investments.T
    variable = variable.T
    
    #combine invetment df and variable df to one total overview df, ordered by periods (ascending) and type (invest to variable)
    combined = pd.concat([investments, variable])
    combined = combined.fillna(0)
    combined.sort_values(by=['year', 'type'], inplace=True)
    try:
        #reorder columns so that capacity/flow information and respective cost information is at the left hand side and LCA data afterwards
        desired_column_order = ['unit', 'year', 'type', 'invest_capacity', 'total_capacity', 'invest_cost', 'invest_cost+wacc', 'lifetime',
                                'annualized_invest', 'om_p.a.', 'flow_p.a.', 'variable_costs_p.a.', 'ann_total_cost_p.a.', 'objective']
        LCA_columns = [col for col in combined.columns if col not in desired_column_order]
        desired_column_order += sorted(LCA_columns)
        combined = combined[desired_column_order]
    except:
        proxy = 'nothing'

    return combined, investments, variable, flow_overview, LCA_columns



def processing_investments(y, r, t, results_meta):
    '''
        Parameters
    ----------
    y : integer - year of interest
    r : series - representing a row containing results of a specific technology
    t : series - representing a row containing parameter information of a specific technology (coming from scenario excel)
    results_meta : dict containing objective and information about the problem and solver

    Returns
    -------
    investment_series : series containing information about specific investment, that is afterwards added to a dict

    '''
    
    investment_series = pd.Series(data={'unit': t['unit'], 'year': y, 'type': 'investment', 'invest_capacity': r['invest'], 'total_capacity': r['total'], 'objective': results_meta['objective']}, name=y)
    investment_series['invest_cost'] = r['invest'] * t['invest']
    investment_series['invest_cost+wacc'] = economics.annuity(r['invest'] * t['invest'], t['lifetime'], config_laend.InvestWacc) * t['lifetime']
    investment_series['lifetime'] = t['lifetime']
    investment_series['om_p.a.'] = r['total'] * t['om']
    investment_series['annualized_invest'] = 0
    investment_series['ann_total_cost_p.a.'] = 0
    LCA_ser = pd.Series(data=r['invest'] * t['inv1'], name=y)
    investment_series = pd.concat([investment_series, LCA_ser])
    
    return investment_series



def processing_variable_flows(variable, flow, results_main, results_meta, flow_overview, calc_years, scenario):
    '''
    Parameters
    ----------
    variable: empty pd.DataFrame - will be filled within this function with the wanted data/information
    flow: dictionary containing the flows of oemof main_results
    results_main: dict containing the results for all nodes and flows
    results_meta: dict containing objective and information about the problem and solver
    flow_overview: DataFrame containing overview of flows
    calc_years: list of representative years (=periods) that are optimized
    scenario: scenario dictionary containing all relevant techno-ecological and -economic information.

    Returns
    -------
    variable: filled pd.DataFrame, containing information about all variable flows
    
    '''

    #check, if there are results containing dsm information - and if so, neglect them
    if not any("dsm" in col.lower() for col in results_main[flow]['sequences'].columns):
        #check, if there are results without flows (i.e. all flows=0) - and if so, neglect them
        if (results_main[flow]['sequences']['flow'] >= 0).any():
            df = results_main[flow]['sequences']
        
            #generate an flow overview for all periods
            total_flow_series = df['flow']
            flow_overview[f'{flow[0]}/{flow[1]}'] = total_flow_series

            #generate an overview of all flows per year and their respective impacts (financial and ecological)
            for cy in calc_years:
                y_flow_series = pd.Series(data=df[df.index.year == cy]['flow'], name=cy)
                y_flow_series.index = pd.RangeIndex(1, len(y_flow_series) + 1)
                ####----commodity flows
                if str(flow[0])[:8] == 'resource':
                    scen_comm = scenario['commodity_sources']
                    for i, t in scen_comm.iterrows():
                        t_name = str(flow[0])
                        if t_name == t['label']:
                            if t['timevariable_costs']:
                                y_cost = sum(y_flow_series * scenario['timeseries'][f'{t["label"]}.timevariable_costs'])
                                break
                            else:
                                y_cost = sum(y_flow_series) * t['variable_costs']
                                finished = True
                                break
    
                        elif f'{t_name}_{str(cy)[-2:]}' == t['label']:
                            if t['timevariable_costs']:
                                y_cost = sum(y_flow_series * scenario['timeseries'][f'{t["label"]}.timevariable_costs'])
                                break
                            else:
                                y_cost = sum(y_flow_series) * t['variable_costs']
                                finished = True
                                break
                        
                    variable_series = pd.Series(data={'unit': t['unit'], 'year': cy, 'type': 'variable', 'flow_p.a.': sum(y_flow_series), 'variable_costs_p.a.': y_cost, 'objective': results_meta['objective']}, name=f'{flow[0]}/{flow[1]}_{cy}')
                    LCA_ser = pd.Series(data=sum(y_flow_series) * t['var_env1'], name=f'{flow[0]}/{flow[1]}_{cy}')
                    variable_series = pd.concat([variable_series, LCA_ser])
                
                elif str(flow[0])[:3] == 'bus':
                    ####----excess or shortage flows
                    if str(flow[1]).split('_')[-1:][0] == 'excess' or str(flow[1]).split('_')[-1:][0] == 'shortage':
                        if str(flow[1]).split('_')[-1:][0] == 'excess':
                            pre_syllable = 'excess'
                        elif str(flow[1]).split('_')[-1:][0] == 'shortage':
                            pre_syllable = 'shortage'
                    
                        scen_bus = scenario['buses']
                        for i, t in scen_bus.iterrows():
                            t_name = str(flow[0])
                            if t_name == t['label']:
                                #check, if bus parameters are based on flexible timeseries (i.e. iterables)
                                #if yes
                                if isinstance(t[f'{pre_syllable}_costs'], str):
                                    if t[f'{pre_syllable}_costs'] == "c_avar":
                                        y_cost = sum(y_flow_series) * t[f'c_avar_{str(cy)[-2:]}']
                                        LCA_ser = pd.Series(data=sum(y_flow_series) * t[f'{pre_syllable}_env'], name=f'{flow[0]}/{flow[1]}_{cy}')
                                    elif t[f'{pre_syllable}_costs'] == "c_var":
                                        y_cost = sum(y_flow_series * scenario['timeseries'][f'{t["label"]}_{str(cy)[-2:]}.timevariable_costs'])
                                        LCA_ser = pd.Series(data=sum(y_flow_series) * t[f'{pre_syllable}_env'], name=f'{flow[0]}/{flow[1]}_{cy}')
                                    break
                                #if not
                                else:
                                    y_cost = sum(y_flow_series) * t[f'{pre_syllable}_costs']
                                    LCA_ser = pd.Series(data=sum(y_flow_series) * t[f'{pre_syllable}_env'], name=f'{flow[0]}/{flow[1]}_{cy}')
                                    break
                    
                        variable_series = pd.Series(data={'unit': t['unit'], 'year': cy, 'type': 'variable', 'flow_p.a.': sum(y_flow_series), 'variable_costs_p.a.': y_cost, 'objective': results_meta['objective']}, name=f'{flow[0]}/{flow[1]}_{cy}')
                        variable_series = pd.concat([variable_series, LCA_ser])
    
                    ####----variable demand flows
                    elif str(flow[1])[:4] == 'load':
                        y_cost = 0
                        unit = next((x['unit'] for _, x in scenario['buses'].iterrows() if x['label'] == str(flow[0])), None)
                        variable_series = pd.Series(data={'unit': unit, 'year': cy, 'type': 'variable', 'flow_p.a.': sum(y_flow_series), 'variable_costs_p.a.': y_cost, 'objective': results_meta['objective']}, name=f'{flow[0]}/{flow[1]}_{cy}')
                        LCA_ser = pd.Series(data=0, index=config_laend.system_impacts_index[1:-3], name=f'{flow[0]}/{flow[1]}_{cy}')
                        variable_series = pd.concat([variable_series, LCA_ser])
                    
                    ####----variable input flows for storages or converters    
                    else: 
                        finished = False
                        scen_data_list = ['storages', 'converters_in', 'converters_out']
                        for scen_data in scen_data_list:
                            for i, t in scenario[scen_data].iterrows():
                                t_name = str(flow[1])
                                if t_name == t['label'] or f'{t_name}_{str(cy)[-2:]}' == t['label']:
                                    if scen_data == 'storages':
                                        y_cost = sum(y_flow_series) * t['variable_input_costs']
                                        finished = True
                                    else:
                                        if str(flow[0]) == t['from1']:
                                            y_cost = sum(y_flow_series) * t['var_from1_costs']
                                            finished = True
                                        elif str(flow[0]) == t['from2']:
                                            y_cost = sum(y_flow_series) * t['var_from2_costs']
                                            finished = True
                                        else:
                                            KeyError(f'Can not find data of {flow[0]}/{t_name} flow in scenario-file for processing of results')
                                if finished == True:
                                    break
                            if finished == True:
                                break
    
                        unit = next((x['unit'] for _, x in scenario['buses'].iterrows() if x['label'] == str(flow[0])), None)
                        variable_series = pd.Series(data={'unit': unit, 'year': cy, 'type': 'variable', 'flow_p.a.': sum(y_flow_series), 'variable_costs_p.a.': y_cost, 'objective': results_meta['objective']}, name=f'{flow[0]}/{flow[1]}_{cy}')
                        LCA_ser = pd.Series(data=0, index=config_laend.system_impacts_index[1:-3], name=f'{flow[0]}/{flow[1]}_{cy}')
                        variable_series = pd.concat([variable_series, LCA_ser])
            
                ####----variable output flows for renewables, storages, converters        
                else:
                    finished = False
                    scen_data_list = ['renewables', 'storages', 'converters_in', 'converters_out']
                    for scen_data in scen_data_list:
                        for i, t in scenario[scen_data].iterrows():
                            t_name = str(flow[0])
                            if t_name == t['label'] or f'{t_name}_{str(cy)[-2:]}' == t['label']:
                                if scen_data == 'renewables':
                                    y_cost = sum(y_flow_series) * t['variable_costs']
                                    LCA_ser = pd.Series(data=sum(y_flow_series) * t['var_env1'], name=f'{flow[0]}/{flow[1]}_{cy}')
                                    finished = True
                                elif scen_data == 'storages':
                                    y_cost = sum(y_flow_series) * t['variable_output_costs']            
                                    LCA_ser = pd.Series(data=0, index=config_laend.system_impacts_index[1:-3], name=f'{flow[0]}/{flow[1]}_{cy}')
                                    finished = True
                                else:
                                    if str(flow[1]) == t['to1']:
                                        if t['var_to1_costs'] == "timevariable_costs":
                                            y_cost = sum(y_flow_series * scenario['timeseries'][f'{t["label"]}.timevariable_costs'])
                                        else:
                                            y_cost = sum(y_flow_series) * t['var_to1_costs']
                                        LCA_ser = pd.Series(data=sum(y_flow_series) * t['var_env1'], name=f'{flow[0]}/{flow[1]}_{cy}')
                                        finished = True
                                    elif str(flow[1]) == t['to2']:
                                        y_cost = 0
                                        LCA_ser = pd.Series(data=sum(y_flow_series) * t['var_env2'], name=f'{flow[0]}/{flow[1]}_{cy}')
                                        finished = True
                                    else:
                                        KeyError(f'Can not find data of {flow[0]}/{t_name} flow in scenario-file for processing of results')
                            if finished == True:
                                break
                        if finished == True:
                            break
                                    
                    unit = next((x['unit'] for _, x in scenario['buses'].iterrows() if x['label'] == str(flow[1])), None)
                    variable_series = pd.Series(data={'unit': unit, 'year': cy, 'type': 'variable', 'flow_p.a.': sum(y_flow_series), 'variable_costs_p.a.': y_cost, 'objective': results_meta['objective']}, name=f'{flow[0]}/{flow[1]}_{cy}')
                    variable_series = pd.concat([variable_series, LCA_ser])
            
                variable[f'{flow[0]}/{flow[1]}_{cy}'] = variable_series
    
    return variable, flow_overview



def summarizeIndividualResults(xls, LCA_columns, calc_years):
    '''
    summarizeIndividualResults summarizes individual results/flows for subsequent (external) figure generation
    Concretely: 
        regional and overregional grid fees are summed up to one grid fee position;
        water supply costs, that are differentiated into water_ultrapure and water_deionized are summed up to one water supply position
        !However, their individual share still remains accessible!
        
    Parameters
    ----------
    xls : initial xls, that was named "combined" during "processResults" function. xls contains all results without summarization
    LCA_columns: list containing the column names containing LCA information
    calc_years: list containing all considered representative years    
    
    Returns
    -------
    combined_summarization: DataFrame with summarized individual results, containing investment results and variable results including LCA impacts
    '''
    
    ####initializing fee & water summarization
    fee = {}
    water = {}
    
    for y in calc_years:
        fee[str(y)] = {}
        fee[str(y)]['variable_costs_p.a.'] = 0
        for LCA_data in LCA_columns:
            fee[str(y)][LCA_data] = 0
        water[str(y)] = {}
        water[str(y)]['variable_costs_p.a.'] = 0
        for LCA_data in LCA_columns:
            water[str(y)][LCA_data] = 0
    
    ####summarizing
    for i, t in xls.iterrows():
        # collect grid fee data
        if "bus_el-re/regional-el_converter" in i:
            y = str(t['year'])
            fee[y]['variable_costs_p.a.'] = fee[y]['variable_costs_p.a.'] + t['variable_costs_p.a.']
            
            for LCA_data in LCA_columns:
                fee[y][LCA_data] = fee[y][LCA_data] + t[LCA_data]
        
        if "bus_el-over/overreg-el_converter" in i:
            y = str(t['year'])
            fee[y]['variable_costs_p.a.'] = fee[y]['variable_costs_p.a.'] + t['variable_costs_p.a.']
            
            for LCA_data in LCA_columns:
                fee[y][LCA_data] = fee[y][LCA_data] + t[LCA_data]
        
        if "bus_el-grid/grid-el_converter" in i:
            y = str(t['year'])
            fee[y]['variable_costs_p.a.'] = fee[y]['variable_costs_p.a.'] + t['variable_costs_p.a.']
            
            for LCA_data in LCA_columns:
                fee[y][LCA_data] = fee[y][LCA_data] + t[LCA_data]
        
        # collect water cost data    
        if i[:14] == "resource_water": 
            y = str(t['year'])
            water[y]['variable_costs_p.a.'] = water[y]['variable_costs_p.a.'] + t['variable_costs_p.a.']
            
            for LCA_data in LCA_columns:
                water[y][LCA_data] = water[y][LCA_data] + t[LCA_data]
    
    ####writing
    copy_t = t   
    for y in fee:
        copy_t[:] = 0 
        copy_t['unit'] = "kWh"
        copy_t['year'] = y
        copy_t['type'] = "variable"
        copy_t['variable_costs_p.a.'] = fee[y]['variable_costs_p.a.']
        
        for LCA_data in LCA_columns:
            copy_t[LCA_data] = fee[y][LCA_data]
        
        xls.loc[f"gridfees/gridfees_{y}"] = copy_t
    
    for y in water:
        copy_t[:] = 0 
        copy_t['unit'] = "kg"
        copy_t['year'] = y
        copy_t['type'] = "variable"
        copy_t['variable_costs_p.a.'] = water[y]['variable_costs_p.a.']
        
        for LCA_data in LCA_columns:
            copy_t[LCA_data] = water[y][LCA_data]
        
        xls.loc[f"resource_water/resource_water_{y}"] = copy_t
    
    return xls



def annualization(xls):
    '''
    This function fills the hitherto empty columns "annualized_invest" and "ann_total_cost_p.a."
    
    Parameters
    ----------
    xls : combined_summarization, see returned xls in previous function "summarizeIndividualResults"
        
    Returns
    -------
    combined_annualized: DataFrame including the optimization results with all columns filled, ready for figure generation
    '''
    
    for i, t in xls.iterrows():
        if t['total_capacity'] > 0:
            ann_prev = 0
            y = int(t['year'])
            t_search = i[:-2]
            
            for j, t_prev in xls.iterrows():
                if j[:-2] == t_search and t_prev['invest_capacity'] > 0 and t_prev['year'] + t_prev['lifetime'] > y:
                    ann_prev = ann_prev + t_prev['invest_cost+wacc'] / t_prev['lifetime']
                
                if i == j:
                    break
                
            t['annualized_invest'] = ann_prev
            
            t['ann_total_cost_p.a.'] = t['annualized_invest'] + t['om_p.a.'] + t['variable_costs_p.a.']
            
            xls.loc[i] = t
            
        else:
            t["annualized_invest"] = 0
            
            t['ann_total_cost_p.a.'] = t['annualized_invest'] + t['om_p.a.'] + t['variable_costs_p.a.']
            
            xls.loc[i] = t
            
    return xls

    
####End


    
    

























            
