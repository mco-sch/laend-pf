
# import general libraries
import logging
from datetime import datetime
import pandas as pd

# import oemof libraries
import oemof.solph as solph
from oemof.tools import logger as oelog
from oemof.solph import Model

# import module files
import utils_pf_auto as utils
import config_pf_auto as config_pf

# this function runs first and is not specific to the objective
def main(run_name, time, filename_f):

    ###############################################################################
    ####Configure Logger
    ###############################################################################


    oelog.define_logging(
        logfile='laend.log',
        screen_level=logging.INFO,
        file_level=config_pf.log_file_level,
        logpath=run_name + '\\logs',
        timed_rotating={'when' : "s", 'interval':5}) #creates new file every 5 seconds to avoid overly large files with write errors

    #define calculation years and associated timeindex plus periods (necessary for perfect foresight)
    calc_years, timeindex, periods = utils.defineYearsForCalculation()
    
    #pass main starting info to logger
    utils.writeParametersLogger(calc_years)



    ###############################################################################
    ####Setup Data: Import
    ###############################################################################

    # import excel sheet of possible investments/usable technologies
    scenario = utils.CompileScenario(f'{run_name}\\files\\{time}_{filename_f}')
    utils.validateExcelInput(scenario)



    ###############################################################################
    ####Fixed Flows: import & configure (weather, demand, renewables)
    ###############################################################################


    ############### work with standard data ######################################

    # if available, import typical meteorological year as downloaded from https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
    try:
        tmy, tmy_month_year = utils.compileTMY(config_pf.filename_tmy)
    except:
        tmy, tmy_month_year = None, None
        logging.info('Continuing without TMY')


#    if config.update_heat_demand:

#        utils.getHeatDemand(testmode=True, ann_demands_per_type=config.ann_demands_per_type, temperature=tmy['T2m'])  
        
#        if config.separate_heat_water:
#            utils.importFixedFlow(run_name, time, config.filename_th_demand, 'demand', config.varname_th_low, col_name='total_heat')
#            utils.importFixedFlow(run_name, time, config.filename_th_demand, 'demand', config.varname_th_high, col_name='total_water')
#        else:
#            utils.importFixedFlow(run_name, time, config.filename_th_demand, 'demand', 'load_th', col_name='total')
   
#    utils.createSolarCollectorFixedFlow(config.varname_solar_collector_high, tmy, run_name, time) if config.update_Solar_Collector_data == True else None
    
#    utils.createSolarCollectorFixedFlow(config.varname_solar_collector_low, tmy, run_name, time) if config.update_Solar_Collector_data == True else None

#    utils.createHeatpumpAirFixedCOP(config.varname_a_w_hp_low, config.hp_temp_low, tmy, run_name, time) if config.update_heatpump_a_w_cop == True else None

#    utils.createHeatpumpAirFixedCOP(config.varname_a_w_hp_high, config.hp_temp_high, tmy, run_name, time) if config.update_heatpump_a_w_cop == True else None
    
#    if config.update_electricity_demand:
#        utils.importFixedFlow(run_name, time, config.filename_el_demand, 'demand', config.varname_el_demand, sum_mult_profiles=True)


#    if config.update_pv_opt_fix:
#        utils.createPvProfileForTMY(config.filename_pv_opt_fix, tmy_month_year, config.varname_pv_1)
#        utils.importFixedFlow(run_name, time, f'in/{config.location_name}/pvgis_tmy_{config.varname_pv_1}.csv', 'renewables', config.varname_pv_1, col_name = 'P', conversion =1/1000)
        
#        utils.createPvProfileForTMY(config.filename_pv_opt_fix, tmy_month_year, config.varname_pv_2)
#        utils.importFixedFlow(run_name, time, f'in/{config.location_name}/pvgis_tmy_{config.varname_pv_2}.csv', config.varname_pv_2, col_name='P', conversion=1/1000)
    
#    if config.update_pv_facade_fix:
#        utils.createPvProfileForTMY(config.filename_pv_facade_fix, tmy_month_year, config.varname_pv_3)

#        utils.importFixedFlow(run_name, time, f'in/{config.location_name}/pvgis_tmy_{config.varname_pv_3}.csv', 'renewables', config.varname_pv_3, col_name = 'P', conversion = 1/1000)

    utils.createWindPowerPlantFixedFlow(scenario, tmy, run_name, time)

       

    ###############################################################################
    ####LCA: Import & configure
    ###############################################################################

    # add LCA data to possible technologies
    scenario, lcia_methods, lcia_units = utils.addLCAData(scenario)
     
    # determine the investment annuitiy of possible technologies
#    tech = utils.calcInvestAnnuity(tech)

    # determine the emission factors used for the emission constraint
#    tech = utils.determineEmissionFactors(tech)

    # save environmental impacts and cost factors for result calculation
#    factors = utils.saveFactorsForResultCalculation(tech, env_units, run_name, time) #HH: run_name
    

#    if config.emission_constraint:

        # run optimization for just one leap year to get best possible result for objective of climate change total
#        result = optimizeForObjective(config.ec_impact_category, tech=tech, factors=factors, emission_limit=None, run_name=run_name, define_climate_neutral=True)

        # calculate climate neutrality as per optimization above
#        climate_neutral = utils.calculateClimateNeutralEmissions(result)

        # get the years that are needed for the optimizations later
#        years_for_calc = range(config.start_year, config.end_year + 1, config.InvestTimeSteps)

        # determine emission goals for each calc year for later optimizations
#        emission_goals = utils.calculateEmissionGoals(tech, years_for_calc, climate_neutral=climate_neutral)

        # export emission targets
#        emission_goals.to_excel(f'{run_name}\\files\\emission targets_{time}.xlsx')
#        logging.debug(emission_goals)

#    else:
#        emission_goals = None



    return scenario, lcia_units, timeindex, periods, calc_years#, factors, emission_goals



def optimizeForObjective(i, scenario, timeindex, periods, calc_years, run_name, time, define_climate_neutral = False):
    '''
    Parameters
    ----------
    i: objective as string, taken from config-file
    scenario: scenario dict including all relevant data
    timeindex: pandas.DatetimeIndex, without considering leap days
    periods: representative years, including hourly breakdown, without considering leap days
    calc_years: list of representative years (periods)
    run_name: name of currently running process containting path and start time
    time: start time of total computation process (also included in run_name)
    define_climate_neutral: defines if climate-neutral constraint is active

    Returns
    -------
    None.
    '''

    # get admin things out of the way: logger, start time, where to save things etc.
    if not i.find('|') == -1:
        i_name = i.replace('|', ',')
    else: i_name = i

    if define_climate_neutral: i_name = 'climate neutrality'

    #configure oemof logger
    oelog.define_logging(
        logfile=f'laend_{i_name}.log',
        screen_level=logging.INFO,
        file_level=logging.INFO,
        logpath= run_name + '\\logs',
        timed_rotating = {'when' : "s", 'interval':5} #creates new file every 5 seconds to avoid overly large files with write errors
    )

    obj_time = datetime.now()
    logging.info(f'Starting calculation for {i} at {obj_time}')

    # read weight and normalization factors for environmental impacts
    weightEnv, normalizationEnv = utils.readWeightingNormalization(config_pf.filename_weight_and_normalisation)

    #determine goal and weighting of environment and cost factors
    goals = utils.determineGoalForObj(i)

    # determine correction factor for solver for env. goals where values are very small
    c_factor = utils.determineCfactorForSolver(i)

    # sum environmental & environmental impacts based on weight, normalization and c_factor
    scenario_obj = utils.adaptForObjective(scenario, i, weightEnv, normalizationEnv, goals, c_factor, calc_years)


    ###################
    #in myopic laend, a lot of functions about climate_neutral/emission constraint,
    #myopic nature of computation are inserted here.
    ###################



    ###############################################################################
    ####Create oemof energy system and solve
    ###############################################################################

    #initialization of the energy system
    logging.info('Initializing energy system')
    
    #create the energy system
    if config_pf.multiperiod_pf == True:
        es = solph.EnergySystem(
            timeindex=timeindex,
            timeincrement=[1] * len(timeindex),
            periods=periods,
            infer_last_interval=False,
            )
    else:
        es = solph.EnergySystem(
            timeindex=timeindex,
            infer_last_interval=True,
            )
    
    #create oemof objects
    new_nodes = utils.createOemofNodes(scenario_obj, calc_years)

    # add new nodes and flows to energy system
    logging.info('Adding nodes to the energy system')
    es.add(*new_nodes)

    # creation of a least cost model from the energy system
    logging.info(f'Creating oemof model for {i}')
    om = Model(es, discount_rate=config_pf.DiscountRate)

    # set emission constraint (if config_pf.emission_constraint = False, all emissions & emission_constraint = 0)
#    if not emission_constraint == 0:

        # if calendar.isleap(year):
        #     emission_constraint = emission_constraint + emission_constraint*config.ec_leap_year_buffer
#        logging.info(f'Optimization includes emission constraint: {config_pf.emission_constraint}')
#        logging.info(f'Optimization includes emission constraint: {emission_constraint}')
#        constraints.emission_limit(om, limit=emission_constraint)

    logging.info(f'Oemof model for {i} has been created')

    # solving the linear problem using the given solver
    logging.info(f'Solving the optimization problem for {i}')
    ####################################################
    om.solve(solver=config_pf.solver, solve_kwargs={"tee": config_pf.solver_verbose,
                                                    "options": {
                                                    **config_pf.solver_options
                                                    }}
             )
    
    logging.info(f'Successfully solved the optimization problem for {i}')


    ###############################################################################
    ####Process results
    ###############################################################################      
       
    logging.info('Saving oemof results for further processing')
    results_main = solph.processing.results(om)
    results_meta = solph.processing.meta_results(om)

    #process results
    combined, investments, variable, flow_overview, LCA_columns = utils.processResults(results_main, results_meta, calc_years, scenario)
    
    #write results to .xlsx:
    with pd.ExcelWriter(f'{run_name}\\files\\Results for {i}_{time}.xlsx') as writer:
        combined.to_excel(writer, sheet_name='combined')
        investments.to_excel(writer, sheet_name='investments')
        variable.to_excel(writer, sheet_name='variable')
    
    flow_overview.to_excel(f'{run_name}\\files\\Results for {i}_FlowOverview_{time}.xlsx')
    
    combined_summarization = utils.summarizeIndividualResults(combined, LCA_columns, calc_years)
    
    with pd.ExcelWriter(f'{run_name}\\files\\Results for {i}_summarized_{time}.xlsx') as writer:
        combined_summarization.to_excel(writer, sheet_name='combined')
        investments.to_excel(writer, sheet_name='investments')
        variable.to_excel(writer, sheet_name='variable')
        
    combined_annualized = utils.annualization(combined_summarization)
    
    with pd.ExcelWriter(f'{run_name}\\files\\Results for {i}_annualized_{time}.xlsx') as writer:
        combined_annualized.to_excel(writer, sheet_name='combined')
        investments.to_excel(writer, sheet_name='investments')
        variable.to_excel(writer, sheet_name='variable')
        
        