from typing import Any


class Configuration(object):
    # reporter   
    reporterspath = 'results/'
    # learning models
    modelnames = ['ridge', 'lasso','pls','rf','svr_linear']
    # elements
    elements = ["Li", "B", "Na", "Mg", "P", "S", "K", "Ca43", "Mn", "Fe", "Co", "Cu", "Zn", "Rb", "Sr", "Mo", "Cd"]
    # element concentration and spectral values files
    sp_path = "data/spectral_data.csv"
    target_path = "data/nutrient_values_withoutdate.csv"
    
            
    