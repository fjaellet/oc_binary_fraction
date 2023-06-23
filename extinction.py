import numpy as np
from astropy.table import Table

def ext_coeff(A0, X, coeffs):
    """
    Implementing the formula given at
    https://www.cosmos.esa.int/web/gaia/edr3-extinction-law
    """
    return coeffs["Intercept"] + coeffs["X"]*X  + coeffs["X2"]*X*X   + coeffs["X3"]*X*X*X \
                               + coeffs["A"]*A0 + coeffs["A2"]*A0*A0 + coeffs["A3"]*A0**3 \
                               + coeffs["XA"]*X*A0 + coeffs["AX2"]*X*X*A0 + coeffs["XA2"]*X*A0*A0

def A0_to_AG_ABP_ARP(A0, G, BP, RP):
    """
    Convert A0 to extinction in the Gaia bands following 
    https://www.cosmos.esa.int/web/gaia/edr3-extinction-law
    
    Parameters:
        A0 - extinction at 5500 \AA
        G  - Gaia DR3 G mag
        BP - Gaia DR3 BP mag
        RP - Gaia DR3 RP mag
    Returns:
        Tuple AG, ABP, ARP
    """
    # Read the table extracted from 
    # https://www.cosmos.esa.int/documents/29201/1658422/Fitz19_EDR3_extinctionlawcoefficients.zip/b0903037-73ac-df18-a818-091747bacc27?t=1623425226157
    bab = Table.read("./data/extinctionlaw_edr3/Fitz19_EDR3_MainSequence.csv")
    # Get the relevant lines 
    coeffs_G  = bab[ (bab["Xname"]=="BPRP") & (bab["Kname"]=="kG")][0]
    coeffs_BP = bab[ (bab["Xname"]=="BPRP") & (bab["Kname"]=="kBP")][0]
    coeffs_RP = bab[ (bab["Xname"]=="BPRP") & (bab["Kname"]=="kRP")][0]
    # Return the extinction in the Gaia bands
    return A0 * ext_coeff(A0, BP-RP, coeffs_G), A0 * ext_coeff(A0, BP-RP, coeffs_BP), A0 * ext_coeff(A0, BP-RP, coeffs_RP)
