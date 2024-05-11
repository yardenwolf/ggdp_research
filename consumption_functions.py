import pandas as pd
from numpy import nan


def get_consumption_cost(elec_consumption: float, currency: str, year: int, elec_price: pd.DataFrame) -> float:
    currency_year_df = elec_price[
        (elec_price['currency'] == currency)
        & (elec_price['year'] == year)
        ]
    #looking for greatest tax_tier s.t. tax_tier>consumption && tax_tier exists in data (not all tax tiers exist for all countries)
    if currency_year_df.empty:
        return nan
    existing_tax_tiers = list(currency_year_df['tax_tier'].unique())
    existing_tax_tiers.sort()
    fitting_tax_tier = None
    for curr_tax_tier in existing_tax_tiers:
        if elec_consumption < curr_tax_tier:
            fitting_tax_tier = curr_tax_tier
            break
    if fitting_tax_tier is None:
        #we couldn't find a tax tier that was greater, so we take the biggest one we have to complete the data
        fitting_tax_tier = existing_tax_tiers[-1]

    tax_tier = currency_year_df[currency_year_df['tax_tier'] == fitting_tax_tier]
    tax_tier = tax_tier.squeeze(axis=0)
    return elec_consumption * tax_tier['price']


def calculate_energy_cost(wte: float, year: int, currency: str, elec_consumption: pd.DataFrame,
                          hh_elec: pd.DataFrame, nh_elec: pd.DataFrame) -> float:
    """
    :param wte: waste to energy
    :param elec_consumption:
    :param hh_elec_price:
    :param nh_elec_price:
    :return:
    """
    #need to check that we have all the data to actually calculate this
    current_consumption = elec_consumption[elec_consumption['year'] == year]
    if current_consumption.empty:
        return nan
    hh_consumption = current_consumption['hh_consumption'].iloc[0]
    nh_consumption = current_consumption['nh_consumption'].iloc[0]
    hh_to_total = hh_consumption / (hh_consumption + nh_consumption)
    #the absolute part of the total wte. we do this to calculate separately the price of hh and nh elec prices
    current_hh_consumption = wte * hh_to_total
    current_nh_consumption = wte - current_hh_consumption

    hh_consumption_cost = get_consumption_cost(current_hh_consumption, currency=currency, year=year, elec_price=hh_elec)
    nh_consumption_cost = get_consumption_cost(current_nh_consumption, currency=currency, year=year, elec_price=nh_elec)
    return hh_consumption_cost + nh_consumption_cost
