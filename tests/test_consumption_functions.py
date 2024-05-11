import unittest
from unittest import TestCase

from consumption_functions import get_consumption_cost, calculate_energy_cost

import pandas as pd

hh_data = {
    'currency': ['EUR', 'EUR', 'EUR', 'EUR'],
    'tax_tier': [1000, 1000, 2000, 2000],
    'year': [2003, 2004, 2003, 2004],
    'price': [50, 55, 60, 65]
}

nh_data = {
    'currency': ['EUR', 'EUR', 'EUR', 'EUR'],
    'tax_tier': [1000, 1000, 2000, 2000],
    'year': [2003, 2004, 2003, 2004],
    'price': [25, 30, 35, 40]
}

consumption_data = {
    'year': [2003, 2004],
    'hh_consumption': [1, 2],
    'nh_consumption': [2, 1]
}


class ElectricityTotalCostTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hh_elec_df = pd.DataFrame(hh_data)
        cls.nh_elec_df = pd.DataFrame(nh_data)
        cls.consumption_df = pd.DataFrame(consumption_data)

    def test_get_consumption_cost(self):
        hh_cost_1 = get_consumption_cost(elec_consumption=500, currency='EUR', year=2003, elec_price=self.hh_elec_df)
        self.assertEquals(hh_cost_1, 500 * 50)
        hh_cost_2 = get_consumption_cost(elec_consumption=1500, currency='EUR', year=2003, elec_price=self.hh_elec_df)
        self.assertEquals(hh_cost_2, 1500 * 60)

    def test_calculate_energy_cost(self):
        cost_2003 = calculate_energy_cost(wte=2700,
                                          year=2003,
                                          currency='EUR',
                                          elec_consumption=self.consumption_df,
                                          hh_elec=self.hh_elec_df,
                                          nh_elec=self.nh_elec_df)
        #hh tax tier up to 1000 is 50
        #nh tax tier from 1000 to 2000 is 25
        self.assertEquals(cost_2003, 900 * 50 + 1800 * 35)
        cost_2004 = calculate_energy_cost(wte=2700,
                                          year=2004,
                                          currency='EUR',
                                          elec_consumption=self.consumption_df,
                                          hh_elec=self.hh_elec_df,
                                          nh_elec=self.nh_elec_df)
        #hh tax tier in 2004  from 1000 to 20000 is 65
        #nh tax tier up in 2004 to 1000 is 30
        self.assertEquals(cost_2004, 1800 * 65 + 900 * 30)


unittest.main(argv=[''], verbosity=2, exit=False)
