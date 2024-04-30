import pandas as pd

carbon_pricing_data = pd.read_excel('./ggdp_data/world_bank_carbon_pricing.xlsx', header=1, sheet_name='Compliance_Price', na_values='-')
carbon_pricing_data.head()

eu_ets = carbon_pricing_data.loc['Name of the initiative' == 'EU ETS']
