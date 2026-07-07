# Cooling for Comfort, Warming the World

**Residential and office cooling and its environmental impacts in The Hague, the Netherlands.**

This site documents the model behind my MSc Industrial Ecology thesis (joint degree of Leiden
University and TU Delft). It estimates how much cooling the building stock of The Hague needs,
and what that cooling costs in electricity, greenhouse-gas emissions and material use, under
current conditions and across future scenarios for 2030 and 2050.

The pipeline combines three layers:

1. **Geospatial data** — building footprints and attributes from the Dutch BAG
   (*Basisregistratie Adressen en Gebouwen*), processed into building archetypes
   ([`geometric`](api/geometric.md), [`data_handling`](api/data_handling.md)).
2. **Thermodynamic modelling** — an hourly heat-balance model (transmission, infiltration,
   ventilation, solar gains and internal loads) driven by KNMI weather data
   ([`thermodynamic`](api/thermodynamic.md), [`time_series`](api/time_series.md)).
3. **Environmental impact assessment** — life-cycle-based impacts (climate change, abiotic
   resource depletion, crustal scarcity) from both the operational energy and the cooling
   equipment itself ([`environmental`](api/environmental.md)).

![Cooling demand and emissions across scenarios](scenario_overview.png)

## Key findings

- Offices occupy only **13%** of the floor area but account for **34%** of current cooling
  demand and **65%** of cooling-related greenhouse-gas emissions.
- An estimated **77%** of cooling demand is currently unmet, and that gap falls hardest on
  economically disadvantaged neighbourhoods.
- Under a business-as-usual 2050 scenario, cooling energy demand roughly **doubles** relative
  to today.

The full method and discussion are in the
[thesis](https://repository.tudelft.nl/record/uuid:32222863-536f-464a-b8c6-6c2283a7249a), and
the source lives on [GitHub](https://github.com/simonvanlierde/msc-thesis-ie). Use the **API
reference** in the navigation for the documented model functions.
