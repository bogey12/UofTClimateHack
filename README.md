# UofTClimateHack

This repository outlines the solutions and code behind the University of Toronto Team's Submission for 2022 [ClimateHack](https://climatehack.ai/).

# Background
Currently, UK energy grids use a combination of both solar and natural gas sources to generate power. Due to a lack of accurate solar forecasting, natural gas turbines are on all the time, with solar power stored as excess in reserves. If they had more accurate solar forecasting algorithms, they could turn on/off natural gas sources when solar production is adequate. 

This would allow them to minimize the use of standby gas turbines, potentially leading to a substantial reduction in carbon emissions of up to 100 kilotonnes a year. While it is incredibly difficult to accurately predict the climate impact, a rough estimate suggests that better solar power forecasts if deployed worldwide could reduce global carbon emissions by about 100 million tonnes of CO2 a year by 2030.

# Challenge
We had ~1.5 months as individuals, and 1 week as a team to apply cutting-edge machine learning techniques in order to develop the best satellite imagery prediction algorithm for use in solar photovolatic output forecasting. **The specific challenge:** from a series of 12 images covering a 128×128-pixel region cropped out of a series of much larger satellite images taken five minutes apart, accurately predict the next 24 images for the central 64×64-pixel area, corresponding to the next two hours of satellite imagery.
