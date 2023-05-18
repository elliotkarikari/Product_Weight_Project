# Product_Weight_Project
 LIDA Project 2. Standardised Products Database

Store product data is found in multiple location and/or often contains missing weight information for unpackaged and package products. We need to produce a generally accepted standard product weight data to use in the absense of retailer-specific data (Sold weights and Portion weights) in order to explore varying compontent of human interaction with retail purchases such as understanding the carbon footprint of a persons grocery shop or calculating the total number of calories from a persons basket shop. 



# Aim
**Goal 1**: Produce a product list and meta-data documentation of info sources and decisions made.


# Think about
* What methods of imputation could be used? 
* How can we match retail data to our generic product list? E.g. fuzzy matching? What rules should we use to find the best match? 
* What level of granularity should we go to? E.g. ‘crisps’ individual or share bag. How to incorporate options for standard large/small portion sizes. 
* Which products should be included? E.g. based on McCance & Widdowson’s composition of food integrated dataset [https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid](https://www.gov.uk/government/publications/composition-of-foods-integrated-dataset-cofid)
* What is the best to measure quantity of products


# Decisions Made
* Working with data in Alphabetic Order. Building a workflow. Once this is clear, hopefully automate this if possible. 
* I grouped data by McCance, Widdowson Group (2 or 3 letter code is assigned to every food) and counted the Foods associated with each group. Some grouping had one food so I didn't have to do much. Others have several foods. 
* I started reducing product list looking for it most basic form. I did this by googling some of the products and looking them up on the Tesco website. Also If the product was derived from a raw product (looking at its description) it was removed. 
* Once removed I added an extra column (Super Group) which is the larger group the product falls in from the McCance and Widdowson table. I'm doing this to create an extra layer (larger) for which searches can be made. The plan is to ideally have these food fall within the eat well guide groupings. With the McCance, Widdowson reduced table under it. 
* Join all tables together. 


