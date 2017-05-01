import csv

'''
Data Format:
1. Number of times pregnant.
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable (0 or 1) 
'''

'''
read_data() returns list of lists:
[['6', '148', '72', '35', '0', '33.6', '0.627', '50', '1'],...],
'''
def read_data():
    dataset = []
    with open('data.csv') as csvfile:
        rows = csv.reader(csvfile)
        dataset = list(rows)
    return dataset

