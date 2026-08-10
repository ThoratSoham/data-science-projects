import pandas as pd
import numpy as np
import pandas as pd
df = pd.read_csv("collage_list.txt", sep="\t", skiprows=1)
print(df.head())
my_rank = int(input("Enter your rank: "))
my_category = input("Enter your category: ").upper()
my_gender = input("Enter your gender: (Male) or (Female)").upper()

# Rank and category filter
rank_category_condition = (pd.to_numeric(df["Closing Rank"], errors="coerce") >= my_rank) & \
                          (df["Seat Type"].str.upper() == my_category)

# Gender filter (robust)
if my_gender == "MALE":
    gender_condition = df["Gender"].str.upper().str.contains("GENDER-NEUTRAL")
elif my_gender == "FEMALE":
    gender_condition = df["Gender"].str.upper().str.contains("FEMALE-ONLY")
else:
    gender_condition = df["Gender"].str.upper().str.contains("GENDER-NEUTRAL")  # fallback

# Combine filters
eligible = df[rank_category_condition & gender_condition]

# Select and sort
c = eligible[["Institute", "Academic Program Name", "Closing Rank", "Seat Type", "Gender"]]
c = c.sort_values(by="Closing Rank", ascending=True)
print(c)
