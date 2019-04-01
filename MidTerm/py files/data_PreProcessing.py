import pandas as pd


def filter_restaurant_businesses(file):
    df = pd.read_json(file, lines=True)
    filtered_df = df[(df.categories.str.contains('Restaurants') == True)
                     & (df.state.str.contains('IL') == True)]
    il_business = filtered_df.business_id.to_string(index=False)
    with open('./il_business_ids.txt', 'w') as outfile:
        outfile.write(il_business)


def filter_il_reviews(file):
    df = pd.read_json(file, lines=True)
    business_ids = df.business_id.to_list()
    df.set_index('business_id', inplace=True)
    reviews_df = pd.DataFrame()

    with open('./il_business_ids.txt', 'r') as infile:
        for b_id in infile:
            b_id = b_id.strip()
            if b_id in business_ids:
                data = df.loc[[b_id]]
                reviews_df = reviews_df.append(data)
    with open('./il_reviews.csv', 'a') as outfile:
        reviews_df.to_csv(outfile, header=False)
