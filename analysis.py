import matplotlib.pyplot as plt
import numpy
import pandas
import statsmodels.api as sm # type: ignore
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob # type: ignore
from tqdm import tqdm

def calculate_polarity() -> None:
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    pair_data = pandas.read_csv("results/pair_data.csv", header = [0], index_col = [0])
    single_data = pandas.read_csv("results/single_data.csv", header = [0], index_col = [0])

    pair_data['polarity'] = pandas.Series(dtype = "float")
    single_data['polarity'] = pandas.Series(dtype = "float")

    for i in tqdm(range(0, len(pair_data))):
        doc = nlp(pair_data['answer'][i])
        pair_data.loc[i, 'polarity'] = doc._.blob.polarity

    for i in tqdm(range(0, len(single_data))):
        doc = nlp(single_data['answer'][i])
        single_data.loc[i, 'polarity'] = doc._.blob.polarity

    pair_data.to_csv("results/new_pair_data.csv")
    single_data.to_csv("results/new_single_data.csv")

def calculate_bias_percentages_by_type():
    pair_data = pandas.read_csv("results/new_pair_data.csv", header = [0], index_col = [0])
    single_data = pandas.read_csv("results/single_data.csv", header = [0], index_col = [0])
    
    biased_pair_data = pair_data.drop(pair_data[pair_data['biased'] == "False"].index)
    biased_single_data = single_data.drop(single_data[single_data['biased'] == False].index)
    
    types = set(pair_data['type'])
    types.update(single_data['type'])
    
    bias_percentages = {}
    
    for type in types:
        biased = len(biased_pair_data[biased_pair_data['type'] == type]) + len(biased_single_data[biased_single_data['type'] == type])
        total = len(pair_data[pair_data['type'] == type]) + len(single_data[single_data['type'] == type])
        
        bias_percentages[type] = biased / total
    
    return bias_percentages

def predict_pair_bias():
    pair_data = pandas.read_csv("results/new_pair_data.csv", header = [0], index_col = [0])
    pair_data['biased'] = pair_data['biased'].map({'False': False})
    pair_data['biased'] = pair_data['biased'].astype(bool).fillna(True)
    
    #x = pair_data[['category', 'group_x', 'group_y', 'type', 'polarity']].to_numpy()
    x = numpy.absolute(pair_data[['polarity']].to_numpy())
    y = pair_data[['biased']].to_numpy()
    x = sm.add_constant(x)
    
    model = sm.Logit(y, x)
    fit = model.fit(method='newton')
    predictions = fit.predict(x)
    plt.scatter(x[:, 1], y, label='Data', color='blue')
    plt.plot(x, predictions, color='red', label='Logistic Regression Curve')
    plt.xlabel('Polarity')
    plt.ylabel('Biased (True/False)')
    plt.title('Logistic Regression: Polarity vs Biased')
    plt.legend()
    plt.show()
    
    """ plt.scatter(x, y)
    plt.plot(predictions, x, c = "red")
    plt.show() """

def main():
    # calculate_polarity()
    # print(calculate_bias_percentages_by_type())
    predict_pair_bias()

if __name__ == "__main__":
    main()