# DataSet Creator
from . import variables_calculator
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def dataset(file, p1, p2, flavor1 = 0, flavor2 = 0, charge = 0, Signal = False, Noise = False, save_csv = False):
    """
    Function which creates a dataframe (and if selected a .csv file)
    based on the one proposed in Balazs Kegl, CecileGermain, ChallengeAdmin, 
    ClaireAdam, David Rousseau, Djabbz, fradav, Glen Cowan, Isabelle, and joycenv. 
    Higgs Boson Machine Learning Challenge. 
    https://kaggle.com/competitions/higgs-boson, 2014. Kaggle.

    This function creates either a noise or a signal Dataset. It is the same model
    for both.

    By default it is asigned not to put a label for the kind of data. This can be 
    change by either putting Signal = True or Noise = True. For signal events we
    asigne a 1 and for noise events we asigne 0 label. 
    """
    if Signal == False and Noise == False:

        inv_mass = variables_calculator.inv_m(file, p1 = p1, p2 = p2, flavor1 = flavor1, flavor2 = flavor2, charge = charge)

        PRI_leading_and_subleading_jets = variables_calculator.leading_n_subleading_jets(file)
        PRI_jet_num = variables_calculator.PRI_jet_num(file)
        PRI_jet_all = variables_calculator.PRI_jet_all_pt(file)

        DER_prodeta_jet_jet = variables_calculator.DER_prodeta_jet_jet(file, flavor1 = flavor1, flavor2 = flavor2, charge = charge)
        DER_deltaeta_jet_jet = variables_calculator.DER_deltaeta_jet_jet(file, j1 = p1, j2 = p2, flavor1 = flavor1, flavor2 = flavor2, charge = charge)

        met = variables_calculator.met(file)

        jets_data = pd.merge(PRI_jet_all, PRI_jet_num, on = "event_1")
        jets_data = pd.merge(jets_data, PRI_leading_and_subleading_jets, on = "event_1")
        jets_data = pd.merge(jets_data, met, on = "event_1")
        jets_data = pd.merge(jets_data, DER_prodeta_jet_jet, on = "event_1")
        jets_data = pd.merge(jets_data, DER_deltaeta_jet_jet, on = "event_1")
        jets_data = pd.merge(jets_data, inv_mass, on = "event_1")

        muon_data = variables_calculator.inv_m(file, "muon", "muon")
        electron_data = variables_calculator.inv_m(file, "electron", "electron")
        lepton_data = pd.concat([muon_data, electron_data])

        lepton_data = lepton_data[["event_1", "DER_INV_m"]].copy()
        lepton_data.columns = ["event_1", "DER_mass_lep"]

        df = pd.merge(jets_data, lepton_data, on = "event_1")
        df = df.drop(columns = ["event_1"])

    elif Signal == True and Noise == False:

        inv_mass = variables_calculator.inv_m(file, p1 = p1, p2 = p2, flavor1 = flavor1, flavor2 = flavor2, charge = charge)

        PRI_leading_and_subleading_jets = variables_calculator.leading_n_subleading_jets(file)
        PRI_jet_num = variables_calculator.PRI_jet_num(file)
        PRI_jet_all = variables_calculator.PRI_jet_all_pt(file)

        DER_prodeta_jet_jet = variables_calculator.DER_prodeta_jet_jet(file, flavor1 = flavor1, flavor2 = flavor2, charge = charge)
        DER_deltaeta_jet_jet = variables_calculator.DER_deltaeta_jet_jet(file, j1 = p1, j2 = p2, flavor1 = flavor1, flavor2 = flavor2, charge = charge)

        met = variables_calculator.met(file)

        jets_data = pd.merge(PRI_jet_all, PRI_jet_num, on = "event_1")
        jets_data = pd.merge(jets_data, PRI_leading_and_subleading_jets, on = "event_1")
        jets_data = pd.merge(jets_data, met, on = "event_1")
        jets_data = pd.merge(jets_data, DER_prodeta_jet_jet, on = "event_1")
        jets_data = pd.merge(jets_data, DER_deltaeta_jet_jet, on = "event_1")
        jets_data = pd.merge(jets_data, inv_mass, on = "event_1")

        jets_data["label"] = np.full(len(jets_data), 1)

        muon_data = variables_calculator.inv_m(file, "muon", "muon")
        electron_data = variables_calculator.inv_m(file, "electron", "electron")
        lepton_data = pd.concat([muon_data, electron_data])

        lepton_data = lepton_data[["event_1", "DER_INV_m"]].copy()
        lepton_data.columns = ["event_1", "DER_mass_lep"]

        df = pd.merge(jets_data, lepton_data, on = "event_1")
        df = df.drop(columns = ["event_1"])

    elif Signal == False and Noise == True:

        inv_mass = variables_calculator.inv_m(file, p1 = p1, p2 = p2, flavor1 = flavor1, flavor2 = flavor2, charge = charge)

        PRI_leading_and_subleading_jets = variables_calculator.leading_n_subleading_jets(file)
        PRI_jet_num = variables_calculator.PRI_jet_num(file)
        PRI_jet_all = variables_calculator.PRI_jet_all_pt(file)

        DER_prodeta_jet_jet = variables_calculator.DER_prodeta_jet_jet(file, flavor1 = flavor1, flavor2 = flavor2)
        DER_deltaeta_jet_jet = variables_calculator.DER_deltaeta_jet_jet(file, j1 = p1, j2 = p2, flavor1 = flavor1, flavor2 = flavor2, charge = charge)

        met = variables_calculator.met(file)

        jets_data = pd.merge(PRI_jet_all, PRI_jet_num, on = "event_1")
        jets_data = pd.merge(jets_data, PRI_leading_and_subleading_jets, on = "event_1")
        jets_data = pd.merge(jets_data, met, on = "event_1")
        jets_data = pd.merge(jets_data, DER_prodeta_jet_jet, on = "event_1")
        jets_data = pd.merge(jets_data, DER_deltaeta_jet_jet, on = "event_1")
        jets_data = pd.merge(jets_data, inv_mass, on = "event_1")

        jets_data["label"] = np.full(len(jets_data), 0)

        muon_data = variables_calculator.inv_m(file, "muon", "muon")
        electron_data = variables_calculator.inv_m(file, "electron", "electron")
        lepton_data = pd.concat([muon_data, electron_data])

        lepton_data = lepton_data[["event_1", "DER_INV_m"]].copy()
        lepton_data.columns = ["event_1", "DER_mass_lep"]

        df = pd.merge(jets_data, lepton_data, on = "event_1")
        df = df.drop(columns = ["event_1"])

    else:
        print("Error")

    return df

def model_sets(signal_df, noise_df, test = False, weight_classes = False, train_per = 0.5, val_per = 0.5,
               test_per = 0., esc = 1., random_state = 45):
    """
    This program creates the sets which will be used to train a Signal and Noise ANN classificator model.
    By default it is set not no consider a test subset. You can configurate to create it by setting test = True.
    The percentage for each subset is configured by default to be:
    - 50% training data
    - 50% validation data
    - 0 % test data (since it is not considered)
    The user can change this values. 
    The signal_df and noise_df correspond to Pandas DataFrames the user has to create using the dataset() 
    function from this program.
    The value of esc is the one used to escalate the size of the signal set since I noticed the noise sets
    could be in some ocassions really small and the ResNet ANN models can have troubles classificating.
    By default esc = 1 but it can be changed by the user.
    """
    if weight_classes == True:

        if test == False:
            bg_sample1 = noise_df.sample(frac = train_per, random_state = random_state)
            bg_sample2 = noise_df.sample(frac = val_per, random_state = random_state)

            sg_sample1 = signal_df.sample(n = len(bg_sample1)*esc, random_state = random_state)
            sg_sample2 = signal_df.sample(n = len(bg_sample2)*esc, random_state = random_state)

            train = pd.concat([sg_sample1, bg_sample1])
            val = pd.concat([sg_sample2, bg_sample2])

            train = train.sample(frac = 1, random_state = random_state).reset_index(drop = True)
            val = val.sample(frac = 1, random_state = random_state).reset_index(drop = True)

            X_train, X_val = train.drop(columns = ["label"]), val.drop(columns = ["label"])
            y_train, y_val = train["label"], val["label"]

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            weight_signal = 1.0 / train["label"].value_counts()[0]
            weight_background = 1.0 / train["label"].value_counts()[1]
            class_weight = {0: weight_background, 1: weight_signal}

            print("Train size:", len(X_train))
            print("Validation size:", len(X_val))

            return X_train, X_val, y_train, y_val, class_weight
        
        if test == True:
            bg_sample1 = noise_df.sample(frac = train_per, random_state = random_state)
            bg_sample2 = noise_df.sample(frac = val_per, random_state = random_state)
            bg_sample3 = noise_df.sample(frac = test_per, random_state = random_state)

            sg_sample1 = signal_df.sample(n = len(bg_sample1)*esc, random_state = 4)
            sg_sample2 = signal_df.sample(n = len(bg_sample2)*esc, random_state = 4)
            sg_sample3 = signal_df.sample(n = len(bg_sample3)*esc, random_state = 4)

            train = pd.concat([sg_sample1, bg_sample1])
            test = pd.concat([sg_sample2, bg_sample2])
            val = pd.concat([sg_sample3, bg_sample3])

            train = train.sample(frac = 1, random_state = random_state).reset_index(drop = True)
            test = test.sample(frac = 1, random_state = random_state).reset_index(drop = True)
            val = val.sample(frac = 1, random_state = random_state).reset_index(drop = True)

            X_train, X_test, X_val = train.drop(columns = ["label"]), test.drop(columns = ["label"]), val.drop(columns = ["label"])
            y_train, y_test, y_val = train["label"], test["label"], val["label"]

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            weight_signal = 1.0 / train["label"].value_counts()[0]
            weight_background = 1.0 / train["label"].value_counts()[1]
            class_weight = {0: weight_background, 1: weight_signal}

            print("Train size:", len(X_train))
            print("Validation size:", len(X_val))
            print("Test size:", len(X_test))

            return X_train, X_val, X_test, y_train, y_val, y_test, class_weight
    else:
        if test == False:
            bg_sample1 = noise_df.sample(frac = train_per, random_state = random_state)
            bg_sample2 = noise_df.sample(frac = val_per, random_state = random_state)

            sg_sample1 = signal_df.sample(n = len(bg_sample1)*esc, random_state = random_state)
            sg_sample2 = signal_df.sample(n = len(bg_sample2)*esc, random_state = random_state)

            train = pd.concat([sg_sample1, bg_sample1])
            val = pd.concat([sg_sample2, bg_sample2])

            train = train.sample(frac = 1, random_state = random_state).reset_index(drop = True)
            val = val.sample(frac = 1, random_state = random_state).reset_index(drop = True)

            X_train, X_val = train.drop(columns = ["label"]), val.drop(columns = ["label"])
            y_train, y_val = train["label"], val["label"]

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            print("Train size:", len(X_train))
            print("Validation size:", len(X_val))

            return X_train, X_val, y_train, y_val
    
        if test == True:
            bg_sample1 = noise_df.sample(frac = train_per, random_state = random_state)
            bg_sample2 = noise_df.sample(frac = val_per, random_state = random_state)
            bg_sample3 = noise_df.sample(frac = test_per, random_state = random_state)

            sg_sample1 = signal_df.sample(n = len(bg_sample1)*esc, random_state = 4)
            sg_sample2 = signal_df.sample(n = len(bg_sample2)*esc, random_state = 4)
            sg_sample3 = signal_df.sample(n = len(bg_sample3)*esc, random_state = 4)

            train = pd.concat([sg_sample1, bg_sample1])
            test = pd.concat([sg_sample2, bg_sample2])
            val = pd.concat([sg_sample3, bg_sample3])

            train = train.sample(frac = 1, random_state = random_state).reset_index(drop = True)
            test = test.sample(frac = 1, random_state = random_state).reset_index(drop = True)
            val = val.sample(frac = 1, random_state = random_state).reset_index(drop = True)

            X_train, X_test, X_val = train.drop(columns = ["label"]), test.drop(columns = ["label"]), val.drop(columns = ["label"])
            y_train, y_test, y_val = train["label"], test["label"], val["label"]

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            print("Train size:", len(X_train))
            print("Validation size:", len(X_val))
            print("Test size:", len(X_test))

            return X_train, X_val, X_test, y_train, y_val, y_test
        
def corr(df, cmap = "coolwarm", method = "pearson", fig_size = (20,20), droplabel = True):
    """
    Function to see the correlation matrix between variables.
    Dropping the label for signal o noise event is set by default with True. This can be
    changed by the user.
    """

    if droplabel == True:
        correlation_matrix = df.drop(columns = ["label"]).corr(method = method)
        plt.figure(figsize = fig_size)
        sns.heatmap(correlation_matrix, annot = True, cmap = cmap, fmt = ".2f")
        plt.title("Correlation Heatmap")
        plt.show()
    else: 
        correlation_matrix = df.corr(method = method)
        plt.figure(figsize = fig_size)
        sns.heatmap(correlation_matrix, annot = True, cmap = cmap, fmt = ".2f")
        plt.title("Correlation Heatmap")
        plt.show()

        


