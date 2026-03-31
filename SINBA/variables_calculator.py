import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inv_m(file, p1, p2, charge = 0, flavor1 = 0, flavor2 = 0, save_df = False, graph = False, mass = 0):
    file = uproot.open(file)
    tree = file["Delphes"]
    
    if flavor1 != 0 or flavor2 != 0:
        flavor = tree["Jet/Jet.Flavor"].array(library = "np")
    
    if p1 != p2:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            eta1 = tree["Muon/Muon.Eta"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            eta1 = tree["Electron/Electron.Eta"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            eta1 = tree["Photon/Photon.Eta"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet_1":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            eta1 = tree["Jet/Jet.Eta"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
        
        if p2 == "muon":
            pt2 = tree["Muon/Muon.PT"].array(library = "np")
            eta2 = tree["Muon/Muon.Eta"].array(library = "np")
            phi2 = tree["Muon/Muon.Phi"].array(library = "np")
            charge2 = tree["Muon/Muon.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "electron":
            pt2 = tree["Electron/Electron.PT"].array(library = "np")
            eta2 = tree["Electron/Electron.Eta"].array(library = "np")
            phi2 = tree["Electron/Electron.Phi"].array(library = "np")
            charge2 = tree["Electron/Electron.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "photon":
            pt2 = tree["Photon/Photon.PT"].array(library = "np")
            eta2 = tree["Photon/Photon.Eta"].array(library = "np")
            phi2 = tree["Photon/Photon.Phi"].array(library = "np")
            c2 = False
            
        elif p2 == "jet_2":
            pt2 = tree["Jet/Jet.PT"].array(library = "np")
            eta2 = tree["Jet/Jet.Eta"].array(library = "np")
            phi2 = tree["Jet/Jet.Phi"].array(library = "np")
            charge2 = tree["Jet/Jet.Charge"].array(library = "np")
            c2 = True
    else:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            eta1 = tree["Muon/Muon.Eta"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            eta1 = tree["Electron/Electron.Eta"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            eta1 = tree["Photon/Photon.Eta"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            eta1 = tree["Jet/Jet.Eta"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
        
    # The jet flavor is computed using the the Simple Δ𝑅
    # highest-flavor match and it is asigned as follows:
    #   gluon --> 21
    #   d --> 1
    #   u --> 2
    #   s --> 3
    #   c --> 4
    #   b --> 5
    
    # For different particles that aren't photons:
    if p1 != p2 and (c1 == True and c2 == True):
        
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        cha_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []
        cha_2 = []
        
        event_1 = []
        event_2 = []

        # Both kind of particles aren't jets
        if flavor1 == 0 and flavor2 == 0:
            
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0: # At least one of each
                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])
                    
                    if charge1[i][index1] + charge2[i][index2] == charge:
                        
                        pt_1.append(pt1[i][index1])
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                        cha_1.append(charge1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        eta_2.append(eta2[i][index2])
                        phi_2.append(phi2[i][index2])
                        cha_2.append(charge2[i][index2])
                            
                        event_1.append(i)
                        event_2.append(i)
            
        # For collisions with decayment particles being jets
        elif flavor1 != 0 or flavor2 !=0:

            # For both particles being jets with different flavor:
            if flavor1 != 0 and flavor2 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)
                    suma_2 = np.sum(flavor[i] == flavor2)
                    positions1 = []
                    positions2 = []
                    trials_pt1 = []
                    trials_pt2 = []

                    if suma_1 > 0 and suma_2 > 0:

                        for k in range(len(flavor[i])):
                            if flavor[i][k] == flavor1:
                                positions1.append(k)
                                trials_pt1.append(pt1[i][k]) 
                                
                            elif flavor[i][k] == flavor2:
                                positions2.append(k)
                                trials_pt2.append(pt2[i][k])

                        pt_max1 = trials_pt1.index(max(trials_pt1))
                        pt_max2 = trials_pt2.index(max(trials_pt2))

                        index1 = positions1[pt_max1]
                        index2 = positions2[pt_max2]

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                            cha_1.append(charge1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                            cha_2.append(charge2[i][index2])
                                
                            event_1.append(i)
                            event_2.append(i)             

            # For just the first particle being a jet:
            elif flavor1 != 0 and flavor2 == 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                            cha_1.append(charge1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                            cha_2.append(charge2[i][index2])
                                
                            event_1.append(i)
                            event_2.append(i)

            # For just the 2nd particle being a jet: 
            elif flavor1 == 0 and flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                            cha_1.append(charge1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                            cha_2.append(charge2[i][index2])
                                
                            event_1.append(i)
                            event_2.append(i)
             
        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
        cha_1 = np.array(cha_1)
            
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)
        cha_2 = np.array(cha_2)
             
        event_1 = np.array(event_1)
        event_2 = np.array(event_2)
        
        data = {
                "PRI_pt_1": pt_1,
                "PRI_pt_2": pt_2,
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "PRI_phi_1": phi_1,
                "PRI_phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 
            
    # For different particles with one kind of them are photons
    elif p1 != p2 and (c1 == False or c2 == False):
        
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []
        
        event_1 = []
        event_2 = []

        charge_12 = []

        # When none of them are jets so one kind are photons and the others
        # are some other particle
        if flavor1 == 0 and flavor2 == 0:
        
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0:

                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])

                    if c1 == True and charge1[i][index1] == charge:
                        charge_12.append(charge1[i][index1])

                        pt_1.append(pt1[i][index1])
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        eta_2.append(eta2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)
                        event_2.append(i)

                    elif c2 == True and charge2[i][index2] == charge:
                        charge_12.append(charge2[i][index2])

                        pt_1.append(pt1[i][index1])
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        eta_2.append(eta2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)
                        event_2.append(i)

        # If one of the particles is a jet and the other a photon:
        elif flavor1 != 0 or flavor2 != 0:
            # If the first kind are jets and the second are photons
            if flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] == charge:
                            charge_12.append(charge1[i][index1])

                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i,index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)
                            event_2.append(i)
            # If the first ones are photons and the second are jets
            elif flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge2[i][index2] == charge:
                            charge_12.append(charge2[i][index2])

                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)
                            event_2.append(i)

        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
        
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)
         
        event_1 = np.array(event_1)
        event_2 = np.array(event_2)

        charge_12 = np.array(charge_12)
        
        data = {
                "PRI_pt_1": pt_1,
                "PRI_pt_2": pt_2,
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "PRI_phi_1": phi_1,
                "PRI_phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # In the case of detections of particles from the same type.
    # Events with no photons:
    elif p1 == p2 and (c1 == True):
        
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        cha_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []
        cha_2 = []
        
        event_1 = []
        event_2 = []

        # Particles that aren't jets:
        if p1 != "jet":
            
            for i in range(events):
                if len(pt1[i]) > 1: # At least two particles
                    index1 = np.argmax(pt1[i])
                    trial_pt1 = pt1[i][index1]

                    pt1[i][index1] = 1e-1000

                    index2 = np.argmax(pt1[i])
                    trial_pt2 = pt1[i][index2]

                    if charge1[i][index1] + charge1[i][index2] == charge:
                        
                        pt_1.append(trial_pt1)
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                        cha_1.append(charge1[i][index1])
                            
                        pt_2.append(trial_pt2)
                        eta_2.append(eta1[i][index2])
                        phi_2.append(phi1[i][index2])
                        cha_2.append(charge1[i][index2])
                            
                        event_1.append(i)
                        event_2.append(i)

        # Events with jets:
        elif p1 == "jet":
            if flavor1 == flavor2 and flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 1: # At least two particles
                        index1 = np.argmax(pt1[i])
                        trial_pt1 = pt1[i][index1]

                        pt1[i][index1] = 1e-1000

                        index2 = np.argmax(pt1[i])
                        trial_pt2 = pt1[i][index2]

                        if charge1[i][index1] + charge1[i][index2] == charge:
                            
                            pt_1.append(trial_pt1)
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                            cha_1.append(charge1[i][index1])
                                
                            pt_2.append(trial_pt2)
                            eta_2.append(eta1[i][index2])
                            phi_2.append(phi1[i][index2])
                            cha_2.append(charge1[i][index2])
                                
                            event_1.append(i)
                            event_2.append(i)

        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
        cha_1 = np.array(cha_1)
            
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)
        cha_2 = np.array(cha_2)
             
        event_1 = np.array(event_1)
        event_2 = np.array(event_2)
        
        data = {
                "PRI_pt_1": pt_1,
                "PRI_pt_2": pt_2,
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "PRI_phi_1": phi_1,
                "PRI_phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # For photons
    elif p1 == p2 and (c1 == False):
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []

        cha_12 = []
        
        event_1 = []
        event_2 = []
            
        for i in range(events):
            if len(pt1[i]) > 1: # At least two particles
                index1 = np.argmax(pt1[i])
                trial_pt1 = pt1[i][index1]

                pt1[i][index1] = 1e-1000

                index2 = np.argmax(pt1[i])
                trial_pt2 = pt1[i][index2]
                        
                pt_1.append(trial_pt1)
                eta_1.append(eta1[i][index1])
                phi_1.append(phi1[i][index1])
                            
                pt_2.append(trial_pt2)
                eta_2.append(eta1[i][index2])
                phi_2.append(phi1[i][index2])

                cha_12.append(0)
                            
                event_1.append(i)
                event_2.append(i)

        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
            
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)

        cha_12 = np.array(cha_12)
             
        event_1 = np.array(event_1)
        event_2 = np.array(event_2)
        
        data = {
                "PRI_pt_1": pt_1,
                "PRI_pt_2": pt_2,
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "PRI_phi_1": phi_1,
                "PRI_phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # Compute the momentum components
    data["p1_x"] = data["PRI_pt_1"]*np.cos(data["PRI_phi_1"])
    data["p1_y"] = data["PRI_pt_1"]*np.sin(data["PRI_phi_1"])
    data["p1_z"] = data["PRI_pt_1"]*np.sinh(data["PRI_eta_1"])

    data["p2_x"] = data["PRI_pt_2"]*np.cos(data["PRI_phi_2"])
    data["p2_y"] = data["PRI_pt_2"]*np.sin(data["PRI_phi_2"])
    data["p2_z"] = data["PRI_pt_2"]*np.sinh(data["PRI_eta_2"])

    # Then we can calculate the invariant mass

    data["DER_INV_m"] = np.sqrt((np.sqrt(data["p1_x"]**2 + data["p1_y"]**2 + data["p1_z"]**2) + 
                                np.sqrt(data["p2_x"]**2 + data["p2_y"]**2 + data["p2_z"]**2))**2 - 
                                (data["p1_x"] + data["p2_x"])**2 - (data["p1_y"] + data["p2_y"])**2 -
                                (data["p1_z"] + data["p2_z"])**2 )
    
    data = data.drop(columns = ["p1_x", "p1_y", "p1_z", "p2_x", "p2_y", "p2_z"])

    # To save the dataframe as a .csv file:
    if save_df == True:
        data.to_csv("Invariant-Mass.csv")
    
    if graph == True and mass == 0:
        sns.histplot(data["inv_m"], bins = 1000, kde = False)
        plt.xlabel(r"$m_{inv}$ (GeV)")
        plt.ylabel("Number of events")
        plt.title(r"Event Invariant mass")
        plt.grid(True, alpha = 0.3)
        plt.show()

    elif graph == True and mass != 0:
        sns.histplot(data["inv_m"], bins = 1000, kde = False)
        plt.axvline(x = mass, color = "red", linestyle = ':', label = rf"$m_{{inv}}$ = {mass}")
        plt.xlabel(r"$m_{inv}$ (GeV)")
        plt.ylabel("Number of events")
        plt.title(r"Event Invariant mass")
        plt.legend()
        plt.grid(True, alpha = 0.3)
        plt.show()
    return data

def pseudorapidity_separation(file, p1, p2, flavor1 = 0, flavor2 = 0, charge = 0, save_df = False):
    file = uproot.open(file)
    tree = file["Delphes"]
    
    if flavor1 != 0 or flavor2 != 0:
        flavor = tree["Jet/Jet.Flavor"].array(library = "np")
    
    if p1 != p2:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            eta1 = tree["Muon/Muon.Eta"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            eta1 = tree["Electron/Electron.Eta"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            eta1 = tree["Photon/Photon.Eta"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet_1":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            eta1 = tree["Jet/Jet.Eta"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
        
        if p2 == "muon":
            pt2 = tree["Muon/Muon.PT"].array(library = "np")
            eta2 = tree["Muon/Muon.Eta"].array(library = "np")
            phi2 = tree["Muon/Muon.Phi"].array(library = "np")
            charge2 = tree["Muon/Muon.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "electron":
            pt2 = tree["Electron/Electron.PT"].array(library = "np")
            eta2 = tree["Electron/Electron.Eta"].array(library = "np")
            phi2 = tree["Electron/Electron.Phi"].array(library = "np")
            charge2 = tree["Electron/Electron.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "photon":
            pt2 = tree["Photon/Photon.PT"].array(library = "np")
            eta2 = tree["Photon/Photon.Eta"].array(library = "np")
            phi2 = tree["Photon/Photon.Phi"].array(library = "np")
            c2 = False
            
        elif p2 == "jet_2":
            pt2 = tree["Jet/Jet.PT"].array(library = "np")
            eta2 = tree["Jet/Jet.Eta"].array(library = "np")
            phi2 = tree["Jet/Jet.Phi"].array(library = "np")
            charge2 = tree["Jet/Jet.Charge"].array(library = "np")
            c2 = True
    else:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            eta1 = tree["Muon/Muon.Eta"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            eta1 = tree["Electron/Electron.Eta"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            eta1 = tree["Photon/Photon.Eta"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            eta1 = tree["Jet/Jet.Eta"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
        
    # The jet flavor is computed using the the Simple Δ𝑅
    # highest-flavor match and it is asigned as follows:
    #   gluon --> 21
    #   d --> 1
    #   u --> 2
    #   s --> 3
    #   c --> 4
    #   b --> 5
    
    # For different particles that aren't photons:
    if p1 != p2 and (c1 == True and c2 == True):
        
        events = len(pt1)
        
        eta_1 = []
        eta_2 = []
        
        event_1 = []

        # Both kind of particles aren't jets
        if flavor1 == 0 and flavor2 == 0:
            
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0: # At least one of each
                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])
                    
                    if charge1[i][index1] + charge2[i][index2] == charge:
                        
                        eta_1.append(eta1[i][index1])  
                        eta_2.append(eta2[i][index2])
                            
                        event_1.append(i)
            
        # For collisions with decayment particles being jets
        elif flavor1 != 0 or flavor2 !=0:

            # For both particles being jets with different flavor:
            if flavor1 != 0 and flavor2 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)
                    suma_2 = np.sum(flavor[i] == flavor2)
                    positions1 = []
                    positions2 = []
                    trials_pt1 = []
                    trials_pt2 = []

                    if suma_1 > 0 and suma_2 > 0:

                        for k in range(len(flavor[i])):
                            if flavor[i][k] == flavor1:
                                positions1.append(k)
                                trials_pt1.append(pt1[i][k]) 
                                
                            elif flavor[i][k] == flavor2:
                                positions2.append(k)
                                trials_pt2.append(pt2[i][k])

                        pt_max1 = trials_pt1.index(max(trials_pt1))
                        pt_max2 = trials_pt2.index(max(trials_pt2))

                        index1 = positions1[pt_max1]
                        index2 = positions2[pt_max2]

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            eta_1.append(eta1[i][index1])
                            eta_2.append(eta2[i][index2])
                                
                            event_1.append(i)

            # For just the first particle being a jet:
            elif flavor1 != 0 and flavor2 == 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            eta_1.append(eta1[i][index1])
                            eta_2.append(eta2[i][index2])
                                
                            event_1.append(i)

            # For just the 2nd particle being a jet: 
            elif flavor1 == 0 and flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            eta_1.append(eta1[i][index1])                                
                            eta_2.append(eta2[i][index2])
                                
                            event_1.append(i)
             
        eta_1 = np.array(eta_1) 
        eta_2 = np.array(eta_2)
             
        event_1 = np.array(event_1)
        
        data = {
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 
            
    # For different particles with one kind of them are photons
    elif p1 != p2 and (c1 == False or c2 == False):
        
        events = len(pt1)
        
        eta_1 = []
        eta_2 = []
        
        event_1 = []


        # When none of them are jets so one kind are photons and the others
        # are some other particle
        if flavor1 == 0 and flavor2 == 0:
        
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0:

                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])

                    if c1 == True and charge1[i][index1] == charge:

                        eta_1.append(eta1[i][index1])
                        eta_2.append(eta2[i][index2])
                            
                        event_1.append(i)

                    elif c2 == True and charge2[i][index2] == charge:

                        eta_1.append(eta1[i][index1])
                        eta_2.append(eta2[i][index2])
                            
                        event_1.append(i)

        # If one of the particles is a jet and the other a photon:
        elif flavor1 != 0 or flavor2 != 0:
            # If the first kind are jets and the second are photons
            if flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] == charge:

                            eta_1.append(eta1[i][index1])
                            eta_2.append(eta2[i][index2])
                                
                            event_1.append(i)
            # If the first ones are photons and the second are jets
            elif flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge2[i][index2] == charge:

                            eta_1.append(eta1[i][index1])
                            eta_2.append(eta2[i][index2])
                                
                            event_1.append(i)

        eta_1 = np.array(eta_1)
        eta_2 = np.array(eta_2)
         
        event_1 = np.array(event_1)
        
        data = {
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # In the case of detections of particles from the same type.
    # Events with no photons:
    elif p1 == p2 and (c1 == True):
        
        events = len(pt1)
        
        eta_1 = []
        eta_2 = []
        
        event_1 = []

        # Particles that aren't jets:
        if p1 != "jet":
            
            for i in range(events):
                if len(pt1[i]) > 1: # At least two particles
                    index1 = np.argmax(pt1[i])
                    pt1[i][index1] = 1e-1000
                    index2 = np.argmax(pt1[i])

                    if charge1[i][index1] + charge1[i][index2] == charge:
                        
                        eta_1.append(eta1[i][index1])
                        eta_2.append(eta1[i][index2])
                            
                        event_1.append(i)

        # Events with jets:
        elif p1 == "jet":
            if flavor1 == flavor2 and flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 1: # At least two particles
                        index1 = np.argmax(pt1[i])
                        pt1[i][index1] = 1e-1000
                        index2 = np.argmax(pt1[i])

                        if charge1[i][index1] + charge1[i][index2] == charge:
                            
                            eta_1.append(eta1[i][index1])
                            eta_2.append(eta1[i][index2])
                                
                            event_1.append(i)

        eta_1 = np.array(eta_1)
        eta_2 = np.array(eta_2)
        
        event_1 = np.array(event_1)
        
        data = {
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # For photons
    elif p1 == p2 and (c1 == False):
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []

        cha_12 = []
        
        event_1 = []
        event_2 = []
            
        for i in range(events):
            if len(pt1[i]) > 1: # At least two particles
                index1 = np.argmax(pt1[i])
                pt1[i][index1] = 1e-1000
                index2 = np.argmax(pt1[i])
                        
                eta_1.append(eta1[i][index1])      
                eta_2.append(eta1[i][index2])
                            
                event_1.append(i)

        eta_1 = np.array(eta_1)  
        eta_2 = np.array(eta_2)    
        event_1 = np.array(event_1)
        
        data = {
                "PRI_eta_1": eta_1,
                "PRI_eta_2": eta_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    data["DER_pseudorapidity_separation"] = abs(data["PRI_eta_1"] - data["PRI_eta_2"])
    data = data.drop(columns = ["PRI_eta_1", "PRI_eta_2"])

    # To save the dataframe as a .csv file:
    if save_df == True:
        data.to_csv("Pseudorepidity-Separation.csv")
    
    return data

def trans_m(file, p1, p2, charge = 0, flavor1 = 0, flavor2 = 0, save_df = False, graph = False, mass = 0):

    file = uproot.open(file)
    tree = file["Delphes"]
    
    if flavor1 != 0 or flavor2 != 0:
        flavor = tree["Jet/Jet.Flavor"].array(library = "np")
    
    if p1 != p2:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            eta1 = tree["Muon/Muon.Eta"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            eta1 = tree["Electron/Electron.Eta"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            eta1 = tree["Photon/Photon.Eta"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet_1":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            eta1 = tree["Jet/Jet.Eta"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
        
        if p2 == "muon":
            pt2 = tree["Muon/Muon.PT"].array(library = "np")
            eta2 = tree["Muon/Muon.Eta"].array(library = "np")
            phi2 = tree["Muon/Muon.Phi"].array(library = "np")
            charge2 = tree["Muon/Muon.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "electron":
            pt2 = tree["Electron/Electron.PT"].array(library = "np")
            eta2 = tree["Electron/Electron.Eta"].array(library = "np")
            phi2 = tree["Electron/Electron.Phi"].array(library = "np")
            charge2 = tree["Electron/Electron.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "photon":
            pt2 = tree["Photon/Photon.PT"].array(library = "np")
            eta2 = tree["Photon/Photon.Eta"].array(library = "np")
            phi2 = tree["Photon/Photon.Phi"].array(library = "np")
            c2 = False
            
        elif p2 == "jet_2":
            pt2 = tree["Jet/Jet.PT"].array(library = "np")
            eta2 = tree["Jet/Jet.Eta"].array(library = "np")
            phi2 = tree["Jet/Jet.Phi"].array(library = "np")
            charge2 = tree["Jet/Jet.Charge"].array(library = "np")
            c2 = True
    else:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            eta1 = tree["Muon/Muon.Eta"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            eta1 = tree["Electron/Electron.Eta"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            eta1 = tree["Photon/Photon.Eta"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            eta1 = tree["Jet/Jet.Eta"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
        
    # The jet flavor is computed using the the Simple Δ𝑅
    # highest-flavor match and it is asigned as follows:
    #   gluon --> 21
    #   d --> 1
    #   u --> 2
    #   s --> 3
    #   c --> 4
    #   b --> 5
    
    # For different particles that aren't photons:
    if p1 != p2 and (c1 == True and c2 == True):
        
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []
        
        event_1 = []

        # Both kind of particles aren't jets
        if flavor1 == 0 and flavor2 == 0:
            
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0: # At least one of each
                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])
                    
                    if charge1[i][index1] + charge2[i][index2] == charge:
                        
                        pt_1.append(pt1[i][index1])
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        eta_2.append(eta2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)
            
        # For collisions with decayment particles being jets
        elif flavor1 != 0 or flavor2 !=0:

            # For both particles being jets with different flavor:
            if flavor1 != 0 and flavor2 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)
                    suma_2 = np.sum(flavor[i] == flavor2)
                    positions1 = []
                    positions2 = []
                    trials_pt1 = []
                    trials_pt2 = []

                    if suma_1 > 0 and suma_2 > 0:

                        for k in range(len(flavor[i])):
                            if flavor[i][k] == flavor1:
                                positions1.append(k)
                                trials_pt1.append(pt1[i][k]) 
                                
                            elif flavor[i][k] == flavor2:
                                positions2.append(k)
                                trials_pt2.append(pt2[i][k])

                        pt_max1 = trials_pt1.index(max(trials_pt1))
                        pt_max2 = trials_pt2.index(max(trials_pt2))

                        index1 = positions1[pt_max1]
                        index2 = positions2[pt_max2]

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)

            # For just the first particle being a jet:
            elif flavor1 != 0 and flavor2 == 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)

            # For just the 2nd particle being a jet: 
            elif flavor1 == 0 and flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)
             
        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
            
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)
             
        event_1 = np.array(event_1)
        
        data = {
                "pt_1": pt_1,
                "pt_2": pt_2,
                "eta_1": eta_1,
                "eta_2": eta_2,
                "phi_1": phi_1,
                "phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 
            
    # For different particles with one kind of them are photons
    elif p1 != p2 and (c1 == False or c2 == False):
        
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []
        
        event_1 = []

        # When none of them are jets so one kind are photons and the others
        # are some other particle
        if flavor1 == 0 and flavor2 == 0:
        
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0:

                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])

                    if c1 == True and charge1[i][index1] == charge:

                        pt_1.append(pt1[i][index1])
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        eta_2.append(eta2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)

                    elif c2 == True and charge2[i][index2] == charge:

                        pt_1.append(pt1[i][index1])
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        eta_2.append(eta2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)

        # If one of the particles is a jet and the other a photon:
        elif flavor1 != 0 or flavor2 != 0:
            # If the first kind are jets and the second are photons
            if flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] == charge:

                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i,index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)

            # If the first ones are photons and the second are jets
            elif flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge2[i][index2] == charge:

                            pt_1.append(pt1[i][index1])
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            eta_2.append(eta2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)

        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
        
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)
         
        event_1 = np.array(event_1)
        
        data = {
                "pt_1": pt_1,
                "pt_2": pt_2,
                "eta_1": eta_1,
                "eta_2": eta_2,
                "phi_1": phi_1,
                "phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # In the case of detections of particles from the same type.
    # Events with no photons:
    elif p1 == p2 and (c1 == True):
        
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []
        
        event_1 = []

        # Particles that aren't jets:
        if p1 != "jet":
            
            for i in range(events):
                if len(pt1[i]) > 1: # At least two particles
                    index1 = np.argmax(pt1[i])
                    trial_pt1 = pt1[i][index1]

                    pt1[i][index1] = 1e-1000

                    index2 = np.argmax(pt1[i])
                    trial_pt2 = pt1[i][index2]

                    if charge1[i][index1] + charge1[i][index2] == charge:
                        
                        pt_1.append(trial_pt1)
                        eta_1.append(eta1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(trial_pt2)
                        eta_2.append(eta1[i][index2])
                        phi_2.append(phi1[i][index2])
                            
                        event_1.append(i)

        # Events with jets:
        elif p1 == "jet":
            if flavor1 == flavor2 and flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 1: # At least two particles
                        index1 = np.argmax(pt1[i])
                        trial_pt1 = pt1[i][index1]

                        pt1[i][index1] = 1e-1000

                        index2 = np.argmax(pt1[i])
                        trial_pt2 = pt1[i][index2]

                        if charge1[i][index1] + charge1[i][index2] == charge:
                            
                            pt_1.append(trial_pt1)
                            eta_1.append(eta1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(trial_pt2)
                            eta_2.append(eta1[i][index2])
                            phi_2.append(phi1[i][index2])
                                
                            event_1.append(i)

        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
            
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)
             
        event_1 = np.array(event_1)
        
        data = {
                "pt_1": pt_1,
                "pt_2": pt_2,
                "eta_1": eta_1,
                "eta_2": eta_2,
                "phi_1": phi_1,
                "phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # For photons
    elif p1 == p2 and (c1 == False):
        events = len(pt1)
        
        pt_1 = []
        eta_1 = []
        phi_1 = []
        
        pt_2 = []
        eta_2 = []
        phi_2 = []
        
        event_1 = []
            
        for i in range(events):
            if len(pt1[i]) > 1: # At least two particles
                index1 = np.argmax(pt1[i])
                trial_pt1 = pt1[i][index1]

                pt1[i][index1] = 1e-1000

                index2 = np.argmax(pt1[i])
                trial_pt2 = pt1[i][index2]
                        
                pt_1.append(trial_pt1)
                eta_1.append(eta1[i][index1])
                phi_1.append(phi1[i][index1])
                            
                pt_2.append(trial_pt2)
                eta_2.append(eta1[i][index2])
                phi_2.append(phi1[i][index2])
                            
                event_1.append(i)

        pt_1 = np.array(pt_1)
        eta_1 = np.array(eta_1)
        phi_1 = np.array(phi_1)
            
        pt_2 = np.array(pt_2)
        eta_2 = np.array(eta_2)
        phi_2 = np.array(phi_2)
             
        event_1 = np.array(event_1)
        
        data = {
                "pt_1": pt_1,
                "pt_2": pt_2,
                "eta_1": eta_1,
                "eta_2": eta_2,
                "phi_1": phi_1,
                "phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # Compute the momentum components
    data["p1_x"] = data["pt_1"]*np.cos(data["phi_1"])
    data["p1_y"] = data["pt_1"]*np.sin(data["phi_1"])

    data["p2_x"] = data["pt_2"]*np.cos(data["phi_2"])
    data["p2_y"] = data["pt_2"]*np.sin(data["phi_2"])

    # Then we can calculate the transverse mass

    data["DER_trans_m"] = np.sqrt((np.sqrt(data["p1_x"]**2 + data["p1_y"]**2) + 
                                np.sqrt(data["p2_x"]**2 + data["p2_y"]**2))**2 - 
                                (data["p1_x"] + data["p2_x"])**2 - (data["p1_y"] + data["p2_y"])**2)
    
    # Create the corresponding dataframe
    data = data.drop(columns = ["pt_1", "pt_2","eta_1", "eta_2", "phi_1", "phi_2"])

    # To save the dataframe as a .csv file:
    if save_df == True:
        data.to_csv("Transverse-Mass.csv")
    
    if graph == True and mass == 0:
        sns.histplot(data["trans_m"], bins = 1000, kde = False)
        plt.xlabel(r"$m_{inv}$ (GeV)")
        plt.ylabel("Number of events")
        plt.title(r"Events Invariant mass")
        plt.grid(True, alpha = 0.3)
        plt.show()

    elif graph == True and mass != 0:
        sns.histplot(data["trans_m"], bins = 1000, kde = False)
        plt.axvline(x = mass, color = "red", linestyle = ':', label = rf"$m_{{tr}}$ = {mass}")
        plt.xlabel(r"$m_{inv}$ (GeV)")
        plt.ylabel("Number of events")
        plt.title(r"Events Transverse mass")
        plt.legend()
        plt.grid(True, alpha = 0.3)
        plt.show() 

    return data

def trans_momentum(file, p1, p2, charge, flavor1 = 0, flavor2 = 0, save_df = True):
    # The transverse momentum.
    file = uproot.open(file)
    tree = file["Delphes"]
    
    if flavor1 != 0 or flavor2 != 0:
        flavor = tree["Jet/Jet.Flavor"].array(library = "np")
    
    if p1 != p2:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet_1":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
        
        if p2 == "muon":
            pt2 = tree["Muon/Muon.PT"].array(library = "np")
            phi2 = tree["Muon/Muon.Phi"].array(library = "np")
            charge2 = tree["Muon/Muon.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "electron":
            pt2 = tree["Electron/Electron.PT"].array(library = "np")
            phi2 = tree["Electron/Electron.Phi"].array(library = "np")
            charge2 = tree["Electron/Electron.Charge"].array(library = "np")
            c2 = True
            
        elif p2 == "photon":
            pt2 = tree["Photon/Photon.PT"].array(library = "np")
            phi2 = tree["Photon/Photon.Phi"].array(library = "np")
            c2 = False
            
        elif p2 == "jet_2":
            pt2 = tree["Jet/Jet.PT"].array(library = "np")
            phi2 = tree["Jet/Jet.Phi"].array(library = "np")
            charge2 = tree["Jet/Jet.Charge"].array(library = "np")
            c2 = True
    else:
        if p1 == "muon":
            pt1 = tree["Muon/Muon.PT"].array(library = "np")
            phi1 = tree["Muon/Muon.Phi"].array(library = "np")
            charge1 = tree["Muon/Muon.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "electron":
            pt1 = tree["Electron/Electron.PT"].array(library = "np")
            phi1 = tree["Electron/Electron.Phi"].array(library = "np")
            charge1 = tree["Electron/Electron.Charge"].array(library = "np")
            c1 = True
            
        elif p1 == "photon":
            pt1 = tree["Photon/Photon.PT"].array(library = "np")
            phi1 = tree["Photon/Photon.Phi"].array(library = "np")
            c1 = False
            
        elif p1 == "jet":
            pt1 = tree["Jet/Jet.PT"].array(library = "np")
            phi1 = tree["Jet/Jet.Phi"].array(library = "np")
            charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            c1 = True
            
    # The jet flavor is computed using the the Simple Δ𝑅
    # highest-flavor match and it is asigned as follows:
    #   gluon --> 21
    #   d --> 1
    #   u --> 2
    #   s --> 3
    #   c --> 4
    #   b --> 5
    
    # For different particles that aren't photons:
    if p1 != p2 and (c1 == True and c2 == True):
        
        events = len(pt1)
        
        pt_1 = []
        phi_1 = []
        
        pt_2 = []
        phi_2 = []
        
        event_1 = []

        # Both kind of particles aren't jets
        if flavor1 == 0 and flavor2 == 0:
            
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0: # At least one of each
                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])
                    
                    if charge1[i][index1] + charge2[i][index2] == charge:
                        
                        pt_1.append(pt1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)
            
        # For collisions with decayment particles being jets
        elif flavor1 != 0 or flavor2 !=0:

            # For both particles being jets with different flavor:
            if flavor1 != 0 and flavor2 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)
                    suma_2 = np.sum(flavor[i] == flavor2)
                    positions1 = []
                    positions2 = []
                    trials_pt1 = []
                    trials_pt2 = []

                    if suma_1 > 0 and suma_2 > 0:

                        for k in range(len(flavor[i])):
                            if flavor[i][k] == flavor1:
                                positions1.append(k)
                                trials_pt1.append(pt1[i][k]) 
                                
                            elif flavor[i][k] == flavor2:
                                positions2.append(k)
                                trials_pt2.append(pt2[i][k])

                        pt_max1 = trials_pt1.index(max(trials_pt1))
                        pt_max2 = trials_pt2.index(max(trials_pt2))

                        index1 = positions1[pt_max1]
                        index2 = positions2[pt_max2]

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)

            # For just the first particle being a jet:
            elif flavor1 != 0 and flavor2 == 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)

            # For just the 2nd particle being a jet: 
            elif flavor1 == 0 and flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] + charge2[i][index2] == charge:
                            pt_1.append(pt1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)
             
        pt_1 = np.array(pt_1)
        phi_1 = np.array(phi_1)
            
        pt_2 = np.array(pt_2)
        phi_2 = np.array(phi_2)
             
        event_1 = np.array(event_1)
        
        data = {
                "pt_1": pt_1,
                "pt_2": pt_2,
                "phi_1": phi_1,
                "phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 
            
    # For different particles with one kind of them are photons
    elif p1 != p2 and (c1 == False or c2 == False):
        
        events = len(pt1)
        
        pt_1 = []
        phi_1 = []
        
        pt_2 = []
        phi_2 = []
        
        event_1 = []

        # When none of them are jets so one kind are photons and the others
        # are some other particle
        if flavor1 == 0 and flavor2 == 0:
        
            for i in range(events):
                if len(pt1[i]) > 0 and len(pt2[i]) > 0:

                    index1 = np.argmax(pt1[i])
                    index2 = np.argmax(pt2[i])

                    if c1 == True and charge1[i][index1] == charge:

                        pt_1.append(pt1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)

                    elif c2 == True and charge2[i][index2] == charge:

                        pt_1.append(pt1[i][index1])
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(pt2[i][index2])
                        phi_2.append(phi2[i][index2])
                            
                        event_1.append(i)

        # If one of the particles is a jet and the other a photon:
        elif flavor1 != 0 or flavor2 != 0:
            # If the first kind are jets and the second are photons
            if flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 0 and len(pt2[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge1[i][index1] == charge:

                            pt_1.append(pt1[i][index1])
                            phi_1.append(phi1[i,index1])
                                
                            pt_2.append(pt2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)
            # If the first ones are photons and the second are jets
            elif flavor2 != 0:
                for i in range(events):
                    suma_2 = np.sum(flavor[i] == flavor2)

                    if suma_2 > 0 and len(pt1[i]) > 0:
                        index1 = np.argmax(pt1[i])
                        index2 = np.argmax(pt2[i])

                        if charge2[i][index2] == charge:

                            pt_1.append(pt1[i][index1])
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(pt2[i][index2])
                            phi_2.append(phi2[i][index2])
                                
                            event_1.append(i)

        pt_1 = np.array(pt_1)
        phi_1 = np.array(phi_1)
        
        pt_2 = np.array(pt_2)
        phi_2 = np.array(phi_2)
         
        event_1 = np.array(event_1)

        
        data = {
                "pt_1": pt_1,
                "pt_2": pt_2,
                "phi_1": phi_1,
                "phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # In the case of detections of particles from the same type.
    # Events with no photons:
    elif p1 == p2 and (c1 == True):
        
        events = len(pt1)
        
        pt_1 = []
        phi_1 = []
        
        pt_2 = []
        phi_2 = []
        
        event_1 = []

        # Particles that aren't jets:
        if p1 != "jet":
            
            for i in range(events):
                if len(pt1[i]) > 1: # At least two particles
                    index1 = np.argmax(pt1[i])
                    trial_pt1 = pt1[i][index1]

                    pt1[i][index1] = 1e-1000

                    index2 = np.argmax(pt1[i])
                    trial_pt2 = pt1[i][index2]

                    if charge1[i][index1] + charge1[i][index2] == charge:
                        
                        pt_1.append(trial_pt1)
                        phi_1.append(phi1[i][index1])
                            
                        pt_2.append(trial_pt2)
                        phi_2.append(phi1[i][index2])
                            
                        event_1.append(i)

        # Events with jets:
        elif p1 == "jet":
            if flavor1 == flavor2 and flavor1 != 0:
                for i in range(events):
                    suma_1 = np.sum(flavor[i] == flavor1)

                    if suma_1 > 1: # At least two particles
                        index1 = np.argmax(pt1[i])
                        trial_pt1 = pt1[i][index1]

                        pt1[i][index1] = 1e-1000

                        index2 = np.argmax(pt1[i])
                        trial_pt2 = pt1[i][index2]

                        if charge1[i][index1] + charge1[i][index2] == charge:
                            
                            pt_1.append(trial_pt1)
                            phi_1.append(phi1[i][index1])
                                
                            pt_2.append(trial_pt2)
                            phi_2.append(phi1[i][index2])
                                
                            event_1.append(i)

        pt_1 = np.array(pt_1)
        phi_1 = np.array(phi_1)
            
        pt_2 = np.array(pt_2)
        phi_2 = np.array(phi_2)
             
        event_1 = np.array(event_1)
        
        data = {
                "pt_1": pt_1,
                "pt_2": pt_2,
                "phi_1": phi_1,
                "phi_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # For photons
    elif p1 == p2 and (c1 == False):
        events = len(pt1)
        
        pt_1 = []
        phi_1 = []
        
        pt_2 = []
        phi_2 = []
        
        event_1 = []
            
        for i in range(events):
            if len(pt1[i]) > 1: # At least two particles
                index1 = np.argmax(pt1[i])
                trial_pt1 = pt1[i][index1]

                pt1[i][index1] = 1e-1000

                index2 = np.argmax(pt1[i])
                trial_pt2 = pt1[i][index2]
                        
                pt_1.append(trial_pt1)
                phi_1.append(phi1[i][index1])
                            
                pt_2.append(trial_pt2)
                phi_2.append(phi1[i][index2])
                            
                event_1.append(i)

        pt_1 = np.array(pt_1)
        phi_1 = np.array(phi_1)
            
        pt_2 = np.array(pt_2)
        phi_2 = np.array(phi_2)
             
        event_1 = np.array(event_1)
        
        data = {
                "PRI_PT_1": pt_1,
                "PRI_PT_2": pt_2,
                "PRI_PHI_1": phi_1,
                "PRI_PHI_2": phi_2,
                "event_1": event_1,
                    }
                
        data = pd.DataFrame(data) 

    # Compute the momentum components
    data["p1_x"] = data["pt_1"]*np.cos(data["phi_1"])
    data["p1_y"] = data["pt_1"]*np.sin(data["phi_1"])

    data["p2_x"] = data["pt_2"]*np.cos(data["phi_2"])
    data["p2_y"] = data["pt_2"]*np.sin(data["phi_2"])

    # Then we can calculate the transverse momentum

    data["trans_momentum"] = np.sqrt(data["p1_x"]**2 + data["p1_y"]**2) 

    # Create the corrected dataframe
    data = data.drop(columns = ["pt_1", "pt_2", "phi_1", "phi_2"])


    # To save the dataframe as a .csv file:
    if save_df == True:
        data.to_csv("Transverse-Momentum.csv")

    return data

def leading_n_subleading_jets(file):
    # With this function we can get the pt, eta, phi of the leading
    # and subleading jets for all events
    file = uproot.open(file)
    tree = file["Delphes"]

    pt = tree["Jet/Jet.PT"].array(library = "np")
    eta = tree["Jet/Jet.Eta"].array(library = "np")
    phi = tree["Jet/Jet.Phi"].array(library = "np")

    events = len(pt)

    leading_pt = []
    subleading_pt = []
    leading_eta = []
    subleading_eta = []
    leading_phi = []
    subleading_phi = []
    event_1 = []

    for i in range(events):
        suma = len(pt[i])
        if suma >= 2: # More than a pair or a pair
            index1 = np.argmax(pt[i])
            trial_pt1 = pt[i][index1]

            pt[i][index1] = 1e-1000

            index2 = np.argmax(pt[i])
            trial_pt2 = pt[i][index2]
                            
            leading_pt.append(trial_pt1)
            leading_phi.append(phi[i][index1])
            leading_eta.append(eta[i][index1])
                                
            subleading_pt.append(trial_pt2)
            subleading_phi.append(phi[i][index2])
            subleading_eta.append(eta[i][index2])
                                
            event_1.append(i)
                    
        elif suma == 1: # Just one jet

            leading_pt.append(pt[i][0])
            leading_phi.append(phi[i][0])
            leading_eta.append(eta[i][0])
                                
            subleading_pt.append(0)
            subleading_phi.append(0)
            subleading_eta.append(0)

            event_1.append(i)
        
        elif suma == 0:                    
                            
            leading_pt.append(0)
            leading_phi.append(0)
            leading_eta.append(0)
                                
            subleading_pt.append(0)
            subleading_phi.append(0)
            subleading_eta.append(0)

            event_1.append(i)
        
    leading_pt = np.array(leading_pt)
    subleading_pt = np.array(subleading_pt)
    leading_eta = np.array(leading_eta)
    subleading_eta = np.array(subleading_eta)
    leading_phi = np.array(leading_phi)
    subleading_phi = np.array(subleading_phi)
    event_1 = np.array(event_1)

    data = {
        "PRI_jet_leading_pt": leading_pt,
        "PRI_jet_subleading_pt": subleading_pt,
        "PRI_jet_leading_eta": leading_eta,
        "PRI_jet_subleading_eta": subleading_eta,
        "PRI_jet_leading_phi": leading_phi,
        "PRI_jet_subleading_phi": subleading_phi,
        "event_1": event_1
    }
    
    data = pd.DataFrame(data)
    return data

def met(file):
    # Values from the MET:
    # This function gives the missing transverse energy and the met phi
    file = uproot.open(file)
    tree = file["Delphes"]

    met = tree["MissingET/MissingET.MET"].array(library = "np")
    met_phi = tree["MissingET/MissingET.Phi"].array(library = "np")

    events = len(met)

    MET = []
    MET_phi = []
    event = []

    for i in range(events):
        MET.append(met[i][0])
        MET_phi.append(met_phi[i][0])
        event.append(i)

    MET = np.array(MET)
    MET_phi = np.array(MET_phi)

    data = {
        "PRI_met": MET,
        "PRI_met_phi": MET_phi,
        "event_1": event
    }

    data = pd.DataFrame(data)

    return data

###############################################################################################################

def PRI_jet_num(file):
    file = uproot.open(file)
    tree = file["Delphes"]

    jets = tree["Jet/Jet.PT"].array(library = "np")

    event = []
    number_jets = []

    events = len(jets)

    for i in range(events):
        event.append(i)
        number_jets.append(len(jets[i]))
    
    event_1 = np.array(event)
    number_jets = np.array(number_jets)

    data = {
            "event_1": event_1,
            "PRI_jet_num" : number_jets
                    }
                
    data = pd.DataFrame(data) 

    return data

def PRI_jet_all_pt(file):
    file = uproot.open(file)
    tree = file["Delphes"]

    jets_pt = tree["Jet/Jet.PT"].array(library = "np")
                                       
    events = len(jets_pt)
    event_1 = []
    pt_sums = []

    for i in range(events):
        suma = np.sum(jets_pt[i])
        pt_sums.append(suma)
        event_1.append(i)
    
    event_1 = np.array(event_1)
    pt_sums = np.array(pt_sums)

    data = {
            "event_1": event_1,
            "PRI_jet_all_pt" :pt_sums
            }
                
    data = pd.DataFrame(data)

    return data

def DER_prodeta_jet_jet(file, flavor1, flavor2, charge = 0, save_df = False):
    # The product of the pseudorapidities of the two jets (undefined if
    # PRI jet num ≤ 1).
    file = uproot.open(file)
    tree = file["Delphes"]

    flavor = tree["Jet/Jet.Flavor"].array(library = "np")

    if flavor1 != flavor2:
        pt1 = tree["Jet/Jet.PT"].array(library = "np")
        eta1 = tree["Jet/Jet.Eta"].array(library = "np")
        charge1 = tree["Jet/Jet.Charge"].array(library = "np")
            
        pt2 = tree["Jet/Jet.PT"].array(library = "np")
        eta2 = tree["Jet/Jet.Eta"].array(library = "np")
        charge2 = tree["Jet/Jet.Charge"].array(library = "np")
    else:
        pt1 = tree["Jet/Jet.PT"].array(library = "np")
        eta1 = tree["Jet/Jet.Eta"].array(library = "np")
        charge1 = tree["Jet/Jet.Charge"].array(library = "np")
        
    # The jet flavor is computed using the the Simple Δ𝑅
    # highest-flavor match and it is asigned as follows:
    #   gluon --> 21
    #   d --> 1
    #   u --> 2
    #   s --> 3
    #   c --> 4
    #   b --> 5
    
    # For different particles:
    if flavor1 != flavor2:
        
        events = len(pt1)

        eta_1 = []
        eta_2 = []
        event_1 = []

        # For both particles being jets with different flavor:
        if flavor1 != 0 and flavor2 != 0:
            for i in range(events):
                suma_1 = np.sum(flavor[i] == flavor1)
                suma_2 = np.sum(flavor[i] == flavor2)
                positions1 = []
                positions2 = []
                trials_pt1 = []
                trials_pt2 = []

                if suma_1 > 0 and suma_2 > 0:

                    for k in range(len(flavor[i])):
                        if flavor[i][k] == flavor1:
                            positions1.append(k)
                            trials_pt1.append(pt1[i][k]) 
                                
                        elif flavor[i][k] == flavor2:
                            positions2.append(k)
                            trials_pt2.append(pt2[i][k])

                    pt_max1 = trials_pt1.index(max(trials_pt1))
                    pt_max2 = trials_pt2.index(max(trials_pt2))

                    index1 = positions1[pt_max1]
                    index2 = positions2[pt_max2]

                    if charge1[i][index1] + charge2[i][index2] == charge:
                        eta_1.append(eta1[i][index1])   
                        eta_2.append(eta2[i][index2])
                             
                        event_1.append(i)

    elif flavor1 == flavor2 and flavor1 != 0:
        for i in range(events):
            suma_1 = np.sum(flavor[i] == flavor1)

            if suma_1 > 1: # At least two particles
                index1 = np.argmax(pt1[i])
                pt1[i][index1] = 1e-1000

                index2 = np.argmax(pt1[i])

                if charge1[i][index1] + charge1[i][index2] == charge:
                            
                    eta_1.append(eta1[i][index1])
                    eta_2.append(eta1[i][index2])          
                    event_1.append(i)

        eta_1 = np.array(eta_1)
        eta_2 = np.array(eta_2)
             
        event_1 = np.array(event_1)
        
    data = {
            "eta_1": eta_1,
            "eta_2": eta_2,
            "event_1": event_1,
                }
                
    data = pd.DataFrame(data)

    # Compute the prodeta jet jet
    data["Der_prodeta_jet_jet"] = data["eta_1"]*data["eta_2"]

    # Then the dataframe is created
    data = data.drop(columns = ["eta_1", "eta_2"])

    # To save the dataframe as a .csv file:
    if save_df == True:
        data.to_csv("prodeta_jet_jet.csv")
    
    return data

def DER_deltaeta_jet_jet(file, j1, j2, flavor1, flavor2, charge = 0, save_df = False):
    # The absolute value of the pseudorapidity separation between
    # the two jets (undefined if PRI jet num ≤ 1)
    data = pseudorapidity_separation(file, j1 , j2, flavor1, flavor2, charge, save_df)
    data = data.rename(columns = {"DER_pseudorapidity_separation": "DER_deltaeta_jet_jet"})
    return data

