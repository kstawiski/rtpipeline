%pip install numpy pandas openpyxl
import pandas as pd
import numpy as np
import os
import re

os.chdir("../Data_Labs/")

data_all = pd.DataFrame(columns = ("date", "pesel", "name", "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDWSD", "RDWCV", "PDW", "MPV", "PLCR", "PCT", "neu", "lymph", "mono", "eos", "baso", "NRBC", "ig", "n_neu", "n_lymph", "n_mono", "n_eos", "n_baso", "n_ig", "glucose", "sodium", "potassium", "urea", "creatinine", "egfr", "uric_acid", "bilirubine", "ALT", "AST", "phosphates", "calcium", "CRP", "CK-MB", "amylase", "lipase", "troponin_t", "NT-proBNP", "d-dimers", "PT", "PT_index", "INR", "APTT", "fibrinogen", "procalcytonine", "PSA", "testosteron"))

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

for filename in listdir_nohidden("."):
    if filename.endswith('.htm'):
        print(filename)
        file = open(filename, "r", encoding = "utf8").read()
        date = re.findall('[0-9]{3} / ([0-9]{2}-[0-9]{2}-[1-2][0-1][0-9][0-9])', file)

        #create list of date positions
        dateindex = []
        for match in re.finditer('[0-9]{3} / ([0-9]{2}-[0-9]{2}-[1-2][0-1][0-9][0-9])', file):
            dateindex.append(match.start())
        dateindex.append(-1)

        #create dataframe with dates, names and ids
        name = re.findall('<body>(.*),\s[0-9]{11}',file)*len(date)
        print("Name: " + str(name))
        pesel = re.findall('<body>.*\s([0-9]{11})',file)*len(date)
        print("PESEL: " + str(pesel))
        data = pd.DataFrame(list(zip(date, pesel, name)))
        data.columns = ("date", "pesel", "name")
        data = pd.concat([data, pd.DataFrame(columns = ("WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "RDWSD", "RDWCV", "PDW", "MPV", "PLCR", "PCT", "neu", "lymph", "mono", "eos", "baso", "NRBC", "ig", "n_neu", "n_lymph", "n_mono", "n_eos", "n_baso", "n_ig", "glucose", "sodium", "potassium", "urea", "creatinine", "egfr", "uric_acid", "bilirubine", "ALT", "AST", "phosphates", "calcium", "CRP", "CK-MB", "amylase", "lipase", "troponin_t", "NT-proBNP", "d-dimers", "PT", "PT_index", "INR", "APTT", "fibrinogen", "procalcytonine", "PSA", "testosteron", "CEA", "CA19-9"))])

        for i in range(len(date)):
            filefr = file[dateindex[i]:dateindex[i+1]]
            WBC = re.findall('Krwinki białe \(WBC\): .*?([0-9]+,[0-9]{2}) ×10.{1,3}/',filefr)
            data.at[i,'WBC'] = WBC[0] if len(WBC)>0 else None
            RBC = re.findall('Krwinki czerwone \(RBC\): .*?([0-9]+,[0-9]{2}) ×10.{1,3}/',filefr)
            data.at[i,'RBC'] = RBC[0] if len(RBC)>0 else None
            HGB = re.findall('Hemoglobina: .*?([0-9]+,[0-9]{1,3}) g/dl',filefr)
            data.at[i,'HGB'] = HGB[0] if len(HGB)>0 else None
            HCT = re.findall('Hematokryt: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'HCT'] = HCT[0] if len(HCT)>0 else None
            MCV = re.findall('MCV: .*?([0-9]+,[0-9]{1,3}) fl',filefr)
            data.at[i,'MCV'] = MCV[0] if len(MCV)>0 else None
            MCH = re.findall('MCH: .*?([0-9]+,[0-9]{1,3}) pg',filefr)
            data.at[i,'MCH'] = MCH[0] if len(MCH)>0 else None
            MCHC = re.findall('MCHC: .*?([0-9]+,[0-9]{1,3}) g/dl',filefr)
            data.at[i,'MCHC'] = MCHC[0] if len(MCHC)>0 else None
            PLT = re.findall('Płytki krwi: .*?([0-9]+) ×10.{1,3}/',filefr)
            data.at[i,'PLT'] = PLT[0] if len(PLT)>0 else None
            RDWSD = re.findall('RDW-SD: .*?([0-9]+,[0-9]{1,3}) fl',filefr)
            data.at[i,'RDWSD'] = RDWSD[0] if len(RDWSD)>0 else None
            RDWCV = re.findall('RDW-CV: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'RDWCV'] = RDWCV[0] if len(RDWCV)>0 else None
            PDW = re.findall('PDW: .*?([0-9]+,[0-9]{1,3}) fl',filefr)
            data.at[i,'PDW'] = PDW[0] if len(PDW)>0 else None
            mpv = re.findall('MPV: .*?([0-9]+,[0-9]{1,3}) fl',filefr)
            data.at[i,'MPV'] = mpv[0] if len(mpv)>0 else None
            plcr = re.findall('P-LCR: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'PLCR'] = plcr[0] if len(plcr)>0 else None
            pct = re.findall('PCT: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'PCT'] = pct[0] if len(pct)>0 else None
            neu = re.findall('% neutrocytów: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'neu'] = neu[0] if len(neu)>0 else None
            lymph = re.findall('% limfocytów: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'lymph'] = lymph[0] if len(lymph)>0 else None
            mono = re.findall('% monocytów: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'mono'] = mono[0] if len(mono)>0 else None
            eos = re.findall('% eozynocytów: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'eos'] = eos[0] if len(eos)>0 else None
            baso = re.findall('% bazocytów: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'baso'] = baso[0] if len(baso)>0 else None
            nRBC = re.findall('% NRBC: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'NRBC'] = nRBC[0] if len(nRBC)>0 else None
            ig = re.findall('% niedojrzałych granulocytów .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'ig'] = ig[0] if len(ig)>0 else None
            n_neu = re.findall('Liczba neutrocytów: .*?([0-9]+,[0-9]{2}) ×10.{1,3}/',filefr)
            data.at[i,'n_neu'] = n_neu[0] if len(n_neu)>0 else None
            n_lymph = re.findall('Liczba limfocytów: .*?([0-9]+,[0-9]{2}) ×10.{1,3}/',filefr)
            data.at[i,'n_lymph'] = n_lymph[0] if len(n_lymph)>0 else None
            n_mono = re.findall('Liczba monocytów: .*?([0-9]+,[0-9]{2}) ×10.{1,3}/',filefr)
            data.at[i,'n_mono'] = n_mono[0] if len(n_mono)>0 else None
            n_eos = re.findall('Liczba eozynocytów: .*?([0-9]+,[0-9]{2}) ×10.{1,3}/',filefr)
            data.at[i,'n_eos'] = n_eos[0] if len(n_eos)>0 else None
            n_baso = re.findall('Liczba bazocytów: .*?([0-9]+,[0-9]{2}) ×10.{1,3}/',filefr)
            data.at[i,'n_baso'] = n_baso[0] if len(n_baso)>0 else None
            n_ig = re.findall('Liczba niedojrzałych granulocytów \(IG\): .*?([0-9]+,[0-9]{2}) x10.{1,3}/',filefr)
            data.at[i,'n_ig'] = n_ig[0] if len(n_ig)>0 else None
            glu = re.findall('Glukoza: .*?([0-9]+) mg/dl',filefr)
            data.at[i,'glucose'] = glu[0] if len(glu)>0 else None
            sod = re.findall('Sód: .*?([0-9]+) mmol/l',filefr)
            data.at[i,'sodium'] = sod[0] if len(sod)>0 else None
            pot =  re.findall('Potas: .*?([0-9]+,[0-9]{1,3}) mmol/l',filefr)
            data.at[i,'potassium'] = pot[0] if len(pot)>0 else None
            urea = re.findall('Mocznik: .*?([0-9]+,[0-9]{1,3}) mg/dl',filefr)
            data.at[i,'urea'] = urea[0] if len(urea)>0 else None
            creat = re.findall('Kreatynina: .*?([0-9]+,[0-9]{1,3}) mg/dl',filefr)
            data.at[i,'creatinine'] = creat[0] if len(creat)>0 else None
            gfr = re.findall('GFR wg MDRD: .*?([0-9]+) ml/min/1,73',filefr)
            data.at[i,'egfr'] = gfr[0] if len(gfr)>0 else None
            uric = re.findall('Kwas moczowy: .*?([0-9]+,[0-9]{1,3}) mg/dl',filefr)
            data.at[i,'uric_acid'] = uric[0] if len(uric)>0 else None
            bili = re.findall('Bilirubina całkowita: .*?([0-9]+,[0-9]{1,3}) mg/dl',filefr)
            data.at[i,'bilirubine'] = bili[0] if len(bili)>0 else None
            alt = re.findall('ALT: .*?([0-9]+) U/l',filefr)
            data.at[i,'ALT'] = alt[0] if len(alt)>0 else None
            ast = re.findall('AST: .*?([0-9]+) U/l',filefr)
            data.at[i,'AST'] = ast[0] if len(ast)>0 else None
            phos = re.findall('Fosforany: .*?([0-9]+,[0-9]{1,3}) mmol/l',filefr)
            data.at[i,'phosphates'] = phos[0] if len(phos)>0 else None
            calc = re.findall('Wapń: .*?([0-9]+,[0-9]{1,3}) mmol/l',filefr)
            data.at[i,'calcium'] = calc[0] if len(calc)>0 else None
            crp = re.findall('Białko ostrej fazy - CRP: .*?([0-9]+,[0-9]{1,3}) mg/l',filefr)
            data.at[i,'CRP'] = crp[0] if len(crp)>0 else None
            ckmb = re.findall('CK-MB aktywność: .*?([0-9]+,[0-9]{1,3}) U/l',filefr)
            data.at[i,'CK-MB'] = ckmb[0] if len(ckmb)>0 else None
            amy = re.findall('Amylaza: .*?([0-9]+) U/l',filefr)
            data.at[i,'amylase'] = amy[0] if len(amy)>0 else None
            ly = re.findall('Lipaza: .*?([0-9]+) U/l',filefr)
            data.at[i,'lipase'] = ly[0] if len(ly)>0 else None
            tro = re.findall('Troponina.*?([0-9]+) ng/l',filefr)
            data.at[i,'troponin_t'] = tro[0] if len(tro)>0 else None
            ntpro = re.findall('NT pro - BNP: .*?([0-9]+,[0-9]{1,3}) pg/ml',filefr)
            data.at[i,'NT-proBNP'] = ntpro[0] if len(ntpro)>0 else None
            ddim = re.findall('D-dimer: .*?([0-9]+,[0-9]{1,3})',filefr)
            data.at[i,'d-dimers'] = ddim[0] if len(ddim)>0 else None
            pt = re.findall('PT .*?([0-9]+,[0-9]{1,3})',filefr)
            data.at[i,'PT'] = pt[0] if len(pt)>0 else None
            wskpt = re.findall('Wskaźnik PT: .*?([0-9]+,[0-9]{1,3}) %',filefr)
            data.at[i,'PT_index'] = wskpt[0] if len(wskpt)>0 else None
            inr = re.findall('INR: .*?([0-9]+,[0-9]{1,3})',filefr)
            data.at[i,'INR'] = inr[0] if len(inr)>0 else None
            aptt = re.findall('APTT .*?([0-9]+,[0-9]{1,3})',filefr)
            data.at[i,'APTT'] = aptt[0] if len(aptt)>0 else None
            fibr = re.findall('Fibrynogen: .*?([0-9]+) mg/dl',filefr)
            data.at[i,'fibrinogen'] = fibr[0] if len(fibr)>0 else None
            proc = re.findall('Prokalcytonina: .*?([0-9]+,[0-9]{1,3})',filefr)
            data.at[i,'procalcytonine'] = proc[0] if len(proc)>0 else None
            proc = re.findall('PSA całkowity: .*?([0-9]+,[0-9]{1,3}) ng/ml',filefr)
            data.at[i,'PSA'] = proc[0] if len(proc)>0 else None
            proc = re.findall('Testosteron: .*?([0-9]+,[0-9]{1,3}) ng/ml',filefr)
            data.at[i,'testosteron'] = proc[0] if len(proc)>0 else None
            proc = re.findall('CEA: .*?([0-9]+,[0-9]{1,3}) ng/ml',filefr)
            data.at[i,'CEA'] = proc[0] if len(proc)>0 else None
            proc = re.findall('CA 19-9: .*?([0-9]+,[0-9]{1,3}) U/ml',filefr)
            data.at[i,'CA19-9'] = proc[0] if len(proc)>0 else None
        data_all = pd.concat([data_all, data], axis = 0)

for i in range(len(data.columns)):
    data.iloc[:,i] = data.iloc[:,i].str.replace(',', ".")

data_all

data_all.to_excel("../Data/Labs.xlsx")

