import pandas as pd
path = 'countries.csv'
countries = pd.read_csv(path, sep=';')
numCountries = countries.country.max() + 1
countryIdMap = {i: countries.loc[countries['country']==i, 'filename'].iloc[0][:2] for i in range(numCountries)}

# make dataframe with same rows and columns
df = pd.DataFrame(index=countryIdMap.values(), columns=countryIdMap.values())

# elementwise check between two series




bordersDict = {'de': ['ch', 'cz', 'dk', 'fr', 'lu', 'nl', 'pl', 'at', 'se', 'be'],
                # switzerland, czech republic, denmark, france, luxembourg, netherlands, poland, austria, sweden, belgium
               'dk': ['de', 'se', 'no', 'gb'],
                # germany, sweden, norway, united kingdom
               'ee': ['fi', 'lv', 'ru', 'se'],
                # finland, latvia, russia, sweden
               'es': ['fr', 'ad', 'pt', 'ma', 'dz', 'it', 'gi'], # 'mt'
               # france, andorra, portugal, morocco, algeria, italy, gibraltar, #malta
               'fr': ['be', 'de', 'es', 'it', 'lu', 'mc', 'ch', 'ad', 'gi', 'gb'],
                # belgium, germany, spain, italy, luxembourg, monaco, switzerland, andorra, gibraltar, united kingdom
               'gb': ['ie', 'no', 'se', 'dk', 'de', 'fr', 'ch', 'at', 'it', 'es', 'pt', 'nl', 'be'],
                # ireland, norway, sweden, denmark, germany, france, switzerland, austria, italy, spain, portugal, netherlands, belgium
               'gr': ['al', 'bg', 'mk', 'tr', 'it'],  # 'ro'
                # albania, bulgaria, macedonia, italy, turkey # romania
               'it': ['fr', 'ch', 'at', 'si', 'sm', 'va', 'es', 'gr', 'mt', 'li', 'de', 'gb'],
                # france, switzerland, austria, slovenia, san marino, vatican city, spain, greece, malta, liechtenstein, germany, united kingdom
               'no': ['se', 'dk', 'gb', 'fi', 'ru'],
                # sweden, denmark, united kingdom, finland, russia
               'pl': ['cz', 'de', 'sk', 'ua', 'by', 'lt', 'lv', 'se', 'dk', 'no', 'ru'],
                # czech republic, germany, slovakia, ukraine, belarus, lithuania, latvia, sweden, denmark, norway, russia
               'ro': ['hu', 'md', 'ua', 'bg', 'gr', 'tr', 'rs'],
                # hungary, moldova, ukraine, bulgaria, greece, turkey, serbia
               'se': ['no', 'dk', 'de', 'fi', 'ee', 'ru', 'gb', 'pl', 'lt', 'lv'],
                # norway, denmark, germany, finland, estonia, russia, united kingdom, poland, lithuania, latvia
               'ua': ['pl', 'ro', 'md', 'hu', 'by', 'ru', 'lv', 'lt', 'sk', 'tr'],
                # poland, romania, moldova, hungary, belarus, russia, latvia, lithuania, slovakia, turkey
                # We need to add the rest as to get borde
                'at': ['de', 'ch', 'cz', 'hu', 'sk', 'si', 'it', 'gb', 'fr', 'es'],
                # germany, switzerland, czech republic, hungary, slovakia, slovenia, italy, united kingdom, france, spain
                'be': ['nl', 'fr', 'de', 'lu', 'gb'],
                # netherlands, france, germany, luxembourg, united kingdom
                'ch': ['de', 'fr', 'it', 'at', 'li', 'gb', 'es'],
                # germany, france, italy, austria, liechtenstein, united kingdom, spain
                'cz': ['de', 'pl', 'sk', 'at', 'hu', 'si', 'ua'],
                # germany, poland, slovakia, austria, hungary, slovenia, ukraine
                'rs': ['ro', 'bg', 'mk', 'hu', 'me', 'hr'],
                # romania, bulgaria, macedonia, hungary, montenegro, croatia
                'me': ['rs', 'al', 'ba', 'hr', 'mk', 'gr']
                # serbia, albania, bosnia, croatia, macedonia, greece
                

               }
            

def intersectBorder(a):
    # Borders that we have border for (can be extended)
    return list(set(a) & set(bordersDict.keys()))

def getBorderLoss(countryId, countryId2):
    # Not working
    if countryId2 in bordersDict[countryId]:
        return 1
    else:
        for border in intersectBorder(bordersDict[countryId]).remove(countryId):
            return 1 + getBorderLoss(border, countryId2)

    
        
# if border is in bordersDict, set to 1, else look at borders of borders
#for country in bordersDict.keys():
  #  for country2 in bordersDict.keys():
   #     df[country][country2] = getBorderLoss(country, country2)
        
# meget grimt, men nu skal det bare virke

for country in countryIdMap.values():
    for country2 in countryIdMap.values():
        if country2 in bordersDict[country]:
            df[country][country2] = 1

# runde 2
for country in countryIdMap.values():
    for country2 in countryIdMap.values():
        if df[country][country2] != 1:  # if not border already
            for border in intersectBorder(bordersDict[country]): # see if neighbor is border
                if border in countryIdMap.values() and df[border][country2] == 1:
                    df[country][country2] = 2
                    break

# runde 3
for country in countryIdMap.values():
    for country2 in countryIdMap.values():
        if df[country][country2] != 1 and df[country][country2] != 2:  # if not border already
            for border in intersectBorder(bordersDict[country]): # see if neighbor is border
                if border in countryIdMap.values() and df[border][country2] == 2:
                    df[country][country2] = 3
                    break
                
# runde 4
for country in countryIdMap.values():
    for country2 in countryIdMap.values():
        if df[country][country2] != 1 and df[country][country2] != 2 and df[country][country2] != 3:  # if not border already
            for border in intersectBorder(bordersDict[country]): # see if neighbor is border
                if border in countryIdMap.values() and df[border][country2] == 3:
                    df[country][country2] = 4
                    break

# runde 5
for country in countryIdMap.values():
    for country2 in countryIdMap.values():
        if df[country][country2] not in list(range(1,5)):  # if not border already
            for border in intersectBorder(bordersDict[country]): # see if neighbor is border
                if border in countryIdMap.values() and df[border][country2] == 4:
                    df[country][country2] = 5
                    break

df.to_csv("borderloss.csv", index=False, sep=";", header=False)

border_loss = lambda y_pred, y: df.iloc[y_pred, y]