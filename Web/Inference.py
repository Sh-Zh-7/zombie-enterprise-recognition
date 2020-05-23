import pandas as pd


def Inference(data):
    result = pd.DataFrame()
    result["ID"] = data["ID"]
    zombie_id = data.loc[((data["2015_纳税总额"] == 0) & (data["2016_纳税总额"] == 0) &
                          (data["2017_纳税总额"] == 0)), ["ID"]]
    result["flag"] = result.ID.apply(lambda x: 1
                                     if x in list(zombie_id.ID) else 0)
    return result
