import pandas as pd
import plotly.graph_objects as go


def PlotPieByFlag(df: pd.DataFrame):
    total = len(df)
    flag_0_count = list(df.loc[:, "flag"].value_counts().values)[0]
    flag_1_count = total - flag_0_count

    # No need to provide 3d bar plot, for the extra axis won't give additional information
    labels = ["Not Zombie Enterprise", "Zombie Enterprise"]
    values = [flag_0_count, flag_1_count]
    colors = ["MediumAquaMarine", "DarkOrchid"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.2, 0])])
    fig.update_traces(textinfo="label+percent", textfont_size=10,
                      marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.show()


if __name__ == "__main__":
    target_data_set = pd.read_csv("../TraditionML/data/train_set.csv", encoding="utf-8")
    PlotPieByFlag(target_data_set)

