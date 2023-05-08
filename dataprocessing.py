import pandas as pd
df = pd.read_csv('F:\\PINN\\Physics-Informed-Neural-Networks-Multitask-Learning-master\\AE\\12.csv')
df_clear = df.drop(df[df['Heat']<0.2].index)
df_clear.to_csv('F:\\PINN\\Physics-Informed-Neural-Networks-Multitask-Learning-master\\AE\\121.csv')