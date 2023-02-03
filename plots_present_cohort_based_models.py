import plotly.graph_objects as go
import plotly.offline as py
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gaussian

from _temp.demo_cohort_based_models.utils import get_b_splines
from lib_model_adapters.lightgbm import LightGBMAdapter

dir = '/_tmp/demo_cohort_based_models'
filename = f'{dir}/df2.csv'
df_r = pd.read_csv(filename)
df_r['reg_date'] = pd.to_datetime(df_r['reg_date'])
df_r['upgrade_date'] = pd.to_datetime(df_r['upgrade_date'])

#filename = '/data_cache/_tmp/x_train_old_td20200331.csv'
# df_o = pd.read_csv(filename)
df_o = df_r.loc[df_r['age'] < 365].groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
df_o.columns = ['upgrade_date', 'premiums']
df_o['upgrade_date'] = pd.to_datetime(df_o['upgrade_date'])


def plot_cohorts(regdates, xaxis_range, yaxis_range, upgrade_rng=None,
                 by_age=False, log_transform=False, color='black',
                 opacity=0.5, width=2, xaxis_title='(Upgrade) Date',
                 yaxis_title='Premiums', mode='lines'):

    if upgrade_rng is not None:
        idx = (upgrade_rng[0] <= df_r['upgrade_date']) & (df_r['upgrade_date'] <= upgrade_rng[1])
        df_filt = df_r.loc[idx]
    else:
        df_filt = df_r.copy()

    fig = go.Figure()
    for regdate in regdates:
        df_tmp = df_filt.loc[df_r['reg_date'] == str(regdate)]
        x = df_tmp['upgrade_date']
        if by_age:
            x = df_tmp['age']

        y = df_tmp['premiums']

        if log_transform:
            y = np.log1p(df_tmp['premiums'])
            x = np.log1p(x)

        fig.add_trace(go.Scatter(x=x, y=y, name=str(regdate),
                                 mode=mode, opacity=opacity, line=dict(color=color, width=width)))

    fig.update_layout(title='', xaxis=dict(title=xaxis_title), yaxis=dict(title=yaxis_title),
                      xaxis_range=xaxis_range, yaxis_range=yaxis_range,
                      font=dict(size=20), showlegend=False)

    return fig


# one cohort
fig = plot_cohorts(
    regdates=['2019-01-01'],
    xaxis_range=['2019-01-01', '2019-01-30'],
    yaxis_range=[0, 500], width=3, xaxis_title='Upgrade Date'
)
py.plot(fig)


# several cohorts
# ******************************************************************************
regdates = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-05',
            '2019-01-06', '2019-01-07', '2019-01-08', '2019-01-09', '2019-01-10']

# by upgrade, 30 days
fig = plot_cohorts(
    regdates=regdates,
    xaxis_range=['2019-01-01', '2019-01-30'], yaxis_range=[0, 500],
    width=3, xaxis_title='Upgrade Date'
)
py.plot(fig)

# by age, 30 days
fig = plot_cohorts(
    regdates=regdates,
    xaxis_range=[0, 30], yaxis_range=[0, 500],
    by_age=True, width=3, xaxis_title='Age'
)
py.plot(fig)

# by age, 365 days
fig = plot_cohorts(
    regdates=regdates,
    xaxis_range=['2019-01-01', '2019-12-31'], upgrade_rng=['2019-01-01', '2019-12-31'],
    yaxis_range=[0, 500], by_age=True, width=3, xaxis_title='Age'
)
py.plot(fig)

# by age, 365 days - log vs log
fig = plot_cohorts(
    regdates=regdates,
    xaxis_range=['2019-01-01', '2019-12-31'], upgrade_rng=['2019-01-01', '2019-12-31'],
    log_transform=True, opacity=0.35,
    yaxis_range=[0, 10], by_age=True, width=3, xaxis_title='log(Age)', yaxis_title='log(Premiums)',
)
py.plot(fig)

# by age, 2 years
regdates = pd.date_range(start='2018-01-01', end='2019-12-31', freq='1 D')
regdates = np.random.choice(regdates, 150)
fig = plot_cohorts(
    regdates=regdates,
    xaxis_range=[0, 3*365], # upgrade_rng=['2019-01-01', '2019-12-31'],
    yaxis_range=[0, 50], opacity=0.025, by_age=True, width=1, xaxis_title='Age'
)
py.plot(fig)



# big plot
fig = plot_cohorts(
    regdates=pd.date_range(start='2019-01-01', end='2019-12-31', freq='5 D'),
    xaxis_range=['2019-01-01', '2020-03-31'], upgrade_rng=['2019-01-01', '2019-12-31'],
    yaxis_range=[0, 500], width=3, xaxis_title='Upgrade Date'
)
py.plot(fig)

# existing cohorts: 1 year + 1 quarter
fig = plot_cohorts(
    regdates=pd.date_range(start='2019-01-01', end='2019-12-31', freq='5 D'),
    xaxis_range=['2019-01-01', '2020-03-31'], upgrade_rng=['2019-01-01', '2020-03-31'],
    yaxis_range=[0, 500], width=3, xaxis_title='Upgrade Date'
)
py.plot(fig)

# add future cohorts
regdates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='5 D')
for regdate in regdates:
    df_tmp = df_r.loc[df_r['reg_date'] == str(regdate)]
    fig.add_trace(go.Scatter(x=df_tmp['upgrade_date'], y=df_tmp['premiums'], name=str(regdate),
                             mode='lines', opacity=0.5, line=dict(color='blue', width=3)))
py.plot(fig)

# add old cohorts
regdates = pd.date_range(start='2018-03-01', end='2018-12-15', freq='5 D')
for regdate in regdates:
    df_tmp = df_r.loc[df_r['reg_date'] == str(regdate)]
    fig.add_trace(go.Scatter(x=df_tmp['upgrade_date'], y=df_tmp['premiums'], name=str(regdate),
                             mode='lines', opacity=0.5, line=dict(color='orange', width=3)))
py.plot(fig)

fig = plot_cohorts(
    regdates=['2019-12-15'],
    xaxis_range=['2019-12-14', '2020-03-31'], upgrade_rng=['2019-01-01', '2020-03-31'],
    yaxis_range=[0, 200], width=3, xaxis_title='Upgrade Date'
)
py.plot(fig)

# ******************************************************************************
# recent cohorts
regdates = pd.date_range(start='2019-01-01', end='2019-12-31', freq='1 D')
idx = ('2019-01-01' <= df_r['upgrade_date']) & (df_r['upgrade_date'] <= '2020-03-31')
df_filt = df_r.loc[idx]
fig = go.Figure()
for regdate in regdates:
    df_tmp = df_filt.loc[df_r['reg_date'] == str(regdate)]
    fig.add_trace(go.Scatter(x=df_tmp['upgrade_date'], y=df_tmp['premiums'], name=str(regdate),
                             mode='lines', opacity=0.6, line=dict(color='black', width=2)))
fig.update_layout(title='', xaxis=dict(title='Upgrade Date'), yaxis=dict(title='Premiums'),
                  xaxis_range=['2019-01-01', '2020-03-31'], yaxis_range=[0, 6000],
                  font=dict(size=20), showlegend=False)

# add aggregated future as time series
idx = (df_r['reg_date'] <= '2020-01-01')
df_filt = df_r.loc[idx]
df_agg = df_r.loc[idx].groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
df_agg.columns = ['upgrade_date', 'premiums']
fig.add_trace(go.Scatter(x=df_agg['upgrade_date'], y=df_agg['premiums'], name='recent',
                         mode='lines', opacity=0.60, line=dict(color='black', width=3)))
py.plot(fig)


# ******************************************************************************
# future cohorts
regdates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='1 D')
idx = ('2019-01-01' <= df_r['upgrade_date']) & (df_r['upgrade_date'] <= '2020-03-31')
df_filt = df_r.loc[idx]
fig = go.Figure()
for regdate in regdates:
    df_tmp = df_filt.loc[df_r['reg_date'] == str(regdate)]
    fig.add_trace(go.Scatter(x=df_tmp['upgrade_date'], y=df_tmp['premiums'], name=str(regdate),
                             mode='lines', opacity=0.6, line=dict(color='blue', width=2)))
fig.update_layout(title='', xaxis=dict(title='Upgrade Date'), yaxis=dict(title='Premiums'),
                  xaxis_range=['2019-01-01', '2020-03-31'], yaxis_range=[0, 6000],
                  font=dict(size=20), showlegend=False)

# add aggregated future as time series
idx = (df_r['reg_date'] > '2020-01-01')
df_filt = df_r.loc[idx]
df_agg = df_r.loc[idx].groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
df_agg.columns = ['upgrade_date', 'premiums']
fig.add_trace(go.Scatter(x=df_agg['upgrade_date'], y=df_agg['premiums'], name='recent',
                         mode='lines', opacity=0.60, line=dict(color='blue', width=3)))
py.plot(fig)


# ******************************************************************************
# old cohorts
fig = go.Figure()
regdates = pd.date_range(start='2018-03-01', end='2018-12-25', freq='5 D')
for regdate in regdates:
    df_tmp = df_r.loc[df_r['reg_date'] == str(regdate)]
    fig.add_trace(go.Scatter(x=df_tmp['upgrade_date'], y=df_tmp['premiums'], name=str(regdate),
                             mode='lines', opacity=0.5, line=dict(color='orange', width=3)))
fig.update_layout(title='', xaxis=dict(title='Upgrade Date'), yaxis=dict(title='Premiums'),
                  xaxis_range=['2019-01-01', '2020-03-31'], yaxis_range=[0, 6000],
                  font=dict(size=20), showlegend=False)

# add aggregated as time series
fig.add_trace(go.Scatter(x=df_o['upgrade_date'], y=df_o['premiums'], name='recent',
                         mode='lines', opacity=0.80, line=dict(color='orange', width=3)))
py.plot(fig)


# ******************************************************************************
fig = go.Figure()
fig.update_layout(
    title='', xaxis=dict(title='Upgrade Date'), yaxis=dict(title='Premiums'),
    xaxis_range=['2019-01-01', '2020-03-31'], yaxis_range=[0, 6000],
    font=dict(size=20), showlegend=False)

# add aggregated as time series
df_o_agg = df_o[['upgrade_date', 'premiums']].rename(columns={'premiums': 'old'})
fig.add_trace(go.Scatter(x=df_o_agg['upgrade_date'], y=df_o_agg['old'], name='old',
                         mode='lines', opacity=0.80, line=dict(color='orange', width=3)))

idx = (df_r['reg_date'] <= '2020-01-01')
df_filt = df_r.loc[idx]
df_r_agg = df_r.loc[idx].groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
df_r_agg.columns = ['upgrade_date', 'recent']
fig.add_trace(go.Scatter(x=df_r_agg['upgrade_date'], y=df_r_agg['recent'], name='recent',
                         mode='lines', opacity=0.60, line=dict(color='black', width=3)))


idx = (df_r['reg_date'] > '2020-01-01')
df_filt = df_r.loc[idx]
df_f_agg = df_r.loc[idx].groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
df_f_agg.columns = ['upgrade_date', 'future']
fig.add_trace(go.Scatter(x=df_f_agg['upgrade_date'], y=df_f_agg['future'], name='future',
                         mode='lines', opacity=0.60, line=dict(color='blue', width=3)))

py.plot(fig)


# *********
fig = go.Figure()
fig.update_layout(
    title='', xaxis=dict(title='Upgrade Date'), yaxis=dict(title='Premiums'),
    xaxis_range=['2019-01-01', '2020-03-31'], yaxis_range=[0, 6000],
    font=dict(size=20), showlegend=False)

# add aggregated as time series
df_o_agg = df_o[['upgrade_date', 'premiums']].rename(columns={'premiums': 'old'})
fig.add_trace(go.Scatter(x=df_o_agg['upgrade_date'], y=df_o_agg['old'], name='old',
                         mode='lines', opacity=0.30, line=dict(color='orange', width=3)))

idx = (df_r['reg_date'] <= '2020-01-01')
df_filt = df_r.loc[idx]
df_r_agg = df_r.loc[idx].groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
df_r_agg.columns = ['upgrade_date', 'recent']
fig.add_trace(go.Scatter(x=df_r_agg['upgrade_date'], y=df_r_agg['recent'], name='recent',
                         mode='lines', opacity=0.25, line=dict(color='black', width=3)))


idx = (df_r['reg_date'] > '2020-01-01')
df_filt = df_r.loc[idx]
df_f_agg = df_r.loc[idx].groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
df_f_agg.columns = ['upgrade_date', 'future']
fig.add_trace(go.Scatter(x=df_f_agg['upgrade_date'], y=df_f_agg['future'], name='future',
                         mode='lines', opacity=0.25, line=dict(color='blue', width=3)))

py.plot(fig)

df_all_agg = pd.merge(df_r_agg, df_f_agg, how='outer', on=['upgrade_date'])
df_all_agg = pd.merge(df_all_agg, df_o_agg, how='outer', on=['upgrade_date'])
df_all_agg = df_all_agg.fillna(0)
df_all_agg['total'] = df_all_agg['old'] + df_all_agg['recent'] + df_all_agg['future']
df_all_agg = df_all_agg.sort_values(by=['upgrade_date'])

fig.add_trace(go.Scatter(x=df_all_agg['upgrade_date'], y=df_all_agg['total'], name='total',
                         mode='lines', opacity=0.99, line=dict(color='black', width=3)))
fig.update_layout(yaxis_range=[0, 10000])
py.plot(fig)


# only total different scale range
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_all_agg['upgrade_date'], y=df_all_agg['total'], name='total',
                         mode='lines', opacity=0.99, line=dict(color='black', width=3)))
fig.update_layout(
    title='', xaxis=dict(title='Upgrade Date'), yaxis=dict(title='Premiums'),
    xaxis_range=['2019-01-01', '2020-03-31'], yaxis_range=[5000, 10000],
    font=dict(size=20), showlegend=False)
py.plot(fig)


# only total for 5 years
# adjust december
df_all_agg['m'] = 1
idx = (df_all_agg['upgrade_date'].dt.month == 1) \
      & (df_all_agg['upgrade_date'].dt.day < 15) \
      & (df_all_agg['upgrade_date'].dt.year == 2015)
df_all_agg.loc[idx, 'total'] = 1500 + 0.45 * df_all_agg.loc[idx, 'total']
idx = (df_all_agg['upgrade_date'].dt.month == 12) & (df_all_agg['upgrade_date'].dt.day > 10)
df_all_agg.loc[idx, 'm'] = 1.05
idx = (df_all_agg['upgrade_date'].dt.month == 12) & (df_all_agg['upgrade_date'].dt.day > 25)
df_all_agg.loc[idx, 'm'] = 1.2
idx = (df_all_agg['upgrade_date'].dt.month == 12) & (df_all_agg['upgrade_date'].dt.day == 31)
df_all_agg.loc[idx, 'm'] = 1.3
df_all_agg['total_adj'] = df_all_agg['m'] * df_all_agg['total']

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_all_agg['upgrade_date'], y=df_all_agg['total_adj'], name='total',
                         mode='lines', opacity=0.60, line=dict(color='black', width=2)))
fig.update_layout(
    title='', xaxis=dict(title='Upgrade Date'), yaxis=dict(title='Premiums'),
    font=dict(size=20), showlegend=False)
py.plot(fig)


# histogram
fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Histogram(x=df_r['premiums'], histnorm='probability', marker_color='grey'), row=1, col=1)
fig.add_trace(go.Histogram(x=np.log1p(df_r['premiums']), histnorm='probability', marker_color='black'), row=1, col=2)
fig.update_xaxes(title_text='Premiums (0 to 50)', range=[0, 50], row=1, col=1)
fig.update_xaxes(title_text='log(1+Premiums)', row=1, col=2)
fig.update_yaxes(title_text='Frequency', row=1, col=1)
fig.update_layout(title='', font=dict(size=20), showlegend=False, bargap=0)
fig.update_traces(opacity=0.75)
py.plot(fig)


# run regression with different distributions
idx = np.random.choice(df_r.shape[0] - 1, 100000)
df_fcst = df_r.loc[idx].reset_index(drop=True)

df_fcst['log1p_premiums'] = np.log1p(df_fcst['premiums'])
df_fcst['log1p_age'] = np.log1p(df_fcst['age'])
tmp = get_b_splines(df_fcst['age'], 'age', [10])
df_fcst = pd.concat([df_fcst, tmp], axis=1)

df_fcst['h_age'] = np.maximum(0, 4-df_fcst['age'])
df_fcst['saleXh_age'] = df_fcst['sale_effect']*df_fcst['h_age']
df_fcst['weeklyXh_age'] = df_fcst['term_weekly_seas']*df_fcst['h_age']

print(df_fcst.columns)

target = 'premiums'
feats = ['age', 'sale_effect', 'term_yearly_seas', 'term_weekly_seas', 'term_holiday',
         'log1p_age', 'age_s0', 'age_s1', 'age_s2', 'age_s3',
         'age_s4', 'age_s5', 'age_s6', 'age_s7', 'age_s8',
         'saleXh_age', 'weeklyXh_age',
         ]

x_part = ' + '.join(feats)
model = GLM.from_formula(formula=f'premiums ~ {x_part}', data=df_fcst, family=Gaussian())
model = model.fit()
df_fcst['pred_regression'] = model.predict(df_fcst)

model = GLM.from_formula(formula=f'log1p_premiums ~ {x_part}', data=df_fcst, family=Gaussian())
model = model.fit()
df_fcst['pred_logtrans'] = np.expm1(model.predict(df_fcst))

model = GLM.from_formula(formula=f'premiums ~ {x_part}', data=df_fcst, family=Poisson())
model = model.fit()
df_fcst['pred_poisson'] = model.predict(df_fcst)


model = LightGBMAdapter(target, feats, **{'objective': 'regression'})
model.fit(df_fcst[feats], np.minimum(df_fcst[target], 180) + 0.33*np.maximum(df_fcst[target]-180, 0))
df_fcst['pred_regression'] = model.predict(df_fcst)

model = LightGBMAdapter('log1p_premiums', feats, **{'objective': 'regression'})
model.fit(df_fcst[feats], df_fcst['log1p_premiums'])
df_fcst['pred_logtrans'] = np.expm1(model.predict(df_fcst))

model = LightGBMAdapter(target, feats, **{'objective': 'poisson'})
model.fit(df_fcst[feats], df_fcst[target])
df_fcst['pred_poisson'] = model.predict(df_fcst)

# model = LightGBMAdapter(target, feats, **{'objective': 'tweedie'})
# model.fit(df_r[feats], df_r[target])
# df_fcst['pred_tweedie'] = model.predict(df_fcst)


idx = np.random.choice(df_fcst.shape[0] - 1, 10000)
tmp = df_fcst.loc[idx].reset_index(drop=True)
xmax = 450
xlin = np.arange(start=0, stop=xmax, step=1)
fig = make_subplots(rows=1, cols=3)
fig.add_trace(go.Scatter(x=tmp['premiums'], y=tmp['pred_regression'], name="regression",
                         mode='markers', opacity=0.2), row=1, col=1)
fig.add_trace(go.Scatter(x=xlin, y=xlin, mode='lines', opacity=0.5, line=dict(color='grey')), row=1, col=1)

fig.add_trace(go.Scatter(x=tmp['premiums'], y=tmp['pred_logtrans'], name="log transformed",
                         mode='markers', opacity=0.2), row=1, col=2)
fig.add_trace(go.Scatter(x=xlin, y=xlin, mode='lines', opacity=0.5, line=dict(color='grey')), row=1, col=2)

fig.add_trace(go.Scatter(x=tmp['premiums'], y=tmp['pred_poisson'], name="poisson",
                         mode='markers', opacity=0.2), row=1, col=3)
fig.add_trace(go.Scatter(x=xlin, y=xlin, mode='lines', opacity=0.5, line=dict(color='grey')), row=1, col=3)

fig.update_xaxes(title_text='Actual', range=[0, xmax], row=1, col=1)
fig.update_xaxes(title_text='Actual', range=[0, xmax], row=1, col=2)
fig.update_xaxes(title_text='Actual', range=[0, xmax], row=1, col=3)
fig.update_yaxes(title_text='Predicted (Normal)', range=[0, xmax], row=1, col=1)
fig.update_yaxes(title_text='Predicted (Log transform)', range=[0, xmax], row=1, col=2)
fig.update_yaxes(title_text='Predicted (Poisson)', range=[0, xmax], row=1, col=3)
fig.update_layout(title='', font=dict(size=20), showlegend=False, bargap=0)
fig.update_traces(opacity=0.50)
py.plot(fig)



from sklearn.metrics import r2_score
print(r2_score(df_fcst['premiums'], df_fcst['pred_poisson']))
print(r2_score(df_fcst['premiums'], df_fcst['pred_regression']))
print(r2_score(df_fcst['premiums'], df_fcst['pred_logtrans']))





# import pandas as pd
# self.x_train.to_csv(f'./data_cache/_tmp/x_train_{self.__class__.__name__}.csv', index=False, float_format='%.4f')
#
# idx = ('2019-12-01' <= df_r['upgrade_date']) & (df_r['upgrade_date'] <= '2019-12-31')
# tmp = df_r.loc[idx][['reg_date', 'upgrade_date', 'premiums']]
# tmp.to_csv(f'{dir_tmp}/tmp.csv', index=False, float_format='%.1f')

