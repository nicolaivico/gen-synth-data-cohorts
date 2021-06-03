# GENERATE DATA FOR COHORTS OF USERS AND PREMIUMS
# ******************************************************************************

# Definition of the "true" process
# ******************************************************************************
# - multiplicative model for users:
# users(r) = trend(r) * yearly(r) * weekly(r) * holiday(r)
# where: r - registration date, t - upgrade date

# - model for the exponential decay of premiums by age:
# premium_rate(age) = f(age)
# where: f(age) is a non-linear function

# - Poisson model for premiums:
# lambda(r,t) = offers(t) * trend(r) * yearly(r) * weekly(r) * premium_rate(age=t-r) * frees(r)
# ln(lambda(r,t)) = offers(t) + trend(r) + yearly(r) + weekly(r) + premium_rate(r,t) + frees(r)
# premiums(r,t) ~ Poisson(lambda(r,t))


# ******************************************************************************
import numpy as np
import pandas as pd
import itertools
import os
from statsmodels.gam.api import BSplines
from pandas.tseries.holiday import *
import plotly.graph_objects as go
import plotly.offline as py


# config
# ******************************************************************************
START_DATE = '2015-01-01'
END_DATE = '2021-12-31'

COEFS_BSPLINES_TREND = [0, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.40]
COEFS_BSPLINES_YEARLY_SEAS = [0.15, 0.10, 0.05, 0.05, 0, -0.05, -0.05, -0.02, 0.05, 0.0, 0.0, -0.10, -0.05]
COEFS_BSPLINES_WEEKLY_SEAS = [0.05, 0.10, 0.10, 0, -0.40, -0.30]
COEF_HOLIDAY = -0.25

SCALE_USERS = 50000
USERS_POW_TREND = 1
USERS_POW_YEARLY_SEAS = 1
USERS_POW_WEEKLY_SEAS = 1
USERS_POW_HOLIDAY = 1
ERROR_USERS_STD_DEV = 0.025

PREMIUM_RATE_TOTAL = 0.10
COEFS_BSPLINES_AGE_0_90 = [1, 0.15, 0.07, 0.05, 0.04, 0.03, 0.025, 0.022, 0.02, 0.02, 0.02, 0.02]
PREMIUM_RATE_POW_YEARLY_SEAS = 0.5
PREMIUM_RATE_POW_WEEKLY_SEAS = 0.25
PREMIUM_RATE_POW_HOLIDAY = 0.25
ERROR_PREMS_STD_DEV = 0.05

COEFS_SHAPE_OFFERS_DAYS = [0.8, 1.0, 0.7, 0.5, 0.3, 0.2, 0.2]
OFFERS_SCHEDULE = [
    {'start': '2015-03-08', 'days': 3, 'factor': 0.2},
    {'start': '2015-06-01', 'days': 3, 'factor': 0.2},
    {'start': '2015-11-24', 'days': 3, 'factor': 0.3},
    {'start': '2015-12-23', 'days': 2, 'factor': 0.25},
    {'start': '2016-02-13', 'days': 3, 'factor': 0.2},
    {'start': '2016-03-01', 'days': 3, 'factor': 0.2},
    {'start': '2016-11-22', 'days': 3, 'factor': 0.35},
    {'start': '2016-12-22', 'days': 4, 'factor': 0.2},
    {'start': '2017-03-01', 'days': 5, 'factor': 0.2},
    {'start': '2017-06-01', 'days': 3, 'factor': 0.25},
    {'start': '2017-11-21', 'days': 3, 'factor': 0.35},
    {'start': '2017-12-23', 'days': 2, 'factor': 0.25},
    {'start': '2018-03-01', 'days': 5, 'factor': 0.2},
    {'start': '2018-06-01', 'days': 3, 'factor': 0.25},
    {'start': '2018-07-01', 'days': 3, 'factor': 0.2},
    {'start': '2018-11-20', 'days': 3, 'factor': 0.33},
    {'start': '2018-12-23', 'days': 2, 'factor': 0.22},
    {'start': '2019-02-10', 'days': 5, 'factor': 0.22},
    {'start': '2019-07-01', 'days': 3, 'factor': 0.25},
    {'start': '2019-09-01', 'days': 3, 'factor': 0.2},
    {'start': '2019-11-26', 'days': 3, 'factor': 0.35},
    {'start': '2019-12-23', 'days': 2, 'factor': 0.23},
    {'start': '2020-02-10', 'days': 5, 'factor': 0.2},
    {'start': '2020-03-01', 'days': 3, 'factor': 0.25},
    {'start': '2020-07-01', 'days': 3, 'factor': 0.2},
    {'start': '2020-09-01', 'days': 3, 'factor': 0.15},
    {'start': '2020-11-24', 'days': 3, 'factor': 0.36},
    {'start': '2020-12-23', 'days': 2, 'factor': 0.25},
    {'start': '2021-02-10', 'days': 5, 'factor': 0.2},
    {'start': '2021-03-01', 'days': 3, 'factor': 0.25},
    {'start': '2021-07-01', 'days': 3, 'factor': 0.25},
    {'start': '2021-09-01', 'days': 3, 'factor': 0.25},
    {'start': '2021-11-24', 'days': 3, 'factor': 0.33},
    {'start': '2021-12-23', 'days': 2, 'factor': 0.27},
]
np.random.seed(1)


# UDFs
# ******************************************************************************
def get_b_splines(values, name, degrees_freedom=None, degree=None, include_intercept=False):

    if degrees_freedom is None:
        degrees_freedom = [10]

    if degree is None:
        degree = [3]

    df_values = pd.DataFrame({name: values})
    splines = BSplines(x=df_values, df=degrees_freedom, degree=degree,
                       include_intercept=include_intercept)
    df_splines = pd.DataFrame(splines.basis, columns=splines.col_names)
    return df_splines


def get_holiday_calendar_as_df(idx_date):
    class HolidaysCalendar(AbstractHolidayCalendar):
        rules = [
            Holiday('new_year', month=1, day=1, observance=nearest_workday),
            Holiday('mlking_day', month=1, day=1, offset=DateOffset(weekday=MO(3))),
            Holiday('good_friday', month=1, day=1, offset=[Easter(), Day(-2)]),
            Holiday('easter', month=1, day=1, offset=[Easter(), Day(0)]),
            Holiday('easter_monday', month=1, day=1, offset=[Easter(), Day(1)]),
            Holiday('memorial_day', month=5, day=31, offset=DateOffset(weekday=MO(-1))),
            Holiday('july4th', month=7, day=4),
            Holiday('thanksgiving', month=11, day=1, offset=DateOffset(weekday=TH(4))),
            Holiday('black_friday', month=11, day=1, offset=pd.DateOffset(weekday=FR(4))),
            Holiday('christmas_eve', month=12, day=24),
            Holiday('christmas', month=12, day=25),
            Holiday('new_year_eve', month=12, day=31)
        ]
    holidays = HolidaysCalendar().holidays(start=idx_date.min(), end=idx_date.max(), return_name=True).reset_index()
    holidays.columns = ['date', 'holiday']
    return holidays


def get_holiday_calendar_as_df_dummy(idx_date):
    holidays = get_holiday_calendar_as_df(idx_date)
    holidays['holiday'] = 1
    df_out = pd.merge(pd.DataFrame({'date': idx_date}), holidays, 'left', on='date')
    df_out = df_out.fillna(0)
    return df_out['holiday']


# generate seasonality terms
# ******************************************************************************
df_seas = pd.DataFrame({'date': pd.date_range(START_DATE, END_DATE)})

# - trend component
x = ((df_seas['date'] - df_seas['date'].min()).dt.days /
     (df_seas['date'].max() - df_seas['date'].min()).days)
tmp = get_b_splines(x, 'numtime', [len(COEFS_BSPLINES_TREND)+1])
df_seas = pd.concat([df_seas, tmp], axis=1)
df_seas['term_trend'] = 1 + tmp.multiply(COEFS_BSPLINES_TREND, axis=1).sum(axis=1)

# - yearly seasonality
x = np.minimum(df_seas['date'].dt.dayofyear, 365)
tmp = get_b_splines(x, 'dayofyear_regdate', [len(COEFS_BSPLINES_YEARLY_SEAS)+1])
df_seas = pd.concat([df_seas, tmp], axis=1)
df_seas['term_yearly_seas'] = 1 + tmp.multiply(COEFS_BSPLINES_YEARLY_SEAS, axis=1).sum(axis=1)

# - weekly seasonality
x = df_seas['date'].dt.weekday + 1
tmp = get_b_splines(x, 'weekday_regdate', [len(COEFS_BSPLINES_WEEKLY_SEAS)], include_intercept=True)
df_seas = pd.concat([df_seas, tmp], axis=1)
df_seas['term_weekly_seas'] = 1 + tmp.multiply(COEFS_BSPLINES_WEEKLY_SEAS, axis=1).sum(axis=1)

# - holidays for US calendar
df_seas['holiday'] = get_holiday_calendar_as_df_dummy(df_seas['date'])
df_seas['term_holiday'] = 1 + COEF_HOLIDAY * df_seas['holiday']

# - error term
df_seas['term_error'] = 1 + ERROR_USERS_STD_DEV * np.random.randn(df_seas.shape[0])

# plot all together
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_seas['date'],
                         y=(df_seas['term_trend']*df_seas['term_yearly_seas']*df_seas['term_weekly_seas']),
                         name='total', mode='lines', opacity=0.70, line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=df_seas['date'], y=df_seas['term_trend'], name='trend',
                         mode='lines', opacity=0.5, line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=df_seas['date'], y=df_seas['term_yearly_seas'], name='yearly',
                         mode='lines', opacity=0.5, line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=df_seas['date'], y=df_seas['term_weekly_seas'], name='weekly',
                         mode='lines', opacity=0.5, line=dict(color='blue', width=2)))
py.plot(fig)


# generate offers (special events, price offers, bundle offers)
# ******************************************************************************
tmp = []
for s in OFFERS_SCHEDULE:
    for d in range(0, s['days']):
        tmp.append({'date': pd.to_datetime(s['start']) + pd.DateOffset(d),
                    'offer_effect': s['factor'] * COEFS_SHAPE_OFFERS_DAYS[d]})
df_offers = pd.DataFrame(tmp)
df_offers['offer_effect'] = df_offers['offer_effect']

# plot
fig = go.Figure()
fig.add_trace(go.Bar(x=df_offers['date'], y=df_offers['offer_effect']))
fig.update_layout(xaxis_range=['2020-01-01', '2020-12-31'])
py.plot(fig)


# generate shape of premium rate by age
# ******************************************************************************
# use b-splines for non-linear shape in the first 90 days
tmp = get_b_splines(np.arange(start=0, stop=90, step=1), 'age', [len(COEFS_BSPLINES_AGE_0_90)], include_intercept=True)
df_age_0_90 = pd.DataFrame(
    {'age': np.arange(start=0, stop=90, step=1),
     'prem_shape_age': tmp.multiply(COEFS_BSPLINES_AGE_0_90, axis=1).sum(axis=1)})
last_90 = np.min(df_age_0_90['prem_shape_age'])

# use linear relations after 90 days (for 10 years time frame)
df_age = pd.concat(
    [df_age_0_90,
     pd.DataFrame(
         {'age': np.arange(start=90, stop=180, step=1),
          'prem_shape_age': np.linspace(last_90, 0.5 * last_90, 90)}),
     pd.DataFrame(
         {'age': np.arange(start=180, stop=365, step=1),
          'prem_shape_age': np.linspace(0.5 * last_90, 0.20 * last_90, 185)}),
     pd.DataFrame(
         {'age': np.arange(start=365, stop=365*10, step=1),
          'prem_shape_age': np.linspace(0.20 * last_90, 0.10 * last_90, 365*9)})
     ])
df_age['prem_rate_age'] = PREMIUM_RATE_TOTAL * df_age['prem_shape_age'] / np.sum(df_age['prem_shape_age'])

# plot the shape on 0 to 90 days
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_age['age'], y=df_age['prem_rate_age'], name='premium rate',
                         mode='lines', opacity=0.60, line=dict(color='black', width=2)))
fig.update_layout(xaxis_range=[0, 90])
py.plot(fig)

# plot the shape on 0 to 365 days
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_age['age'], y=df_age['prem_rate_age'], name='premium rate',
                         mode='lines', opacity=0.60, line=dict(color='black', width=2)))
fig.update_layout(xaxis_range=[0, 365])
py.plot(fig)


# generate cohorts of users by registration date
# ******************************************************************************
df_users = pd.DataFrame({'reg_date': pd.date_range(START_DATE, END_DATE)})

# users (use multiplicative model)
df_users['users'] = (
    np.round(
        SCALE_USERS
        * df_seas['term_trend'] ** USERS_POW_TREND
        * df_seas['term_yearly_seas'] ** USERS_POW_YEARLY_SEAS
        * df_seas['term_weekly_seas'] ** USERS_POW_WEEKLY_SEAS
        * df_seas['term_holiday'] ** USERS_POW_HOLIDAY
        * df_seas['term_error'])
)

# plot users by registration date
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_users['reg_date'], y=df_users['users'], name='users',
                         mode='lines', opacity=0.60, line=dict(color='black', width=2)))
py.plot(fig)


# generate the final table with premiums by registration date and upgrade date
# ******************************************************************************
# cartesian product between reg dates and upgrade dates
df = pd.DataFrame(itertools.product(
    pd.date_range(START_DATE, END_DATE),
    pd.date_range(START_DATE, END_DATE)))
df.columns = ['reg_date', 'upgrade_date']
df = df.loc[df['upgrade_date'] >= df['reg_date']]
df['age'] = (df['upgrade_date'] - df['reg_date']).dt.days

# join components by keys: reg_date, upgrade_date, age
df = pd.merge(df, df_users[['reg_date', 'users']], on=['reg_date'])
df = pd.merge(df, df_age[['age', 'prem_rate_age']], on=['age'])
df = pd.merge(df, df_offers[['date', 'offer_effect']], how='left', left_on=['upgrade_date'], right_on=['date'])
df['offer_effect'] = df['offer_effect'].replace(np.nan, 0)
df.drop(columns=['date'], inplace=True)
df = pd.merge(df, df_seas[['date', 'term_yearly_seas', 'term_weekly_seas', 'term_holiday']],
              how='left', left_on=['upgrade_date'], right_on=['date'])
df.drop(columns=['date'], inplace=True)

# compute premium rates using a multiplicative model
# use the same seasonality terms as for users but assume milder forms (powers < 1)
df['prem_rate'] = (
        df['prem_rate_age']
        * df['term_yearly_seas'] ** PREMIUM_RATE_POW_YEARLY_SEAS
        * df['term_weekly_seas'] ** PREMIUM_RATE_POW_WEEKLY_SEAS
        * df['term_holiday'] ** PREMIUM_RATE_POW_HOLIDAY
        * (1 + df['offer_effect'])
        * (1 + np.random.randn(df.shape[0]) * ERROR_PREMS_STD_DEV)
)
df['lambda'] = df['users'] * df['prem_rate']
df['premiums'] = np.random.poisson(df['lambda'])
# ******************************************************************************


# plot the aggregated series of premiums by upgrade date
# ******************************************************************************
tmp = df.groupby(['upgrade_date'], as_index=False).agg({'premiums': [sum]})
tmp.columns = ['upgrade_date', 'premiums']
fig = go.Figure()
fig.add_trace(go.Scatter(x=tmp['upgrade_date'], y=tmp['premiums'], name='premiums',
                         mode='lines', opacity=0.60, line=dict(color='black', width=2)))
py.plot(fig)


# save data as csv
# ******************************************************************************
dirtmp = os.path.dirname(os.path.realpath(__file__))
cols = ['reg_date', 'upgrade_date', 'age', 'premiums', 'offer_effect',
        'term_yearly_seas', 'term_weekly_seas', 'term_holiday']
df[cols].to_csv(f'{dirtmp}/df.csv', index=False, float_format='%.4f')

