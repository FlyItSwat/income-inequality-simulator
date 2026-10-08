"""Streamlit dashboard for a stylized income redistribution simulation."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from economics import decile_summary, gini, lorenz, simulate

st.set_page_config(page_title='Income Inequality Simulator', page_icon='⚖️', layout='wide')
st.title('⚖️ Income Inequality & Tax Redistribution Simulator')
st.caption('Explore stylized household income distributions, marginal taxes and universal cash transfers.')

@st.cache_data
def example_data():
    rng = np.random.default_rng(2026)
    values = np.round(rng.lognormal(mean=np.log(350000), sigma=0.85, size=1000), 0)
    return pd.DataFrame({'annual_income_inr': values})

with st.sidebar:
    st.header('Income data')
    uploaded = st.file_uploader('Upload CSV with annual_income_inr column', type='csv')
    st.header('Marginal tax rates')
    rate1 = st.slider('₹0–₹3 lakh', 0, 60, 0, 1)
    rate2 = st.slider('₹3–₹7 lakh', 0, 60, 10, 1)
    rate3 = st.slider('Above ₹7 lakh', 0, 60, 25, 1)
    transfer_share = st.slider('Share of tax revenue returned equally (%)', 0, 100, 100, 5)
    st.caption('Illustrative tax brackets, NOT actual Indian tax law.')

try:
    source = pd.read_csv(uploaded) if uploaded is not None else example_data()
    if 'annual_income_inr' not in source.columns:
        raise ValueError('CSV needs a column named annual_income_inr.')
    incomes = pd.to_numeric(source['annual_income_inr'], errors='raise').to_numpy(dtype=float)
    results = simulate(incomes, [0, 300000, 700000],
                       [rate1/100, rate2/100, rate3/100], transfer_share/100)
    before_gini = gini(results['income_before'])
    after_gini = gini(results['income_after'])
    revenue = results['tax'].sum()
    cols = st.columns(4)
    cols[0].metric('Gini before', f'{before_gini:.3f}')
    cols[1].metric('Gini after', f'{after_gini:.3f}', delta=f'{after_gini-before_gini:+.3f}', delta_color='inverse')
    cols[2].metric('Total tax collected', f'₹{revenue:,.0f}')
    cols[3].metric('Transfer per household', f'₹{results.transfer.iloc[0]:,.0f}')

    st.subheader('Lorenz curves')
    fig = go.Figure()
    for label, field in [('Before tax and transfers', 'income_before'),
                         ('After tax and transfers', 'income_after')]:
        population, income_share = lorenz(results[field])
        fig.add_trace(go.Scatter(x=population, y=income_share, name=label, mode='lines'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Perfect equality',
                             line=dict(dash='dash', color='gray')))
    fig.update_layout(xaxis_title='Cumulative share of households',
                      yaxis_title='Cumulative share of income',
                      xaxis=dict(range=[0,1], tickformat='.0%'),
                      yaxis=dict(range=[0,1], tickformat='.0%'))
    st.plotly_chart(fig, use_container_width=True)

    deciles = decile_summary(results)
    st.subheader('Average income by pre-policy income decile')
    plot_df = deciles.melt(id_vars='decile', value_vars=['mean_before','mean_after'],
                           var_name='Measure', value_name='Annual income (₹)')
    st.plotly_chart(px.bar(plot_df, x='decile', y='Annual income (₹)', color='Measure',
                            barmode='group'), use_container_width=True)
    st.dataframe(deciles.style.format({'mean_before':'₹{:,.0f}',
                                      'mean_after':'₹{:,.0f}',
                                      'mean_tax':'₹{:,.0f}',
                                      'mean_transfer':'₹{:,.0f}',
                                      'mean_net_change':'₹{:+,.0f}'}),
                 use_container_width=True, hide_index=True)
    st.download_button('Download household simulation (CSV)',
                       results.to_csv(index=False), 'household_policy_results.csv', 'text/csv')
    st.download_button('Download decile summary (CSV)',
                       deciles.to_csv(index=False), 'decile_summary.csv', 'text/csv')
    st.info('Model assumptions: households have equal weights; annual incomes are nonnegative; '
            'all taxes are collected; selected tax receipts are redistributed as identical '
            'cash transfers; no changes to work, savings, compliance or prices are modeled. '
            'Sample incomes are synthetic and not representative of India. '
            'This is not a model of actual Indian income-tax law.')
except (ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
    st.error(f'Unable to analyze income data: {exc}')
