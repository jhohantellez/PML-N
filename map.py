import pandas as pd
import pickle
import json
import folium
from branca.element import Template, MacroElement

with open('rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)
with open('rf_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


df = pd.read_excel('UNGRD_Cleaned2.xlsx')

with open('municipalitiescolombia.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)

geo_codes = {feat['properties']['MPIO_CCNCT'] for feat in geojson_data['features']}

features = ['MUNICIPALITY_EVENT_COUNT', 'EVENT_DIVERSITY', 'RAINY_SEASON',
            'YEAR', 'MONTH', 'DEPARTMENT', 'EVENT']

X          = df[features]
preds_enc  = rf.predict(X)
proba      = rf.predict_proba(X)

high_idx   = list(le.classes_).index('High')
medium_idx = list(le.classes_).index('Medium')
low_idx    = list(le.classes_).index('Low')

df['PRED_VULN']   = le.inverse_transform(preds_enc)
df['PROB_HIGH']   = proba[:, high_idx]
df['PROB_MEDIUM'] = proba[:, medium_idx]
df['PROB_LOW']    = proba[:, low_idx]

df['DIVIPOLA_STR'] = df['DIVIPOLA'].astype(str).str.zfill(5)
df_matched = df[df['DIVIPOLA_STR'].isin(geo_codes)].copy()

print(f"Total records:              {len(df):,}")
print(f"Records with valid DIVIPOLA:{len(df_matched):,} ({len(df_matched)/len(df)*100:.1f}%)")

grouped = (
    df_matched
    .groupby('DIVIPOLA_STR')
    .agg(
        PREDICTED_VULNERABILITY=('PRED_VULN',          lambda x: x.mode()[0]),
        PROB_HIGH               =('PROB_HIGH',          'mean'),
        PROB_MEDIUM             =('PROB_MEDIUM',        'mean'),
        PROB_LOW                =('PROB_LOW',           'mean'),
        N_EVENTS                =('PRED_VULN',          'count'),
        EVENT_DIVERSITY         =('EVENT_DIVERSITY',    'mean'),
    )
    .reset_index()
)

print(f"Municipalities with prediction: {len(grouped):,}")
print("\nPredicted vulnerability distribution:")
print(grouped['PREDICTED_VULNERABILITY'].value_counts().to_string())

pred_dict = grouped.set_index('DIVIPOLA_STR')['PREDICTED_VULNERABILITY'].to_dict()
prob_dict = grouped.set_index('DIVIPOLA_STR').to_dict(orient='index')


color_map = {
    'High':    '#d73027',
    'Medium':  '#fc8d59',
    'Low':     '#1a9850',
    'No Data': '#d3d3d3',
}

mapa = folium.Map(location=[4.6, -74.1], zoom_start=6, tiles='cartodbpositron')

for feature in geojson_data['features']:
    code          = feature['properties']['MPIO_CCNCT']
    vulnerability = pred_dict.get(code, 'No Data')
    color         = color_map[vulnerability]

    info   = prob_dict.get(code, {})
    prob_h = info.get('PROB_HIGH',      0)
    prob_m = info.get('PROB_MEDIUM',    0)
    prob_l = info.get('PROB_LOW',       0)
    n_ev   = info.get('N_EVENTS',       0)
    div    = info.get('EVENT_DIVERSITY', 0)

    mun_name  = feature['properties']['MPIO_CNMBR']
    dept_name = feature['properties']['DPTO_CNMBR']

    bar_h = f"<div style='width:{prob_h*100:.0f}%;background:#d73027;height:8px;display:inline-block'></div>"
    bar_m = f"<div style='width:{prob_m*100:.0f}%;background:#fc8d59;height:8px;display:inline-block'></div>"
    bar_l = f"<div style='width:{prob_l*100:.0f}%;background:#1a9850;height:8px;display:inline-block'></div>"

    if vulnerability == 'No Data':
        popup_html = f"""
        <div style='font-family:Arial;font-size:13px;width:260px'>
          <b style='font-size:15px'>{mun_name}</b><br>
          <span style='color:#666'>{dept_name}</span><br>
          <hr style='margin:6px 0'>
          <i>No data available in the model</i><br>
          <small>DIVIPOLA: {code}</small>
        </div>"""
    else:
        vuln_color = color_map[vulnerability]
        popup_html = f"""
        <div style='font-family:Arial;font-size:13px;width:280px'>
          <b style='font-size:15px'>{mun_name}</b><br>
          <span style='color:#666'>{dept_name}</span>
          <span style='float:right;color:#999;font-size:11px'>DIVIPOLA: {code}</span>
          <hr style='margin:6px 0'>
          <b>Predicted Vulnerability:</b>
          <span style='background:{vuln_color};color:white;padding:2px 8px;
                       border-radius:4px;margin-left:6px'>{vulnerability}</span>
          <br><br>
          <b>Probabilities:</b><br>
          <div style='margin:4px 0'>High &nbsp;&nbsp; {bar_h} <small>{prob_h:.2%}</small></div>
          <div style='margin:4px 0'>Medium {bar_m} <small>{prob_m:.2%}</small></div>
          <div style='margin:4px 0'>Low &nbsp;&nbsp;&nbsp; {bar_l} <small>{prob_l:.2%}</small></div>
          <hr style='margin:6px 0'>
          <small>📊 Records used: <b>{int(n_ev)}</b> &nbsp;|&nbsp; Event diversity: <b>{div:.1f}</b></small>
        </div>"""

    folium.GeoJson(
        feature,
        style_function=lambda x, c=color: {
            'fillColor': c,
            'color': 'white',
            'weight': 0.4,
            'fillOpacity': 0.75,
        },
        highlight_function=lambda x: {
            'weight': 2,
            'color': '#333',
            'fillOpacity': 0.95,
        },
        tooltip=folium.Tooltip(
            f"{mun_name} — {vulnerability}",
            sticky=True
        ),
        popup=folium.Popup(popup_html, max_width=320),
    ).add_to(mapa)


legend_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed; bottom: 40px; right: 20px; z-index: 1000;
    background: white; padding: 14px 18px; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25); font-family: Arial; font-size: 13px;
">
  <b style="font-size:14px">🗺 Municipal Vulnerability</b><br>
  <small style="color:#888">Random Forest Prediction</small>
  <hr style="margin:8px 0">
  <div><span style="background:#d73027;width:16px;height:16px;display:inline-block;
       border-radius:3px;vertical-align:middle;margin-right:8px"></span>High</div>
  <div style="margin-top:5px"><span style="background:#fc8d59;width:16px;height:16px;display:inline-block;
       border-radius:3px;vertical-align:middle;margin-right:8px"></span>Medium</div>
  <div style="margin-top:5px"><span style="background:#1a9850;width:16px;height:16px;display:inline-block;
       border-radius:3px;vertical-align:middle;margin-right:8px"></span>Low</div>
  <div style="margin-top:5px"><span style="background:#d3d3d3;width:16px;height:16px;display:inline-block;
       border-radius:3px;vertical-align:middle;margin-right:8px"></span>No Data</div>
</div>
{% endmacro %}
"""
legend = MacroElement()
legend._template = Template(legend_html)
mapa.get_root().add_child(legend)

mapa.save('vulnerability_map.html')
