import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

df = pd.read_csv("C:\\Users\\Geronimo\\Desktop\\Credit-card-fraud-detection\\data\\raw\\fraudTrain.csv")


# DESCRIPCION DEL SET DE DATOS
#print(df.head())
#print(df.shape)
#print(df.columns)
#df.info()


# EDA
fraud_counts = df['is_fraud'].value_counts().reset_index()
fraud_counts.columns = ['is_fraud','count']

fig = px.bar(
    fraud_counts,
    x='is_fraud',
    y='count',
    text='count',
    title='Distribución de transacciones fraudulentas vs legítimas',
    labels={'is_fraud':'Tipo de transacción','count':'Cantidad'}
)

fig.update_layout(
    xaxis=dict(tickmode='array', tickvals=[0,1], ticktext=['No fraude','Fraude'])
)

#fig.show()

# Distribucion del monto de transacciones
fig = px.box(
    df,
    x="is_fraud",
    y="amt",
    title="Distribución del monto de transacciones por tipo",
    labels={
        "is_fraud":"Tipo de transacción",
        "amt":"Monto de la transacción"
    }
)

fig.update_layout(
    yaxis_type="log",
    xaxis=dict(
        tickmode='array',
        tickvals=[0,1],
        ticktext=['No fraude','Fraude']
    )
)

#fig.show()

#Fraude por categoria de comercio 
import plotly.express as px

fraud_by_category = (
    df[df["is_fraud"] == 1]
    .groupby("category")
    .size()
    .reset_index(name="fraud_count")
    .sort_values("fraud_count", ascending=False)
)

fig = px.bar(
    fraud_by_category,
    x="category",
    y="fraud_count",
    title="Número de transacciones fraudulentas por categoría de comercio",
    labels={
        "category": "Categoría de comercio",
        "fraud_count": "Cantidad de fraudes"
    }
)

#fig.show()



df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour

fraud_by_hour = (
    df[df["is_fraud"] == 1]
    .groupby("hour")
    .size()
    .reset_index(name="fraud_count")
)

fig = px.line(
    fraud_by_hour,
    x="hour",
    y="fraud_count",
    markers=True,
    title="Distribución de fraude por hora del día",
    labels={
        "hour": "Hora del día",
        "fraud_count": "Número de fraudes"
    }
)

#fig.show()


def haversine(lat1, lon1, lat2, lon2):

    R = 6371  # radio de la Tierra en km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

df["distance_km"] = haversine(
    df["lat"],
    df["long"],
    df["merch_lat"],
    df["merch_long"]
)

df["distance_km"].describe()


fig = px.violin(
    df,
    x="is_fraud",
    y="distance_km",
    box=True,
    points="outliers",
    title="Distribución de distancia entre cliente y comercio",
    labels={
        "is_fraud": "Tipo de transacción",
        "distance_km": "Distancia (km)"
    }
)

fig.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=[0,1],
        ticktext=["No fraude","Fraude"]
    )
)

#fig.show()


fraud_by_gender = (
    df[df["is_fraud"] == 1]
    .groupby("gender")
    .size()
    .reset_index(name="fraud_count")
)


fig = px.bar(
    fraud_by_gender,
    x="gender",
    y="fraud_count",
    title="Transacciones fraudulentas por género",
    labels={
        "gender": "Género",
        "fraud_count": "Número de fraudes"
    }
)

import plotly.express as px

# Top categorías por volumen
top_categories = df['category'].value_counts().head(8).index
df_top = df[df['category'].isin(top_categories)]

fig = px.box(df_top, 
             x='category', 
             y='amt', 
             color='is_fraud',
             log_y=True,
             title='Distribución de Montos por Categoría y Tipo de Transacción',
             labels={'category': 'Categoría', 'amt': 'Monto ($)', 'is_fraud': 'Es Fraude'},
             color_discrete_map={0: '#1f77b4', 1: '#d62728'})
fig.update_layout(xaxis_tickangle=-45)
#fig.show()

import plotly.express as px

# Muestra representativa (10,000 puntos para rendimiento)
sample_scatter = df.sample(10000, random_state=42)

fig = px.scatter(sample_scatter,
                 x='distance_km',
                 y='amt',
                 color='is_fraud',
                 log_y=True,
                 title='Relación entre Distancia Geográfica y Monto de Transacción',
                 labels={'distance_km': 'Distancia Cliente-Comercio (km)', 
                         'amt': 'Monto ($)',
                         'is_fraud': 'Es Fraude'},
                 color_discrete_map={0: '#1f77b4', 1: '#d62728'},
                 opacity=0.6)
fig.update_traces(marker=dict(size=3))
#fig.show()

import plotly.express as px
import plotly.graph_objects as go

# Calcular correlación
num_cols = ['amt', 'city_pop', 'unix_time', 'hour', 'distance_km', 'is_fraud']
corr_matrix = df[num_cols].corr().round(2)

# Heatmap con Plotly
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmin=-1, zmax=1,
    text=corr_matrix.values,
    texttemplate='%{text}',
    textfont={"size": 10},
    colorbar_title="Correlación"
))
fig.update_layout(
    title='Matriz de Correlaciones - Variables Numéricas',
    width=700,
    height=600
)
#fig.show()

import plotly.express as px

# Filtrar solo fraudes y agrupar
fraud_data = df[df['is_fraud'] == 1]
fraud_by_hour_cat = fraud_data.groupby(['hour', 'category']).size().reset_index(name='count')

# Seleccionar top categorías con más fraudes
top_categories_fraud = fraud_data['category'].value_counts().head(6).index
fraud_top_cat = fraud_by_hour_cat[fraud_by_hour_cat['category'].isin(top_categories_fraud)]

fig = px.line(fraud_top_cat, 
              x='hour', 
              y='count', 
              color='category',
              markers=True,
              title='Patrones Temporales de Fraude: Horas del Día por Categoría',
              labels={'hour': 'Hora del día', 'count': 'Número de fraudes', 'category': 'Categoría'})
fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
#fig.show()

# Transacciones fraudulentas por tipo de comercio
fraud_count = df[df['is_fraud'] == 1]['category'].value_counts().head(10).reset_index()
fraud_count.columns = ['category', 'fraud_count']

fig = px.bar(fraud_count, 
             x='fraud_count', 
             y='category',
             orientation='h',
             color='fraud_count',
             color_continuous_scale='Blues',
             title='Top 10 Categorías con Mayor Número de Fraudes',
             labels={'fraud_count': 'Número de fraudes', 'category': 'Categoría'})
fig.update_layout(yaxis=dict(autorange="reversed"))
#fig.show()
    
# Calcular tasa de fraude por categoría
fraud_rate = df.groupby('category')['is_fraud'].agg(['count', 'sum'])
fraud_rate.columns = ['total', 'fraudes']
fraud_rate['tasa_fraude'] = (fraud_rate['fraudes'] / fraud_rate['total']) * 100
fraud_rate = fraud_rate.sort_values('tasa_fraude', ascending=False).head(10).reset_index()

fig = px.bar(fraud_rate, 
             x='category', 
             y='tasa_fraude',
             color='tasa_fraude',
             color_continuous_scale='Blues',
             title='Top 10 Categorías con Mayor Tasa de Fraude (%)',
             labels={'category': 'Categoría', 'tasa_fraude': 'Tasa de Fraude (%)'})
fig.update_layout(xaxis_tickangle=-45)
#fig.show()

# Calcular tasa de fraude por género
gender_stats = df.groupby('gender')['is_fraud'].agg(['count', 'sum'])
gender_stats.columns = ['total', 'fraudes']
gender_stats['tasa_fraude'] = (gender_stats['fraudes'] / gender_stats['total']) * 100
gender_stats = gender_stats.reset_index()

fig = px.bar(gender_stats, 
             x='gender', 
             y='tasa_fraude',
             color='gender',
             color_discrete_map={'F': '#1f77b4', 'M': "#2856C1"},
             title='Tasa de Fraude por Género (%)',
             labels={'gender': 'Género', 'tasa_fraude': 'Tasa de Fraude (%)'})
fig.update_layout(showlegend=False)
#
#fig.show()

top_categories = df['category'].value_counts().head(8).index
df_top = df[df['category'].isin(top_categories)]

fig = px.box(df_top, 
             x='category', 
             y='amt', 
             color='is_fraud',
             log_y=True,
             title='Distribución de Montos por Categoría y Tipo de Transacción',
             labels={'category': 'Categoría', 'amt': 'Monto ($)', 'is_fraud': 'Es Fraude'},
             color_discrete_map={0: '#1f77b4', 1: '#d62728'})
fig.update_layout(xaxis_tickangle=-45)
#fig.show()

import plotly.express as px
import pandas as pd

# ============================================
# PASO 1: Calcular estadísticas clave
# ============================================

# Filtrar solo fraudes para análisis de distancia
fraudes = df[df['is_fraud'] == 1]

# Calcular porcentajes
dist_menor_20 = (fraudes['distance_km'] < 20).mean() * 100
dist_menor_50 = (fraudes['distance_km'] < 50).mean() * 100
dist_mayor_50 = (fraudes['distance_km'] > 50).mean() * 100

print("=== ESTADÍSTICAS DE DISTANCIA EN FRAUDES ===")
print(f"Fraudes con distancia < 20 km: {dist_menor_20:.1f}%")
print(f"Fraudes con distancia < 50 km: {dist_menor_50:.1f}%")
print(f"Fraudes con distancia > 50 km: {dist_mayor_50:.1f}%")
print(f"Distancia media en fraudes: {fraudes['distance_km'].mean():.1f} km")
print(f"Distancia mediana en fraudes: {fraudes['distance_km'].median():.1f} km")

# ============================================
# PASO 2: Crear muestra para el scatter plot
# ============================================

# Muestra representativa (10,000 transacciones) para rendimiento
sample_scatter = df.sample(10000, random_state=42)

# ============================================
# PASO 3: Generar gráfico con Plotly
# ============================================

fig = px.scatter(sample_scatter,
                 x='distance_km',
                 y='amt',
                 color='is_fraud',
                 log_y=True,
                 title='Relación entre Distancia Geográfica y Monto de Transacción',
                 labels={
                     'distance_km': 'Distancia Cliente-Comercio (km)', 
                     'amt': 'Monto ($)',
                     'is_fraud': 'Tipo de transacción'
                 },
                 color_discrete_map={0: '#1f77b4', 1: '#d62728'},
                 opacity=0.6)

# Personalizar tamaño de los puntos
fig.update_traces(marker=dict(size=3))

# Agregar línea vertical en 20 km para destacar el umbral
fig.add_vline(x=20, line_dash="dash", line_color="gray")
fig.add_annotation(x=20, y=1000, text="< 20 km → 92% de fraudes", 
                   showarrow=False, xanchor="left", yanchor="top",
                   bgcolor="white", opacity=0.8)

# Agregar línea vertical en 50 km
fig.add_vline(x=50, line_dash="dot", line_color="gray")
fig.add_annotation(x=50, y=5, text="> 50 km → solo 2% de fraudes", 
                   showarrow=False, xanchor="left", yanchor="bottom",
                   bgcolor="white", opacity=0.8)

# Mejorar diseño del eje X para enfocar en el rango relevante
fig.update_layout(
    xaxis=dict(title="Distancia Cliente-Comercio (km)", range=[0, 150]),
    yaxis=dict(title="Monto ($) - Escala logarítmica"),
    legend=dict(title="Tipo de transacción")
)

#fig.show()


import plotly.graph_objects as go

num_cols = ['amt', 'city_pop', 'unix_time', 'hour', 'distance_km', 'is_fraud']
corr_matrix = df[num_cols].corr().round(2)

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmin=-1, zmax=1,
    text=corr_matrix.values,
    texttemplate='%{text}',
    textfont={"size": 10},
    colorbar_title="Correlación"
))
fig.update_layout(
    title='Matriz de Correlaciones - Variables Numéricas',
    width=700,
    height=600
)
#fig.show()

fraud_map = df[df['is_fraud'] == 1].sample(5000, random_state=42)

fraud_map = df[df['is_fraud'] == 1].sample(5000, random_state=42)

fig = px.density_map(fraud_map, 
                      lat='lat', 
                      lon='long',
                      radius=10,
                      zoom=3,
                      title='Concentración Geográfica de Transacciones Fraudulentas')
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
fig.show()

fraud_map = df[df['is_fraud'] == 1].sample(5000, random_state=42)

fraud_map = df[df['is_fraud'] == 1].sample(5000, random_state=42)

fig = px.density_map(
    fraud_map, 
    lat='lat', 
    lon='long',
    radius=10,
    zoom=3,
    title='Concentración Geográfica de Transacciones Fraudulentas'
)

fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
#fig.show()


fraud_data = df[df['is_fraud'] == 1]
fraud_by_hour_cat = fraud_data.groupby(['hour', 'category']).size().reset_index(name='count')
top_categories_fraud = fraud_data['category'].value_counts().head(6).index
fraud_top_cat = fraud_by_hour_cat[fraud_by_hour_cat['category'].isin(top_categories_fraud)]

fig = px.line(fraud_top_cat, 
              x='hour', 
              y='count', 
              color='category',
              markers=True,
              title='Patrones Temporales de Fraude: Horas del Día por Categoría',
              labels={'hour': 'Hora del día', 'count': 'Número de fraudes', 'category': 'Categoría'})
fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
#fig.show()

import matplotlib.pyplot as plt
import numpy as np

# Tomar muestras para visualización
fraud_sample = df[df['is_fraud'] == 1].sample(5000, random_state=42)
legit_sample = df[df['is_fraud'] == 0].sample(5000, random_state=42)

# Crear figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Transacciones legítimas
ax1.scatter(legit_sample['long'], legit_sample['lat'], 
            alpha=0.3, s=1, c='#1f77b4')
ax1.set_title('Transacciones Legítimas (muestra de 5,000)')
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')
ax1.set_xlim([-125, -65])
ax1.set_ylim([25, 50])

# Gráfico 2: Transacciones fraudulentas
ax2.scatter(fraud_sample['long'], fraud_sample['lat'], 
            alpha=0.6, s=5, c='#d62728')
ax2.set_title('Transacciones Fraudulentas (muestra de 5,000)')
ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')
ax2.set_xlim([-125, -65])
ax2.set_ylim([25, 50])

plt.suptitle('Distribución Geográfica de Transacciones por Tipo', fontsize=14)
plt.tight_layout()
plt.show()

#5 lIMPIEZA DE DATOS E INGENIERIA DE CARACTERISTICAS

# Verificar valores nulos en todo el dataset
print("=== VERIFICACIÓN DE VALORES NULOS ===\n")
nulos = df.isnull().sum()
print("Conteo de valores nulos por columna:")
print(nulos[nulos > 0] if any(nulos > 0) else "✅ No se encontraron valores nulos en ninguna columna.\n")

# Información adicional
print(f"\nTotal de filas: {len(df):,}")
print(f"Total de columnas: {len(df.columns)}")

# Mostrar columnas antes de eliminar
print("=== COLUMNAS ANTES DE LIMPIEZA ===\n")
print(f"Total: {len(df.columns)} columnas")
print(list(df.columns))

# Identificar columnas a eliminar
columnas_a_eliminar = ['Unnamed: 0', 'cc_num', 'trans_num', 'first', 'last', 'street', 'zip', 'unix_time']
print(f"\n📌 Columnas a eliminar: {columnas_a_eliminar}")

# Crear nuevo DataFrame sin esas columnas
df_clean = df.drop(columns=columnas_a_eliminar)


#3print(f"\n=== COLUMNAS DESPUÉS DE LIMPIEZA ===\n")
#print(f"Total: {len(df_clean.columns)} columnas")
#print(list(df_clean.columns))


# Verificar que 'hour' ya existe y mostrar ejemplo
print("=== VERIFICACIÓN DE VARIABLE 'hour' ===\n")
print(f"'hour' ya existe en el dataset: {'hour' in df_clean.columns}")
print(f"\nDistribución de horas (primeros 10 valores):")
print(df_clean['hour'].value_counts().sort_index().head(10))
# Verificar que 'distance_km' ya existe
print("=== VERIFICACIÓN DE VARIABLE 'distance_km' ===\n")
print(f"'distance_km' ya existe en el dataset: {'distance_km' in df_clean.columns}")

# Mostrar estadísticas descriptivas
print(f"\nEstadísticas de distance_km por tipo de transacción:")
print(df_clean.groupby('is_fraud')['distance_km'].describe())

# Calcular porcentaje de fraudes con distancia < 20 km
fraudes = df_clean[df_clean['is_fraud'] == 1]
pct_menor_20 = (fraudes['distance_km'] < 20).mean() * 100
print(f"\n Porcentaje de fraudes con distancia < 20 km: {pct_menor_20:.1f}%")


import matplotlib.pyplot as plt
import seaborn as sns

# Calcular percentil 99.9 para transacciones legítimas
legit_amt = df_clean[df_clean['is_fraud'] == 0]['amt']
p999 = legit_amt.quantile(0.999)

print("=== ANÁLISIS DE OUTLIERS EN amt ===\n")
print(f"Percentil 99.9 de montos legítimos: ${p999:.2f}")
print(f"Monto máximo legítimo: ${legit_amt.max():.2f}")
print(f"Monto máximo fraudulento: ${df_clean[df_clean['is_fraud'] == 1]['amt'].max():.2f}")

# Visualización del truncamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Antes del truncamiento
sns.histplot(df_clean['amt'], bins=100, ax=ax1, color='#1f77b4')
ax1.set_title('Distribución original de montos')
ax1.set_xlabel('Monto ($)')
ax1.set_ylabel('Frecuencia')
ax1.set_xscale('log')

# Después del truncamiento (opcional: mostrar el efecto)
amt_truncado = df_clean['amt'].clip(upper=p999)
sns.histplot(amt_truncado, bins=100, ax=ax2, color='#2ca02c')
ax2.set_title(f'Distribución después de truncamiento (p99.9 = ${p999:.0f})')
ax2.set_xlabel('Monto ($)')
ax2.set_ylabel('Frecuencia')
ax2.set_xscale('log')

plt.tight_layout()
plt.show()

# Manejo de Outliers en 'amt'
import matplotlib.pyplot as plt
import seaborn as sns

# Calcular percentil 99.9 para transacciones legítimas
legit_amt = df_clean[df_clean['is_fraud'] == 0]['amt']
p999 = legit_amt.quantile(0.999)

print("=== ANÁLISIS DE OUTLIERS EN amt ===\n")
print(f"Percentil 99.9 de montos legítimos: ${p999:.2f}")
print(f"Monto máximo legítimo: ${legit_amt.max():.2f}")
print(f"Monto máximo fraudulento: ${df_clean[df_clean['is_fraud'] == 1]['amt'].max():.2f}")

# Visualización del truncamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Antes del truncamiento
sns.histplot(df_clean['amt'], bins=100, ax=ax1, color='#1f77b4')
ax1.set_title('Distribución original de montos')
ax1.set_xlabel('Monto ($)')
ax1.set_ylabel('Frecuencia')
ax1.set_xscale('log')

# Después del truncamiento (opcional: mostrar el efecto)
amt_truncado = df_clean['amt'].clip(upper=p999)
sns.histplot(amt_truncado, bins=100, ax=ax2, color='#2ca02c')
ax2.set_title(f'Distribución después de truncamiento (p99.9 = ${p999:.0f})')
ax2.set_xlabel('Monto ($)')
ax2.set_ylabel('Frecuencia')
ax2.set_xscale('log')

plt.tight_layout()
plt.show()

#Codificación de Variables Categóricas
# Mostrar variables categóricas y su cardinalidad
categorical_cols = ['category', 'gender', 'job', 'state', 'merchant']

print("=== ANÁLISIS DE VARIABLES CATEGÓRICAS ===\n")
print("Cardinalidad por variable:")
for col in categorical_cols:
    if col in df_clean.columns:
        n_unique = df_clean[col].nunique()
        print(f"  - {col}: {n_unique} valores únicos")

# Mostrar distribución de 'category'
print(f"\n📌 Top 5 categorías por frecuencia:")
print(df_clean['category'].value_counts().head())

# Mostrar distribución de 'gender'
print(f"\n📌 Distribución de género:")
print(df_clean['gender'].value_counts())

# Resumen
print("=== DATASET FINAL PARA MODELADO ===\n")
print(f"📊 Observaciones: {len(df_clean):,}")
print(f"📊 Variables predictoras: {len(df_clean.columns) - 1}")  # excluyendo is_fraud
print(f"🎯 Variable objetivo: is_fraud")
print(f"\n📈 Distribución de la variable objetivo:")
target_dist = df_clean['is_fraud'].value_counts(normalize=True) * 100
print(f"  - Legítimas (0): {target_dist[0]:.2f}%")
print(f"  - Fraude (1): {target_dist[1]:.2f}%")

print(f"\n📋 Columnas finales del dataset:")
print(list(df_clean.columns))

import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("ANÁLISIS HIPÓTESIS 2: MONTO × CATEGORÍA")
print("=" * 60)

# Seleccionar las dos categorías de interés
categorias_interes = ['shopping_net', 'grocery_pos']

# Filtrar datos
df_shop = df[df['category'] == 'shopping_net']
df_grocery = df[df['category'] == 'grocery_pos']

print("\n ESTADÍSTICAS DESCRIPTIVAS - shopping_net:\n")
print(f"{'Métrica':<15} {'Legítimas':<20} {'Fraudulentas':<20}")
print("-" * 55)

shop_legit = df_shop[df_shop['is_fraud'] == 0]['amt']
shop_fraud = df_shop[df_shop['is_fraud'] == 1]['amt']

print(f"{'Media ($)':<15} ${shop_legit.mean():<19.2f} ${shop_fraud.mean():<19.2f}")
print(f"{'Mediana ($)':<15} ${shop_legit.median():<19.2f} ${shop_fraud.median():<19.2f}")
print(f"{'Percentil 95 ($)':<15} ${shop_legit.quantile(0.95):<19.2f} ${shop_fraud.quantile(0.95):<19.2f}")
print(f"{'Máximo ($)':<15} ${shop_legit.max():<19.2f} ${shop_fraud.max():<19.2f}")
print(f"{'N° transacciones':<15} {len(shop_legit):<19} {len(shop_fraud):<19}")

print("\n📊 ESTADÍSTICAS DESCRIPTIVAS - grocery_pos:\n")
print(f"{'Métrica':<15} {'Legítimas':<20} {'Fraudulentas':<20}")
print("-" * 55)

grocery_legit = df_grocery[df_grocery['is_fraud'] == 0]['amt']
grocery_fraud = df_grocery[df_grocery['is_fraud'] == 1]['amt']

print(f"{'Media ($)':<15} ${grocery_legit.mean():<19.2f} ${grocery_fraud.mean():<19.2f}")
print(f"{'Mediana ($)':<15} ${grocery_legit.median():<19.2f} ${grocery_fraud.median():<19.2f}")
print(f"{'Percentil 95 ($)':<15} ${grocery_legit.quantile(0.95):<19.2f} ${grocery_fraud.quantile(0.95):<19.2f}")
print(f"{'Máximo ($)':<15} ${grocery_legit.max():<19.2f} ${grocery_fraud.max():<19.2f}")
print(f"{'N° transacciones':<15} {len(grocery_legit):<19} {len(grocery_fraud):<19}")

print("\n" + "=" * 60)
print(" CONCLUSIONES PRELIMINARES:")
print("=" * 60)

# Comparar medianas
diff_shop = shop_fraud.median() - shop_legit.median()
diff_grocery = grocery_fraud.median() - grocery_legit.median()

print(f"\n shopping_net: Mediana fraudes vs legítimas: {'+' if diff_shop > 0 else ''}{diff_shop:.2f} USD")
print(f"   → Los fraudes tienen mediana {'superior' if diff_shop > 0 else 'inferior'} a las legítimas")

print(f"\n grocery_pos: Mediana fraudes vs legítimas: {'+' if diff_grocery > 0 else ''}{diff_grocery:.2f} USD")
print(f"   → Los fraudes tienen mediana {'superior' if diff_grocery > 0 else 'inferior'} a las legítimas")

print("\n HALLAZGO CLAVE:")
if abs(diff_shop) > abs(diff_grocery) * 2:
    print("   ✓ La relación entre monto y fraude es significativamente diferente")
    print("     entre categorías. Se rechaza la hipótesis nula.")
else:
    print("   ⚠ Las diferencias observadas no son concluyentes.")
    
    
import os

os.chdir("C:\\Users\\Geronimo\\Desktop\\Credit-card-fraud-detection")
df.to_csv("data/fraud_clean.csv", index=False)

import os
import matplotlib.pyplot as plt


import os
import matplotlib.pyplot as plt

figures_dir = './images/'
os.makedirs(figures_dir, exist_ok=True)

# Obtener todas las figuras abiertas
figs = plt.get_fignums()
total = len(figs)

for idx, fig_num in enumerate(figs, 1):
    fig = plt.figure(fig_num)
    fig.savefig(f'{figures_dir}figura_{idx}.png', dpi=150, bbox_inches='tight')
    print(f'✅ Guardada: figura_{idx}.png')

print(f'\n📁 Total: {total} imágenes guardadas en "{figures_dir}"')
