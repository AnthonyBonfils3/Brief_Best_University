from app import app
#dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table

# librairie classique
import numpy as np
import pandas as pd

# plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
#from plotly.offline import iplot

#sklearn
from sklearn import decomposition, preprocessing

import base64

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Load Datas
df_total = pd.read_csv("./data/timesData.csv")

nb_ligne = 60                          # nombre de lignes affichées

#####################################################
################# Pre treatment #####################
#####################################################

df = df_total[df_total.year == 2016].iloc[:nb_ligne,:]
df = df.dropna()
df.world_rank = [each.replace('=','') for each in df.world_rank]
df.world_rank = pd.to_numeric(df.world_rank, errors='coerce')
df.income = pd.to_numeric(df.income, errors='coerce')
df.international = pd.to_numeric(df.international, errors='coerce')
df.total_score = pd.to_numeric(df.total_score, errors='coerce')
df.num_students  = [str(each).replace(',','') for each in df.num_students]
df.num_students = pd.to_numeric(df.num_students, errors='coerce')
df.international_students = [each.replace('%','') for each in df.international_students]
df.international_students = pd.to_numeric(df.international_students, errors='coerce')
df.female_male_ratio = [str(each).split() for each in df.female_male_ratio]
df.female_male_ratio = [round((float(each[0]) / float(each[2])),2)*100 for each in df.female_male_ratio] 
df.female_male_ratio = pd.to_numeric(df.female_male_ratio, errors='coerce')

#####################################################
##################### FIGURES #######################
#####################################################

               ##############
               ### Page 1 ###
               ##############
## ------ fig1_1 : Correlation Matrix ------ # première figure sur la page 1
# fig1_1 = px.imshow(df[['world_rank','total_score', 'research', 'teaching', 'citations', 
#                         'international', 'international_students', 'income', 'student_staff_ratio']].corr(), 
#                     title='Correlation Matrix')
corr = df[['world_rank','total_score', 'research', 'teaching', 'citations', 
           'international', 'international_students', 'income', 'student_staff_ratio']].corr()

trace_heat = go.Heatmap(z = np.around(corr.values,2), 
                        x = corr.index.values, 
                        y = corr.columns.values, 
                        opacity = 0.80,
                        colorbar = {'title':'correlation <br> value (%)'},
                        hovertemplate = "<b> Abscisse :</b> %{x}<br>" 
                        + "<b> Ordonnée :</b> %{y}<br>" + "<b>Correlation value:</b> %{z}" 
                        + "<extra></extra>",
                        colorscale = 'viridis')

fig1_1 = go.Figure(data = trace_heat)
fig1_1.update_layout(title="Correlation Matrix")

## ------ fig1_2 : Correlation Matrix ------ # première figure sur la page 1

data = df[["research", "teaching", "citations"]]
data["index"] = np.arange(1,len(data)+1)

# scatter matrix
fig1_2 = ff.create_scatterplotmatrix(data, diag='box', index='index',
                                     colormap='Portland', colormap_type='cat',
                                     title='Relations between research, teatching and citations criteria',
                                     text = df[['university_name', 'world_rank']],
                                     hovertemplate = "<b> University :</b> %{text[0]}<br>" 
                                     + "<b> World rank :</b> %{text[1]}<br>"
                                     + "<extra></extra>",
                                     )#labels={"index": "world <br> rank"}, height=700, width=700)

               ##############
               ### Page 2 ###
               ##############
## ------ fig2_1 : Ebouli des valeurs propres ------ # première figure sur la page 2
# voir plus bas 

## ------ fig2_2 : Cercle des corrélations ------ # 2ème figure sur la page 2
# voir plus bas 

#####################################################
######################## ACP ########################
#####################################################

# choix du nombre de composantes à calculer
n_comp = 10

# selection des colonnes à prendre en compte dans l'ACP
colonnes = ['total_score', 'research', 'teaching', 'citations', 'international', 'income', 
            'student_staff_ratio', 'num_students', 'international_students', 'female_male_ratio']
data_pca = df[colonnes]

# préparation des données pour l'ACP
X = data_pca.values
names = df['world_rank'] # ou data.index pour avoir les intitulés
features = colonnes

# Centrage et Réduction
X_scaled = preprocessing.StandardScaler().fit_transform(X)
## On aurait pu faire
#std_scale = preprocessing.StandardScaler().fit(X)
#X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)

# Eboulis des valeurs propres : plot avec plotly
scree = pca.explained_variance_ratio_*100
fig2_1 = go.Figure()
fig2_1.add_trace(go.Scatter(x = np.arange(len(scree))+1, 
                            y = scree.cumsum(), 
                            mode = 'lines+markers',
                            name = 'Inertie cumulée',
                            hovertemplate = "<b>Pourcentage :</b> %{y} %<br>"
                            + "<extra></extra>"))
              
fig2_1.add_trace(go.Bar(x = np.arange(len(scree))+1, 
                        y = scree,
                        name = 'Inertie par composante',
                        hovertemplate = "<b>Composante :</b> %{x}<br>" 
                            + "<b>Pourcentage :</b> %{y} %<br>"
                            + "<extra></extra>"))
fig2_1.update_layout(title="Eboulis des valeurs propres",
                     xaxis_title="Composantes",
                     yaxis_title="Inertie (%)")


markdown_text1 = '''
## Elbow method (clustering)
* En analysant cette courbe on remarque que le pourcentage d’inertie diminue plus lentement 
à partir de la 6ème composante lorsque l’on parcourt le diagramme des éboulis de gauche à droite.
* On va considérer par la suite l'analyse des 6 premières composantes, donc les trois premier plans
(composantes (F1, F2), composantes (F3, F4) et composantes (F5, F6).
'''

# Cercle des corrélations
pcs = pca.components_
#fig2_2 = display_circles(pcs, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))


X_projected = pca.fit_transform(X_scaled)
pd.DataFrame(X_projected, index = df.index, columns = ["F"+str(i+1) for i in range (n_comp)])
f1 = pca.components_[0]



markdown_text2 = '''
Un point représente une université et elle est labélisée par sont classement mondial.
'''

# # Test avec les 3 première composantes uniquement

# Calcul des composantes principales
pca = decomposition.PCA(n_components=3)
pca.fit(X_scaled)

total_var = pca.explained_variance_ratio_.sum() * 100

# Projection sur les composantes principales
X_projected = pca.fit_transform(X_scaled)

fig2_2 = px.scatter_3d(
    X_projected, x=0, y=1, z=2, color=df['world_rank'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'F1', '1': 'F2', '2': 'F2'},
    height=(800)
)

#####################################################
################# HOME LAYOUT #######################
##################################################### 
layoutHome = html.Div([
    html.H1('Of all the universities, which are the best?',
            style={
                'textAlign': 'center',
                }),
    html.Br(),
    html.H1(html.Img(src='https://www.chapellerie-traclet.com/modules/prestablog/views/img/grid-for-1-7/up-img/119.jpg'),
            style={
                'textAlign': 'center',
                }),
    html.Br(),
    html.Div(id='app-home-display-value'),
    dcc.Link('Standard Data Analysis', href='/apps/app1'),
    html.Br(),
    dcc.Link("Principal Component Analysis", href='/apps/app2'),
    # test de partage écran
   #   html.Section(style={ 'backgroundColor':'green', 'height':'70vh', 'display':'flex', 'justify-content':' space-around'}, children=[
   #       html.Div(style={'backgroundColor':'blue',  'height':'70vh', 'width':'35vw'}),
   #       html.Div(style={'backgroundColor':'red',  'height':'70vh', 'width':'45vw'}),
   # ])
])


#####################################################
################# PAGE 1 LAYOUT #####################
#####################################################
layout1 = html.Div([
    html.H1('Standard Data Analysis',
            style = {
                'textAlign': 'center'
                }),
    html.H3('Table des données'),
    dash_table.DataTable(id='app-1-dropdown',
    columns=[{'id': c, 'name': c} for c in df.columns],
    data= df.to_dict('records'),
    export_format = 'csv',
#    style_as_list_view=True, # voir le tableau comme une liste
    fixed_rows={'headers': True}, # garder les header quand on scroll 
#    fixed_columns={'world_rank': True}, 
    fixed_columns={'headers': True, 'data' :1,
                   'headers': True, 'data' :2},# garder les word_ranf quand on scroll : le chiffre correspond à l'indice de la colonne
    column_selectable = 'multi',
    style_table={'overflowX': 'auto',
                 'overflowY': 'auto',
                 'maxHeight':'400px',
                 'maxWidth':'1200px'},
    #Cell dim + textpos
    style_cell_conditional=[{'height': 'auto',
        'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
        'whiteSpace': 'normal','textAlign':'center'}
    ],
    
    #Line strip
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(50, 50, 50)',
        'fontWeight': 'bold',
        'color':'white'},
    ),
    html.Br(),
    html.H3('Graphiques'),
    html.Br(),
    html.Section(style={'height':'70vh', 'display':'flex', 'justify-content':' space-around'}, children=[
        html.Div(style={'height':'70vh', 'width':'60vw'}, children=[
                  ## Fig 1 : Correlation Matrix
                  dcc.Graph(
                      id='example-graph-1-1',
                      figure=fig1_1
                      ),
                  ]),
        html.Div(style={'height':'70vh', 'width':'40vw'}, children=[
                  ## Fig 1 : Correlation Matrix
                  dcc.Graph(
                      id='example-graph-1-2',
                      figure=fig1_2
                      ),
                  ]),
    ]),
    
    ## types de valeurs
    
    
    ## Lien retours
    html.Br(),
    dcc.Link("Principal Component Analysis", href='/apps/app2'),
    html.Br(),
    dcc.Link("Go to home page", href='/')
])


#####################################################
################# PAGE 2 LAYOUT #####################
#####################################################
layout2 = html.Div([
    html.H1('Principal Component Analysis',
            style = {
                'textAlign': 'center'
                }),
    html.Div(id='app-2-display-value'),
    html.Br(),
    html.H2("1. Analyse avec 10 Composantes (min(n-1, p) avec n nombre d'éléments et p nombre de variables')"),
    #### Premier block
    html.Section(style={'height':'70vh', 'display':'flex', 'justify-content':' space-around'}, children=[
         html.Div(style={'height':'70vh', 'width':'70vw'}, children=[
                  dcc.Graph(id='graph-2_1', figure=fig2_1),
                  ]),
         html.Div(style={'height':'70vh', 'width':'30vw','marginLeft': 10, 
                         'marginRight': 10, 'marginTop': 10, 'marginBottom': 10, 
                         'textAlign': 'justify'
                         }, children=[
             dcc.Markdown(children = markdown_text1)
             ]),
         ]),
    html.Br(),
    #### 2ème block
    html.H3('Représentation des variables dans les plans factoriels',
            style = {
                'textAlign': 'center'
                }),
    html.Br(),
    html.Section(style={'height':'70vh', 'display':'flex', 'justify-content':' space-around'}, children=[
          html.Div(style={'height':'70vh', 'width':'33vw'}, children=[
                    html.Img(style={'height':'60vh', 'width':'30vw'},
                             src=app.get_asset_url('cercle1.png')),
                    ]),
          html.Div(style={'height':'70vh', 'width':'33vw'}, children=[
                    html.Img(style={'height':'60vh', 'width':'30vw'},
                             src=app.get_asset_url('cercle2.png')),
                    ]),
          html.Div(style={'height':'70vh', 'width':'33vw'}, children=[
                    html.Img(style={'height':'60vh', 'width':'30vw'},
                             src=app.get_asset_url('cercle3.png')),
                    ]),
          ]),
    #### 3ème block
    html.H3('Représentation des données dans les plans factoriels',
            style = {
                'textAlign': 'center'
                }),
    html.Br(),
    html.Section(style={'height':'60vh', 'display':'flex', 'justify-content':' space-around'}, children=[
          html.Div(style={'height':'60vh', 'width':'33vw'}, children=[
                    html.Img(style={'height':'60vh', 'width':'30vw'},
                             src=app.get_asset_url('projection_individus1.png')),
                    ]),
          html.Div(style={'height':'60vh', 'width':'33vw'}, children=[
                    html.Img(style={'height':'60vh', 'width':'30vw'},
                             src=app.get_asset_url('projection_individus2.png')),
                    ]),
          html.Div(style={'height':'60vh', 'width':'33vw'}, children=[
                    html.Img(style={'height':'60vh', 'width':'30vw'},
                             src=app.get_asset_url('projection_individus3.png')),
                    ]),
          ]),
    html.Br(),
    html.Div(dcc.Markdown(children = markdown_text2)),
    html.Br(),
    html.H2("2. Analyse avec 3 Composantes uniquement"),
    html.Br(),
    html.H3('Représentation des données en 3D sur les 3 premieres composantes',
            style = {
                'textAlign': 'center'
                }),
    html.Br(),
    html.Div(dcc.Graph(id='graph-2_2', figure=fig2_2),),
    ## Lien retours
    html.Br(),
    dcc.Link('Standard Data Analysis', href='/apps/app1'),
    html.Br(),
    dcc.Link("Go to home page", href='/')
])