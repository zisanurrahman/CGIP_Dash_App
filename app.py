import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import dash_table
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski, Descriptors
from io import BytesIO
import base64
import gunicorn

# Load and preprocess data
df = pd.read_csv('https://zenodo.org/records/13117525/files/unique_drugs_filtered_sorted_volcano_shiny_filtered2.csv')
df.rename(columns={
    "modified_padj": "P_value",
    "Z.Score_ESet_mean": "Z_Score",
    "Product.Description_x": "Product_Description",
    "Locus.Tag": "Locus_Tag",
    "Gene.Name": "Gene_Name",
    "Accession_y": "Accession",
    "Evidence.Ontology.ECO.Code": "Evidence_Ontology_ECO_Code",
    "GO.Term": "GO_Term",
    "Namespace": "GO_Categories",
    "COG2": "COG"
}, inplace=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Load SMILES data
smiles_df = pd.read_csv('https://zenodo.org/records/13117525/files/unique_compounds_smiles.csv')

# Load UMAP data
umap_df = pd.read_csv('https://zenodo.org/records/13117525/files/unique_drugs_umap_clusters.csv')

# Initialize the app with a nice theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

# Function to convert SMILES to image
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol, size=(300, 300))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    return None

# Function to calculate Lipinski parameters
def calculate_lipinski_params(mol):
    params = {
        'Molecular Weight (Da)': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),  # Add LogP calculation
        'Fraction SP3': Lipinski.FractionCSP3(mol),
        'Heavy Atom Count': Lipinski.HeavyAtomCount(mol),
        'N/O Count': Lipinski.NOCount(mol),
        'NH/OH Count': Lipinski.NHOHCount(mol),
        'Aliphatic Carbocycles': Lipinski.NumAliphaticCarbocycles(mol),
        'Aliphatic Heterocycles': Lipinski.NumAliphaticHeterocycles(mol),
        'Aliphatic Rings': Lipinski.NumAliphaticRings(mol),
        'Aromatic Carbocycles': Lipinski.NumAromaticCarbocycles(mol),
        'Aromatic Heterocycles': Lipinski.NumAromaticHeterocycles(mol),
        'Aromatic Rings': Lipinski.NumAromaticRings(mol),
        'Hydrogen Bond Acceptors': Lipinski.NumHAcceptors(mol),
        'Hydrogen Bond Donors': Lipinski.NumHDonors(mol),
        'Heteroatoms': Lipinski.NumHeteroatoms(mol),
        'Rotatable Bonds': Lipinski.NumRotatableBonds(mol),
        'Saturated Carbocycles': Lipinski.NumSaturatedCarbocycles(mol),
        'Saturated Heterocycles': Lipinski.NumSaturatedHeterocycles(mol),
        'Saturated Rings': Lipinski.NumSaturatedRings(mol),
        'Total Rings': Lipinski.RingCount(mol)
    }
    return params

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Markdown(
            """
            # Chemical-Genetic Interaction Profile Viewer: CRISPRi-Seq profiling of _B. cenocepacia_ K56-2 essential genome
            """,
            style={'textAlign': 'center'}
        ), className='mb-4')
    ]),
    dbc.Row([
        dbc.Col(html.H4("About This App"), width=12),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P([
                    "In this app you can explore a large-scale chemical-genetic interaction profile (CGIP) of the ",
                    html.Em("B. cenocepacia"),
                    " K56-2 essential genome."
                ]),
                html.P("This app accesses data from: Rahman ASMZ et al. 202X. XXX. The complete dataset contains over 3 millions interactions generated from screening of an CRISPRi-based essential gene knockdown mutant library against 5000 compounds."),
                html.P("For clarity and visualization purposes, only the compounds exhibiting strong interactions (Z-Score ≤ -5) with at least one essential gene knockdown mutant are displayed here."),
                html.P("Special Note: UMAP plot here projects the 1827-dimensional (609 knockdown mutants X mutant response, GO annotation, and COG) CGIPs for each compound onto two dimensions. The analysis concatenates the features (knockdown mutants and their response, GO annotation, and COG) into a single list for each compound, ensuring all profiles are padded to the same length. UMAP (Uniform Manifold Approximation and Projection) is applied to reduce the high-dimensional data to 2 dimensions, creating a 2D representation of each compound profile. The 2D UMAP results are then used as input for the K-means clustering algorithm. K-means partitioned the data into a predefined number of clusters (in this case, 10), grouping compounds based on the similarity of their 2D representations."),
                
                html.Hr()
            ])
        ]), width=12, className='mb-4')
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Compound:"),
            dcc.Dropdown(id='compound', options=[{'label': i, 'value': i} for i in df['Compounds'].unique()], value=df['Compounds'].unique()[0])
        ], width=4),
        dbc.Col([
            html.Label("Z-Score Range:"),
            dcc.RangeSlider(
                id='z_score',
                min=int(np.floor(df['Z_Score'].min() / 5) * 5),
                max=int(np.ceil(df['Z_Score'].max() / 5) * 5),
                step=5,
                value=[int(np.floor(df['Z_Score'].min() / 5) * 5), int(np.ceil(df['Z_Score'].max() / 5) * 5)],
                marks={i: str(i) for i in range(int(np.floor(df['Z_Score'].min() / 5) * 5), int(np.ceil(df['Z_Score'].max() / 5) * 5) + 1, 5)}
            )
        ], width=4),
        dbc.Col([
            html.Label("P-value Range:"),
            dcc.RangeSlider(
                id='padj_m',
                min=round(df['P_value'].min(), 1),
                max=round(df['P_value'].max(), 1),
                step=0.1,
                value=[round(df['P_value'].min(), 1), round(df['P_value'].max(), 1)],
                marks={i: f'{i:.1f}' for i in np.arange(round(df['P_value'].min(), 1), round(df['P_value'].max(), 1) + 0.1, 0.1)}
            )
        ], width=4)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            html.Label("COG:"),
            dcc.Dropdown(id='cog2', options=[{'label': i, 'value': i} for i in df['COG'].dropna().unique()], value='', clearable=True)
        ], width=4),
        dbc.Col([
            html.Label("GO Term:"),
            dcc.Dropdown(id='go_term', options=[{'label': i, 'value': i} for i in df['GO_Term'].dropna().unique()], value='', clearable=True)
        ], width=4),
        dbc.Col([
            html.Label("GO Category:"),
            dcc.Dropdown(id='go_category', options=[{'label': i, 'value': i} for i in df['GO_Categories'].dropna().unique()], value='', clearable=True)
        ], width=4)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='volcano_plot')
        ], width=5),
        dbc.Col([
            html.H4("Compound Structure", style={'textAlign': 'center'}),
            html.Img(id='compound_structure', style={'width': '500px', 'height': '500px', 'display': 'block', 'margin': 'auto'})
        ], width=7),
        
            dbc.Col([
        dbc.Col([
            html.H4("", style={'textAlign': 'center'}),
            dcc.Graph(id='umap_plot')
        ], width=5)
    ], className='mb-4'),
          dbc.Col([
              dbc.Col([
                  html.H4("", style={'textAlign': 'center'}),
                  dash_table.DataTable(
                      id='features_table',
                      style_table={'overflowX': 'auto'},
                      style_header={
                          'backgroundColor': 'lightblue',
                          'fontWeight': 'bold',
                          'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'  # Shadow effect
                      },
                      style_cell={
                          'height': 'auto',
                          'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                          'whiteSpace': 'normal',
                          'textAlign': 'center'  # Center values
                      },
                      style_data={
                          'whiteSpace': 'normal',
                          'textAlign': 'center',
                          'fontSize': '14px'
                      },
                      page_size=20  # Adjusted pagination
                  )
              ], width=12)
          ]),
          
        dbc.Col([
            html.H4("Top 5 Most Depleted Mutants", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='top_mutants_table',
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'lightblue',
                    'fontWeight': 'bold',
                    'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'  # Shadow effect
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'textAlign': 'center'  # Center values
                },
                style_data={
                    'whiteSpace': 'normal',
                    'textAlign': 'center',
                    'fontSize': '14px'
                },
                page_size=5  # Adjusted pagination to fit within the row
            )
        ], width=12)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col([
            html.H4("The complete chemical-genetic interaction profile dataset", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='selected_value_table',
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': 'lightblue',
                    'fontWeight': 'bold',
                    'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'  # Shadow effect
                },
                style_cell={
                    'height': 'auto',
                    'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'textAlign': 'center'  # Center values
                },
                style_data={
                    'whiteSpace': 'normal',
                    'textAlign': 'center',
                    'fontSize': '14px'
                },
                page_size=10  # Added pagination
            )
        ], width=12)
    ])
])

# Define callbacks
@app.callback(
    Output('volcano_plot', 'figure'),
    Output('selected_value_table', 'data'),
    Output('selected_value_table', 'columns'),
    Output('top_mutants_table', 'data'),
    Output('top_mutants_table', 'columns'),
    Output('compound_structure', 'src'),
    Output('features_table', 'data'),
    Output('features_table', 'columns'),
    Output('umap_plot', 'figure'),
    Input('compound', 'value'),
    Input('z_score', 'value'),
    Input('padj_m', 'value'),
    Input('cog2', 'value'),
    Input('go_term', 'value'),
    Input('go_category', 'value')
)
def update_plots(compound, z_score, padj_m, cog2, go_term, go_category):
    filtered_data = df[
        (df['Compounds'] == compound) &
        (df['Z_Score'] >= z_score[0]) & (df['Z_Score'] <= z_score[1]) &
        (df['P_value'] >= padj_m[0]) & (df['P_value'] <= padj_m[1])
    ]
    filtered_data['log_padj'] = -np.log10(filtered_data['P_value'])
    
    if cog2:
        filtered_data = filtered_data[filtered_data['COG'] == cog2]
    if go_term:
        filtered_data = filtered_data[filtered_data['GO_Term'] == go_term]
    if go_category:
        filtered_data = filtered_data[filtered_data['GO_Categories'] == go_category]
    
    volcano_fig = px.scatter(
        filtered_data, x='Z_Score', y='log_padj',
        color=filtered_data['Z_Score'].apply(lambda x: 'darkred' if x <= -1 else 'darkblue' if x >= 1 else 'grey'),
        hover_data=['Gene', 'GO_Term'],  # Include hover data
        labels={'Z_Score': 'Z-Score', 'log_padj': '-Log10 (P-value)'},
        title=f'<b>Volcano Plot for {compound}</b>'
    )
    volcano_fig.update_layout(
        xaxis_title=dict(text="Z-Score", font=dict(size=22, color="black")),
        yaxis_title=dict(text="-Log10 (P-value)", font=dict(size=22, color="black")),
        plot_bgcolor='white',
        showlegend=False,
        width=600,  # Set the width for square plot
        height=600,  # Set the height for square plot
        title_x=0.5,  # Center the title
        title_font=dict(size=18, color='black'),
        title_font_family="Arial"
    )
    
    # Ensure that axis lines are visible on all four sides
    volcano_fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    volcano_fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    
    table_data = filtered_data.round(1).to_dict('records')
    table_columns = [{'name': col, 'id': col} for col in filtered_data.columns if col != 'Unnamed: 0']
    
    top_mutants_data = filtered_data.sort_values(by='Z_Score').drop_duplicates(subset='Mutants').head(5).round(1).to_dict('records')
    top_mutants_columns = [{'name': col, 'id': col} for col in top_mutants_data[0].keys()] if top_mutants_data else []

    # Get the SMILES string for the selected compound
    smiles = smiles_df.loc[smiles_df['Compounds'] == compound, 'SMILES'].values
    smiles_img = None
    features_data = []
    if smiles:
        smiles_img = smiles_to_image(smiles[0])
        mol = Chem.MolFromSmiles(smiles[0])
        if mol:
            lipinski_params = calculate_lipinski_params(mol)
            features_data = [{'Feature': k, 'Value': v} for k, v in lipinski_params.items()]
    
    features_columns = [{'name': 'Physiochemical Features', 'id': 'Feature'}, {'name': 'Value', 'id': 'Value'}]
    
    # Create UMAP plot with clusters colored separately
    cluster_colors = {
        0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: 'aqua',
        4: '#9467bd', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f',
        8: '#bcbd22', 9: '#17becf'
    }
    umap_fig = px.scatter(
        umap_df, x='UMAP1', y='UMAP2',
        color='Cluster',  # Color by cluster
        color_discrete_map=cluster_colors,
        hover_name='Compounds',
        hover_data={'Compounds': True},
        title='<b>UMAP Clustering of the Compounds</b>'
    )
    
    # Highlight the selected compound with a larger marker
    umap_fig.add_scatter(
        x=umap_df[umap_df['Compounds'] == compound]['UMAP1'],
        y=umap_df[umap_df['Compounds'] == compound]['UMAP2'],
        mode='markers+text',
        marker=dict(color='darkred', size=12, symbol='circle-open'),
        text=compound,
        textposition='top center',
        name='Selected'
    )
    
    umap_fig.update_layout(
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.5, y=0.5, xanchor='center', yanchor='middle'),  # Position legend inside the plot
        width=700,  # Adjust width if necessary
        height=700,  # Adjust height if necessary
        xaxis_title=dict(text="UMAP1", font=dict(size=18, color="black")),
        yaxis_title=dict(text="UMAP2", font=dict(size=18, color="black")),
        title_x=0.5,  # Center the title
        title_font=dict(size=18, color='black'),
        title_font_family="Arial"
    )
    
    # Ensure that axis lines are visible on all four sides
    umap_fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    umap_fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    
    return volcano_fig, table_data, table_columns, top_mutants_data, top_mutants_columns, smiles_img, features_data, features_columns, umap_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
server = app.server
