# Databricks notebook source
# MAGIC %pip install reportlab 

# COMMAND ----------

import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

def create_pdf(data, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    table = Table(data)
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    table.setStyle(style)
    doc.build([table])

def convert_rows_to_pdf(dataframe, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for index, row in dataframe.iterrows():
        filename = os.path.join(folder_path, f"file_{index}.pdf")
        create_pdf([list(row)], filename)
        print(f"PDF created: {filename}")

# Example usage
data = pd.read_csv('/Volumes/ang_nara_catalog/llmops/data/clinical_data.csv')  # Assuming data is in a CSV file
folder_path = '/Volumes/ang_nara_catalog/llmops/data'
convert_rows_to_pdf(data, folder_path)


# COMMAND ----------

import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

def calculate_scores(df, reference_col, generated_col):
    rouge = Rouge()
    bleu_scores = []
    rogue_scores = []
    
    for _, row in df.iterrows():
        reference = row[reference_col]
        generated = row[generated_col]
        
        # Calculate BLEU score
        bleu_score = corpus_bleu([[reference.split()]], [generated.split()])
        bleu_scores.append(bleu_score)
        
        # Calculate ROGUE scores
        rogue_score = rouge.get_scores(generated, reference)[0]['rouge-l']['f']
        rogue_scores.append(rogue_score)
    
    # Add scores to DataFrame
    df['BLEU'] = bleu_scores
    df['ROGUE'] = rogue_scores
    
    return df

# Example usage:
# Assuming df is your DataFrame with columns 'reference' and 'generated'
# df = pd.DataFrame({'reference': reference_texts, 'generated': generated_texts})
# df = calculate_scores(df, 'reference', 'generated')
