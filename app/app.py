from flask import Flask, render_template, request
import faiss
import os
import json
import numpy as np
import google.generativeai as genai
app = Flask(__name__)
# Configure Gemini API key
genai.configure(api_key='YOUR_API_KEY')  # Replace with your actual API key
sicknesses = {
    "Common Cold": {
        "Description": "A viral infection of the upper respiratory tract, primarily affecting the nose and throat.",
        "Symptoms": [
            "Runny or stuffy nose",
            "Sneezing",
            "Sore throat",
            "Coughing",
            "Mild headache",
            "Fatigue",
            "Low-grade fever"
        ],
        "Causes": "Caused by various viruses, most commonly rhinoviruses. Spreads through respiratory droplets or contact with contaminated surfaces.",
        "Medications": [
            "Decongestants (e.g., pseudoephedrine) for nasal congestion",
            "Pain relievers (e.g., ibuprofen or acetaminophen) for fever and headaches",
            "Cough suppressants (e.g., dextromethorphan)",
            "Rest, hydration, and soothing throat lozenges"
        ]
    },
    "Influenza (Flu)": {
        "Description": "A highly contagious viral infection that affects the respiratory system.",
        "Symptoms": [
            "High fever",
            "Chills",
            "Muscle aches",
            "Fatigue",
            "Sore throat",
            "Dry cough",
            "Nasal congestion"
        ],
        "Causes": "Caused by influenza viruses transmitted via respiratory droplets or contaminated surfaces.",
        "Medications": [
            "Antiviral drugs (e.g., oseltamivir or zanamivir) if taken early",
            "Pain relievers (e.g., acetaminophen or ibuprofen)",
            "Rest, hydration, and over-the-counter medications for specific symptoms"
        ]
    },
    "Gastroenteritis (Stomach Flu)": {
        "Description": "An inflammation of the stomach and intestines causing digestive distress.",
        "Symptoms": [
            "Nausea",
            "Vomiting",
            "Diarrhea",
            "Abdominal cramps",
            "Fever",
            "Dehydration"
        ],
        "Causes": "Viral (e.g., norovirus, rotavirus), bacterial (e.g., E. coli, Salmonella), or parasitic infections; consuming contaminated food or water.",
        "Medications": [
            "Oral rehydration solutions (ORS) for dehydration",
            "Antidiarrheal drugs (e.g., loperamide, but not recommended for bacterial causes)",
            "Antiemetics (e.g., ondansetron) for nausea",
            "Probiotics to restore gut flora"
        ]
    },
    "Urinary Tract Infection (UTI)": {
        "Description": "An infection in any part of the urinary system, including the bladder, urethra, or kidneys.",
        "Symptoms": [
            "Pain or burning during urination",
            "Frequent urge to urinate",
            "Cloudy or foul-smelling urine",
            "Pelvic pain",
            "Sometimes fever"
        ],
        "Causes": "Bacterial infection, often Escherichia coli (E. coli), which enters through the urethra. Poor hygiene and dehydration increase risk.",
        "Medications": [
            "Antibiotics (e.g., nitrofurantoin, trimethoprim-sulfamethoxazole)",
            "Pain relievers (e.g., phenazopyridine) for urinary discomfort",
            "Increased water intake to flush out bacteria"
        ]
    },
    "Allergic Rhinitis (Hay Fever)": {
        "Description": "An allergic reaction to airborne allergens like pollen, dust mites, or pet dander.",
        "Symptoms": [
            "Sneezing",
            "Runny or congested nose",
            "Itchy eyes, throat, or nose",
            "Watery eyes",
            "Fatigue"
        ],
        "Causes": "Allergens trigger the immune system to release histamines, causing symptoms.",
        "Medications": [
            "Antihistamines (e.g., loratadine, cetirizine)",
            "Nasal corticosteroids (e.g., fluticasone, mometasone)",
            "Decongestants (e.g., pseudoephedrine) for temporary relief",
            "Allergen avoidance and air purifiers to reduce exposure"
        ]
    }
}
# Load FAISS index and documents mapping
save_folder = "rag_system"
index = faiss.read_index(os.path.join(save_folder, "index.faiss"))
with open(os.path.join(save_folder, "documents.json"), "r") as f:
    docs_mapping = json.load(f)
sickness_names = [docs_mapping[str(i)]["sickness_name"] for i in range(len(docs_mapping))]

# Embedding function
def embed_query(text):
    embedding_result = genai.embed_content(
        model='models/embedding-001',
        content=text,
        task_type='retrieval_query'
    )
    return np.array(embedding_result['embedding'], dtype=np.float32).reshape(1, -1)

# Retriever function
def retriever(query, index, sickness_names, k=1):
    query_embedding = embed_query(query)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    retrieved_sickness = sickness_names[indices[0][0]]
    return retrieved_sickness

# Response generation function
def generate_response(query, retrieved_docs):
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    prompt = f"""
    Respond to the query as if you're having a friendly, informative conversation:
    Context:
    {retrieved_docs}
    Query: {query}
    Explain the answer in a warm, accessible manner that's easy to understand.
    """
    response = model.generate_content(prompt)
    return response.text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    input_symptoms = request.form['symptoms']
    retrieved_sickness = retriever(input_symptoms, index, sickness_names)
    sickness_data = sicknesses[retrieved_sickness]
    response = generate_response(input_symptoms, sickness_data)
    return render_template('response.html', response=response, sickness=retrieved_sickness, sickness_data=sickness_data)

if __name__ == '__main__':
    app.run(debug=True)