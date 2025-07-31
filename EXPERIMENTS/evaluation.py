import os
import pandas as pd
import json
import numpy as np
import sys
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS, InMemoryVectorStore
from langchain.schema import Document

from ragas.run_config import RunConfig
import plotly.graph_objects as go
import time as time
from dotenv import load_dotenv
import plotly.graph_objects as go
import time
#import wandb
# from nltk.translate.bleu_score import sentence_bleu
# from rouge_score import rouge_scorer
# from rouge import Rouge
# from bert_score import score

from ragas.metrics import (
    LLMContextRecall, Faithfulness, FactualCorrectness,
    LLMContextPrecisionWithoutReference, NoiseSensitivity,
    ResponseRelevancy, ContextEntityRecall
)

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper

# Initialize W&B logging
#wandb.init(project="RAGAS_Model_Evaluation", name="multi_model_visualization")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules', 'llmYogaTeacher', 'knowledge_Graph'))
from knowledge_graph import KnowledgeGraph #scusa luca ma devo capire come funzionano gli import, a me questo non va
#from modules.llmYogaTeacher.knowledge_Graph.knowledge_graph import KnowledgeGraph #scusa giulia ma devo capire come funzionano gli import, a me questo non va

load_dotenv()

# Initialize paths and global variables
from plotly.subplots import make_subplots
BASE_PATH = os.path.dirname(__file__) #current directory
VECTOR_STORE_PATH = os.path.join(BASE_PATH, "vector_store")
RESULTS_FOLDER = os.path.join(BASE_PATH, "results_luca")
FACTUAL_RESPONSES_PATH = os.path.join(RESULTS_FOLDER, "factual/modified_prompt/")
FACTUAL_RESPONSES_FILE = os.path.join(RESULTS_FOLDER, "factual/modified_prompt/factual_responses.json")
GENERAL_RESPONSES_FILE = os.path.join(RESULTS_FOLDER, "general/modified_prompt/general_responses_MEGRAPH.json")
GENERAL_RESPONSES_PATH = os.path.join(RESULTS_FOLDER, "general/modified_prompt/")
MEMORY_FOLDER = "yoga_tutor_memories"
QUERY_FILE_CSV = "queries.csv"
FACTUAL_QUERY_FILE_JSON = "yoga_factual_queries.json"
GENERAL_QUERY_FILE_JSON = "yoga_general_queries.json"
OUTPUT_FILE = "responses.json"
file_to_retrieve = "Summary_new.txt"

def initialize_llm():
    return AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0.5,
    max_tokens=2048, #was 128 
    )

def initialize_embeddings():
    return AzureOpenAIEmbeddings(
    deployment=os.environ.get("AZURE_DEPLOYMENT_EMBEDDINGS"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint="https://iitlines-swecentral1.openai.azure.com",
    model="text-embedding-ada-002",
    chunk_size=1024 #fix this one
    )

def initialize_rag_prompt_template():
    return PromptTemplate(
    input_variables=["context", "query"],
    template="""
       You are an AI assistant. Answer the following query using the given context.
       The answer must mirrors the structure and phrasing of the question. Do not repeat the question in the answer! Do not introduce additional words, synonyms, or unnecessary context.

       Context:
       {context}

       Question:
       {query}

       Answer:
       """
    )

def load_conversations(folder_path, file_to_upload):
    conversations = []
    for user_folder in os.listdir(folder_path):
        user_path = os.path.join(folder_path, user_folder)
        if os.path.isdir(user_path):
            for interaction_folder in os.listdir(user_path):
                if interaction_folder.startswith("interaction"):
                    interaction_path = os.path.join(user_path, interaction_folder)
                    raw_chat_path = os.path.join(interaction_path, file_to_upload)
                    if os.path.exists(raw_chat_path):
                        with open(raw_chat_path, 'r', encoding='utf-8') as f:
                            # lines = f.readlines()[1:]  # Skip the first line if you are usinf Raw_chat.txt
                            # conversations.append("".join(lines))
                            conversations.append([user_folder, f.read()])
    return conversations

def load_queries_json(category=None):
    if category == "factual":
        file_path = os.path.join(BASE_PATH, FACTUAL_QUERY_FILE_JSON)
    elif category == "general":
        file_path = os.path.join(BASE_PATH, GENERAL_QUERY_FILE_JSON)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data 
    except Exception as e:
        print(f"❌ Error loading queries as JSON: {e}")

def create_vector_store(conversations):
    # check if vector store already exists
    if os.path.exists(VECTOR_STORE_PATH):
        print("[Vector Store] Vector Store Loaded 🔄")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

    # Use RecursiveCharacterTextSplitter to split long docs into chunks (~500 tokens each)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = []

    for conv_id, conv in enumerate(conversations):
        chunks = text_splitter.split_text(conv[1])
        for chunk in chunks:
            chunked_docs.append(Document(page_content=chunk, metadata={"conversation_id": conv_id, "user_name": conv[0]}))

    print(f"🔹 Total chunks created: {len(chunked_docs)}")  # Debugging info
    vector_store = FAISS.from_documents(chunked_docs, embeddings)

    # save vector_store and embeddings
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"[Vector Store] Vector Store Saved to {VECTOR_STORE_PATH} ✅")

    return vector_store

def retrieve_rag_relevant_documents(vector_store, query, k_val):
    retrieved_docs = []

    # Retrieve relevant documents
    relevant_docs = vector_store.similarity_search(query, k=k_val)
    #print(f"🔹 Retrieved a total of {k_val} relevant documents")
    retrieved_docs.append(relevant_docs)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return retrieved_docs, context

def compute_rag_retrieval_metrics(query, expected_user, ground_truth, retrieved_docs, k=2):
    """Compute Retrieval Metrics based on metadata (conversation_id / user_name)."""

    retrieved_conversations = [doc.metadata["user_name"] for doc in retrieved_docs[0][:k]]
    # retrieved_texts = [doc.page_content for doc in retrieved_docs[0][:k]]

    ## ** Metadata-Based Retrieval Metrics**
    relevant_count_metadata = sum(1 for user in retrieved_conversations if user == expected_user)
    recall_score_metadata = relevant_count_metadata / 1  # Binary relevance (1 if correct, 0 otherwise)

    # Compute MRR (Mean Reciprocal Rank) based on first correct match
    rank_metadata = next((idx + 1 for idx, user in enumerate(retrieved_conversations) if user == expected_user), None)
    reciprocal_rank_metadata = 1 / rank_metadata if rank_metadata else 0

    ## ** Text-Based Retrieval Metrics** @TODO: these are not working as expecting
    # relevant_count_text = sum(1 for text in retrieved_texts if ground_truth.lower() in text.lower())
    #
    # recall_score_text = relevant_count_text / 1  # Binary relevance
    # precision_score_text = relevant_count_text / k  # Precision based on retrieved docs

    ## **Print Metrics**
    print(f"    📊 Retrieval Metrics for Query: {query}:\n")
    print("        🔹 Metadata-Based:")
    print(f"        - Recall@{k}: {recall_score_metadata:.4f}")
    print(f"        - MRR: {reciprocal_rank_metadata:.4f}\n")

    # print("        🔹 Text-Based:")
    # print(f"        - Recall@{k}: {recall_score_text:.4f}")
    # print(f"        - Precision@{k}: {precision_score_text:.4f}\n")

    metrics_dict = {"recall_score_metadata":recall_score_metadata,"reciprocal_rank_metadata":reciprocal_rank_metadata}
    # "recall_score_text":recall_score_text,"precision_score_text":precision_score_text
    return metrics_dict

def generate_responses(queries, vector_store, methods=[], condition=None):
    """ Generate responses for each query in the json file and save the responses in a new json file.
        if a resposes.json file already exists, it will be loaded """

    # for each method check if the responses.json file already exists, if true print a message and load the file, remove the method from the list
    methods_to_remove = []
    for method in methods:
        if os.path.exists(os.path.join(RESULTS_FOLDER, CONDITION, f"responses_{method.lower()}.json")):
            print(f"\033[92m[{method}] Responses already exist for {CONDITION} 🔄\033[0m")
            methods_to_remove.append(method)
    
    for method in methods_to_remove:
        methods.remove(method)

    responses = {"Responses": []}

    for query in queries["Questions"]:

        ### RAG RESPONSE ###
        if "RAG" in methods:
            retrieved_docs, context = retrieve_rag_relevant_documents(vector_store, query["Question"], k_val=len(vector_store.index_to_docstore_id))
            formatted_prompt = prompt_template.format(context=context, query=query["Question"]).strip()
            rag_response = llm.invoke(formatted_prompt).content
            print(f"🤖 RAG-based response: {rag_response}")

        ### GRAPH RESPONSE ###
        if "GRAPH" in methods:
            context = knowledge_graph.get_whole_graph()
            graph_response = knowledge_graph.ask_graph(context, query["Question"])
            print(f"🤖 Graph-based response: {graph_response}")

        ### CYPERGRAPH RESPONSE ###
        if "CypherGRAPH" in methods:
            try:
                graph_cypher_response = knowledge_graph.agentic_ask_graph(query["Question"])["result"]
                print(f"🤖 Cypher-Graph-based response: {graph_cypher_response}")
            except:
                graph_cypher_response = "An error occured with the cypher query"
                print(f"🤖 Cypher-Graph-based response: {graph_cypher_response}")


        # Append the responses and metrics to the user's entry
        response_entry = {
            "Question": query["Question"],
            "Ground_truth": query["GroundTruth"]
        }
        if "RAG" in methods:
            response_entry["RAG"] = rag_response
        if "GRAPH" in methods:
            response_entry["GRAPH"] = graph_response
        if "CypherGRAPH" in methods:
            response_entry["CypherGRAPH"] = graph_cypher_response
        responses["Responses"].append(response_entry)

    # Save the user_responses to a JSON file for each method
    os.makedirs(os.path.join(RESULTS_FOLDER, CONDITION), exist_ok=True)
    for method in methods:
        output_filename = os.path.join(RESULTS_FOLDER, CONDITION, f"responses_{method.lower()}.json")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w") as json_file:
            # dump only the responses coming from the specific method, use responses dict
            responses_to_dump = {"Responses": [entry for entry in responses["Responses"] if method in entry]}
                                                    
            json.dump(responses_to_dump, json_file, indent=4)
            print(f"[{method}] Responses saved to {output_filename} ✅")

def filter_json(data, category=None, user_names=None):
    filtered_data = {
        "Category": data.get("Category", ""),
        "Users": []
    }

    # Filtra per categoria se specificata
    if category and data.get("Category") != category:
        return filtered_data

    # Filtra per nomi utente se specificati
    if user_names:
        user_names = [name.lower() for name in user_names]
        for user_entry in data.get("Users", []):
            for user, details in user_entry.items():
                if user.lower() in user_names:
                    filtered_data["Users"].append({user: details})
    else:
        filtered_data["Users"] = data.get("Users", [])

    return filtered_data

def build_ragas_dataset(method):
    with open(os.path.join(RESULTS_FOLDER, CONDITION, f"responses_{method.lower()}.json"), "r") as file:
        responses = json.load(file)

    list_of_qa = []

    for query in responses.get("Responses", []):

        question = query.get("Question", "")
        response = query.get(method, "")
        ground_truth = query.get("Ground_truth", "")

        list_of_qa.append(SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=[ground_truth],  # used for Faithfulness (Factual Correctness)
            reference=ground_truth  # used for Factual Correctness (coverage)
        ))

    return EvaluationDataset(samples=list_of_qa)

def save_metrics(ragas_metrics, output_filename):
    """ save the ragas eval to a csv file """
    df = ragas_metrics.to_pandas()
    graph_eval_csv = os.path.join(RESULTS_FOLDER, output_filename)
    df.to_csv(graph_eval_csv, index=False)


def extract_metrics_from_df(df):
    """
    Extract metrics from DataFrame.
    This is a placeholder - replace with your actual implementation.
    """
    # Example implementation - update with your actual logic
    metrics = {}
    if 'metric_name' in df.columns and 'metric_value' in df.columns:
        for _, row in df.iterrows():
            metrics[row['metric_name']] = row['metric_value']
    return metrics

def radar_plot(rag_eval, graph_eval, cypher_eval, condition=None):
    modelname1 = "RAG"
    modelname2 = "GRAPH"
    modelname3 = "CypherGRAPH"

    metrics_data_1 = extract_metrics_from_df(rag_eval)
    metrics_data_2 = extract_metrics_from_df(graph_eval)
    metrics_data_3 = extract_metrics_from_df(cypher_eval)
    fig = go.Figure()

    # Prepare data for the first model
    metrics_1 = list(metrics_data_1.keys())
    values_1 = list(metrics_data_1.values())
    metrics_1 += [metrics_1[0]]  # Close radar loop
    values_1 += [values_1[0]]

    # Prepare data for the second model
    metrics_2 = list(metrics_data_2.keys())
    values_2 = list(metrics_data_2.values())
    metrics_2 += [metrics_2[0]]  # Close radar loop
    values_2 += [values_2[0]]

    # Prepare data for the third model
    metrics_3 = list(metrics_data_3.keys())
    values_3 = list(metrics_data_3.values())
    metrics_3 += [metrics_3[0]]  # Close radar loop
    values_3 += [values_3[0]]

    # Add traces for both models
    fig.add_trace(go.Scatterpolar(
        r=values_1,
        theta=metrics_1,
        fill='toself',
        name=modelname1
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_2,
        theta=metrics_2,
        fill='toself',
        name=modelname2
    ))

    fig.add_trace(go.Scatterpolar(
        r=values_3,
        theta=metrics_3,
        fill='toself',
        name=modelname3
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.5, 1])),
        title=dict(text=f"Radar Plot - {condition} questions", x=0.5, xanchor='center'),
        showlegend=True
    )

    # Save and log radar plot
    timestamp = time.strftime("%H-%M")
    radar_html_path = f"./results/{condition}/radar_plot_{modelname1}_vs_{modelname2}_{timestamp}.html"
    fig.write_html(radar_html_path, auto_play=False)
    #wandb.log({f"Radar Plot {model_pair_1} vs {model_pair_2}": wandb.Html(radar_html_path)})

def extract_metrics_from_df(df):
    """ given a dataframe, extract the metrics data """

    metrics_data = {}
    possible_metrics = [
        "factual_correctness",
        "faithfulness",
        "answer_relevancy",
        "noise_sensitivity_relevant",
        "context_recall",
        "llm_context_precision_without_reference",
        "context_entity_recall"
    ]

    for metric in possible_metrics:
        if metric in df.columns:
            metrics_data[metric.replace("_", " ").title()] = df[metric].mean()
    
    print(f"Metrics data: {metrics_data}")

    return metrics_data


def bar_plot(rag_eval, graph_eval, cypher_eval, condition_=None):
    modelname1 = "RAG"
    modelname2 = "GRAPH"
    modelname3 = "CypherGRAPH"
    
    metrics_data_1 = extract_metrics_from_df(rag_eval)
    metrics_data_2 = extract_metrics_from_df(graph_eval)
    metrics_data_3 = extract_metrics_from_df(cypher_eval)

    fig = go.Figure()

    # Prepare data for the first model
    metrics_1 = list(metrics_data_1.keys())
    values_1 = list(metrics_data_1.values())

    # Prepare data for the second model
    metrics_2 = list(metrics_data_2.keys())
    values_2 = list(metrics_data_2.values())

    # Prepare data for the third model
    metrics_3 = list(metrics_data_3.keys())
    values_3 = list(metrics_data_3.values())

    # Add traces for both models
    fig.add_trace(go.Bar(
        x=metrics_1,
        y=values_1,
        name=modelname1
    ))

    fig.add_trace(go.Bar(
        x=metrics_2,
        y=values_2,
        name=modelname2
    ))

    fig.add_trace(go.Bar(
        x=metrics_3,
        y=values_3,
        name=modelname3
    ))

    fig.update_layout(
        barmode='group',
        title=dict(text=f"Bar Plot - {condition_}", x=0.5, xanchor='center'),
        xaxis_title="Metrics",
        yaxis_title="Scores",
        showlegend=True
    )

    # Save and log bar plot
    timestamp = time.strftime("%H-%M")
    bar_html_path = os.path.join(RESULTS_FOLDER, condition_, f"bar_plot_{modelname1}_vs_{modelname2}_vs_{modelname3}_{timestamp}.html")
    os.makedirs(os.path.dirname(bar_html_path), exist_ok=True)
    fig.write_html(bar_html_path, auto_play=True)



def multibar_plot(group1, group2, conditions):

    groups = [group1, group2]
    modelname1 = "Traditional RAG"
    modelname2 = "Naive GRAPH"
    modelname3 = "GraphCypherQA"

    fig = make_subplots(rows=1, cols=2, 
                    #plot_title_text="Query Performances (Faithfulness)",
                    shared_yaxes=True,
                    horizontal_spacing=0.006,
                    )

    for i, group in enumerate(groups):
        metrics_data_1 = extract_metrics_from_df(group[0])
        metrics_data_2 = extract_metrics_from_df(group[1])
        metrics_data_3 = extract_metrics_from_df(group[2])

        #############################################
        metrics_to_plot = ["Faithfulness"]#, "Factual Correctness"]
        #############################################
        metrics_1 = [metric for metric in metrics_data_1.keys() if metric in metrics_to_plot]
        values_1 = [metrics_data_1[metric] for metric in metrics_1]
        
        metrics_2 = [metric for metric in metrics_data_2.keys() if metric in metrics_to_plot]
        values_2 = [metrics_data_2[metric] for metric in metrics_2]
        
        metrics_3 = [metric for metric in metrics_data_3.keys() if metric in metrics_to_plot]
        values_3 = [metrics_data_3[metric] for metric in metrics_3]

        # Only show the legend for the first plot to avoid duplication
        if i == 0:
            showlegend = True
        else:
            showlegend = False

        #specify as tuple
        metrics_1 = [""]
        metrics_2 = [""]
        metrics_3 = [""]

        # Add traces for all three models
        fig.add_trace(
            go.Bar(
                x=metrics_1,
                y=values_1,
                name=modelname1,
                showlegend=showlegend,
                marker_color='#57c7e3',  # Use consistent colors across plots
            ),
            row=1, col=i+1
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_2,
                y=values_2,
                name=modelname2,
                showlegend=showlegend,
                marker_color='#ffe081'
            ),
            row=1, col=i+1
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics_3,
                y=values_3,
                name=modelname3,
                showlegend=showlegend,
                marker_color='#eba98a'
            ),
            row=1, col=i+1
        )
    
    # add space between the y-axis and the plot
    fig.update_yaxes(anchor="free",shift=-10, tickfont_size=40) #, title_text="Faithfullness", row=1, col=1)
    
    # Aggiungi annotazioni per "condizione A" e "condizione B"
    fig.add_annotation(
        text="General Queries", 
        x=0.9,  # aumenta per spostare il testo a destra
        y=-0.08,  # un po' sopra il grafico
        #y=1.1,  # un po' sopra il grafico
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=50)
    )

    fig.add_annotation(
        text="User-Specific Queries",
        x=0.12,  # posizione centrale del secondo grafico
        y=-0.08,
        #y=1.1,  # un po' sopra il grafico
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=50)
    )

    fig.update_layout(
        barmode='group',
        title=dict(text="Faithfulness Scores", x=0.5, xanchor='center'),
        title_y=0.95,
        title_yanchor='bottom',
        font=dict(size=45),
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h',
            itemsizing='constant',
            itemclick='toggleothers',  # Optional: allows toggling of other items
            itemdoubleclick='toggle',  # Optional: allows toggling of the clicked item
            font=dict(size=45),  # Adjust the font size of the legend text
        ),
        margin=dict(t=200)  # Aumenta il margine superiore per evitare che il titolo venga tagliato
    )

    fig.show()
    #Save and log bar plot
    timestamp = time.strftime("%H-%M")
    bar_html_path = os.path.join(RESULTS_FOLDER, f"barplot{modelname1}vs{modelname2}vs{modelname3}_{timestamp}.html")
    os.makedirs(os.path.dirname(bar_html_path), exist_ok=True)
    fig.write_html(bar_html_path, auto_play=True)


if __name__ == "__main__":

    # Initialize models and classes
    llm = initialize_llm()
    embeddings = initialize_embeddings()
    prompt_template = initialize_rag_prompt_template()
    knowledge_graph = KnowledgeGraph()
    llm_evaluator = LangchainLLMWrapper(AzureChatOpenAI(temperature=0, max_tokens=1024)) #1024 was breaking the correctness eva
    vector_store = create_vector_store(load_conversations(os.path.join(BASE_PATH, MEMORY_FOLDER), file_to_retrieve))

    ############################################
    ### DEFINE THE CONDITION TO EVALUATE #####
    CONDITION = "factual" #"general"
    ############################################

    # # Load json queries based on the condition
    # queries = load_queries_json(category=CONDITION)

    # # STEP 1: Answer queries for (a) RAG and (b) Naive Graph (c) CypherGraph

    # # NOTE: this functions load a responses.json, if it already exists
    # methods = ["RAG", "GRAPH", "CypherGRAPH"]
    # generate_responses(queries, vector_store, methods=["RAG", "GRAPH", "CypherGRAPH"], condition=CONDITION)

    # # convert the responses to a RAGAS dataset, we end up with a dictionary of datasets for each method
    # datasets = {}
    # for method in methods:
    #     datasets[method] = build_ragas_dataset(method)
    #     print(f"Dataset for {method} created 📄")


    # # STEP 2: Compute generation metrics for RAG and Graph
    # metrics_ = [
    #         FactualCorrectness(), 
    #         Faithfulness(), 
    #         #ResponseRelevancy(),
    #         #NoiseSensitivity(),
    #     ]

    # for method in datasets.keys():
    #     print(f"Evaluating {method}'s responses 👩‍🏫 ")
    #     # if the metrics are already computed, skip this step
    #     if os.path.exists(os.path.join(RESULTS_FOLDER, CONDITION, f"eval_{method}.csv")):
    #         print(f"\033[92m[{method}] Metrics already exist for {CONDITION} 🔄\033[0m")
    #     else:
    #         # thread_1 = threading.Thread(target=evaluate, args=(datasets[method], metrics_, llm_evaluator, embeddings))
    #         # thread_2 = threading.Thread(target=evaluate, args=(datasets[method], metrics_, llm_evaluator, embeddings))
    #         metrics = evaluate(datasets[method], metrics_, llm_evaluator, embeddings) #, run_config=RunConfig(max_workers=1, timeout=180)) # preventing rate limiting
    #         save_metrics(metrics, os.path.join(CONDITION, f"eval_{method}.csv"))

    # Read all the .csv files with the metrics and concatenate them based on the metrics
    CONDITION = "factual"
    df_rag_f = pd.read_csv(os.path.join(RESULTS_FOLDER, CONDITION, "eval_RAG.csv"))
    df_graph_f = pd.read_csv(os.path.join(RESULTS_FOLDER, CONDITION, "eval_GRAPH.csv"))
    df_cypher_f = pd.read_csv(os.path.join(RESULTS_FOLDER, CONDITION, "eval_CypherGRAPH.csv"))

    CONDITION = "general"
    df_rag_g = pd.read_csv(os.path.join(RESULTS_FOLDER, CONDITION, "eval_RAG.csv"))
    df_graph_g = pd.read_csv(os.path.join(RESULTS_FOLDER, CONDITION, "eval_GRAPH.csv"))
    df_cypher_g = pd.read_csv(os.path.join(RESULTS_FOLDER, CONDITION, "eval_CypherGRAPH.csv"))


    # # # # RADAR and BAR PLOT
    # bar_plot(df_rag_f, df_graph_f, df_cypher_f, condition_="Factual")
    # bar_plot(df_rag_g, df_graph_g, df_cypher_g, condition_="General")
    multibar_plot([df_rag_f, df_graph_f, df_cypher_f], [df_rag_g, df_graph_g, df_cypher_g], conditions=["User-Specific", "General"])

    

