"""
This script is used to ask the graph questions
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules', 'llmYogaTeacher', 'knowledge_Graph'))
from knowledge_graph import KnowledgeGraph
#from modules.llmYogaTeacher.knowledge_Graph.knowledge_graph import KnowledgeGraph
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import Faithfulness
from langchain_openai import AzureChatOpenAI
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FaithfulnesswithHHEM
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import evaluate

from ragas import EvaluationDataset
import asyncio
import glob


load_dotenv()

#read summary.txt based on the person, to use as reference for factual correctness and faithfulness
def read_summary(person):
    person_dir = os.path.join(os.path.dirname(__file__), 'yoga_tutor_memories', person)
    subdirs = [d for d in glob.glob(os.path.join(person_dir, '*')) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectory found in {person_dir}")
    summary_file = os.path.join(subdirs[0], "summary.txt")
    return open(summary_file, "r").read()


#evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(temperature=0.5, max_tokens=1024))
knowledge_graph = KnowledgeGraph()

### WE HAVE TWO TYPES OF QUESTIONS:
# 1. Questions that are about the whole graph (e.g. Who is the best yoga pratictioner based on the number of executed poses?)
# 2. Question that are specific to a person (e.g. How many poses did Hannah Bell execute)

# TEST 1: Questions that are about the whole graph
# retrieved_graph = knowledge_graph.get_whole_graph()
# answer = knowledge_graph.ask_graph(retrieved_graph, "who likes hiking ?")
# print("\033[92m NAIVE GRAPH: " + answer + "\033[0m")


answer = knowledge_graph.agentic_ask_graph("What are elena gomez interests?")
print(answer)
print("\033[92m AGENTIC GRAPH: " + answer["result"] + "\033[0m")


# TEST 2: Questions that are specific to a person
# retrieved_graph = knowledge_graph.get_personal_graph("Lily Parker")
# answer = knowledge_graph.ask_graph(retrieved_graph,"How many poses did Lily Parker succesfully completed?")
# print("\033[92m" + answer + "\033[0m")




##################### EXPERIMENT CODE #####################
# here we run in sequence the two tests above and we evaluate the faithfulness and factual correctness of the answers
############################################################

# async def QA_eval():
#     """
#     given a sample, evaluate the faithfulness of the response
#     """
#     # question = "What's Lily Parker's training level?" -> #reply: Lily Parker's training level is beginner. pero' ... {'faithfulness': 1.0000, 'factual_correctness': 0.1400}
#     question = "What is the most difficult pose for lily parker ?" #-> #reply: Lily Parker found the "Guerriero" (Warrior) pose challenging. {'faithfulness': 0.5000, 'factual_correctness': 0.1300}
#     #-> #reply: The most challenging pose for Lily Parker is "Guerriero." {'faithfulness': 1.0000, 'factual_correctness': 0.1200}

#     retrieved_graph = knowledge_graph.get_personal_graph("Lily Parker")
#     answer = knowledge_graph.ask_graph(retrieved_graph, question)
#     ground_truth = read_summary("lily parker")
#     print("\033[92m" + answer + "\033[0m")

#     sample = SingleTurnSample(
#             user_input=question,
#             response = answer,
#             retrieved_contexts=[ground_truth], #used for Faithfulness (Factual Correctness)
#             reference=ground_truth # used for Factual Correctness (coverage)
#         )
    
#     result = evaluate(dataset=EvaluationDataset(samples=[sample]),metrics=[ Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
#     print(result)


# # Esegui la funzione asincrona

# asyncio.run(QA_eval())